from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .cache import SlidingWindowCache
from .thought_state import ProgressiveThoughtEncoder


@dataclass
class RolloutBatch:
    prompt_ids: torch.Tensor  # [B, Lp]
    prompt_attention_mask: torch.Tensor  # [B, Lp]
    prompt_lengths: torch.Tensor  # [B]
    generated_ids: torch.Tensor  # [B, T]
    generated_mask: torch.Tensor  # [B, T], 1 for valid sampled tokens
    sampled_logprobs: torch.Tensor  # [B, T], masked by generated_mask
    decoded_texts: list[str]
    num_eviction_events: int
    num_evicted_tokens: int

    @property
    def full_ids(self) -> torch.Tensor:
        return torch.cat([self.prompt_ids, self.generated_ids], dim=1)


class CacheAwareRolloutEngine:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        cache: SlidingWindowCache,
        thought_encoder: ProgressiveThoughtEncoder,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
        pad_token_id: int = 0,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.thought_encoder = thought_encoder
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def _sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.temperature == 0:
            tokens = torch.argmax(logits, dim=-1)
            logp = F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            return tokens, logp
        probs = torch.softmax(logits / self.temperature, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        logp = torch.log(torch.gather(probs, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))
        return tokens, logp

    def _decode_batch(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
    ) -> list[str]:
        outputs: list[str] = []
        batch_size = int(prompt_ids.shape[0])
        for idx in range(batch_size):
            prompt_valid = prompt_ids[idx][prompt_attention_mask[idx].bool()].tolist()
            gen_valid = generated_ids[idx][generated_mask[idx].bool()].tolist()
            full = prompt_valid + gen_valid
            outputs.append(self.tokenizer.decode(full, skip_special_tokens=True))
        return outputs

    def generate_batch(
        self,
        prompt_ids: torch.Tensor,
        *,
        prompt_attention_mask: torch.Tensor | None = None,
        prompt_lengths: torch.Tensor | None = None,
        track_grad: bool = False,
    ) -> RolloutBatch:
        if prompt_ids.ndim != 2:
            raise ValueError(f"prompt_ids must be [B, L], got {tuple(prompt_ids.shape)}")
        batch_size, prompt_seq_len = prompt_ids.shape
        device = prompt_ids.device

        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        else:
            if prompt_attention_mask.shape != prompt_ids.shape:
                raise ValueError(
                    f"prompt_attention_mask shape {tuple(prompt_attention_mask.shape)} "
                    f"must match prompt_ids {tuple(prompt_ids.shape)}"
                )
            prompt_attention_mask = prompt_attention_mask.to(device=device, dtype=torch.long)

        if prompt_lengths is None:
            prompt_lengths = prompt_attention_mask.sum(dim=1)
        else:
            if prompt_lengths.ndim != 1 or prompt_lengths.shape[0] != batch_size:
                raise ValueError(
                    f"prompt_lengths must be [B], got {tuple(prompt_lengths.shape)}"
                )
            prompt_lengths = prompt_lengths.to(device=device, dtype=torch.long)

        window_override = None
        if self.cache.config.dynamic_window_from_prompt_max:
            window_override = int(prompt_lengths.max().item())

        state = self.thought_encoder.init_state(
            batch_size=batch_size,
            device=device,
            dtype=self.thought_encoder.global_q.dtype,
        )
        past_key_values = None
        input_ids = prompt_ids
        full_attention_mask = torch.zeros(
            (batch_size, prompt_seq_len + self.max_new_tokens),
            dtype=torch.long,
            device=device,
        )
        full_attention_mask[:, :prompt_seq_len] = prompt_attention_mask
        current_seq_len = prompt_seq_len
        attention_mask = full_attention_mask[:, :current_seq_len]
        active = torch.ones(batch_size, dtype=torch.bool, device=device)

        generated_ids = torch.full(
            (batch_size, self.max_new_tokens),
            fill_value=self.pad_token_id,
            device=device,
            dtype=prompt_ids.dtype,
        )
        generated_mask = torch.zeros((batch_size, self.max_new_tokens), device=device, dtype=torch.float32)
        sampled_logprobs = torch.zeros((batch_size, self.max_new_tokens), device=device, dtype=torch.float32)

        num_eviction_events = 0
        num_evicted_tokens = 0

        grad_ctx = nullcontext() if track_grad else torch.inference_mode()
        with grad_ctx:
            for step in range(self.max_new_tokens):
                with self.thought_encoder.use_runtime_state(state):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                logits = outputs.logits[:, -1, :]
                next_tokens, token_logp = self._sample(logits)

                valid_mask = active.to(dtype=torch.float32)
                generated_ids[:, step] = next_tokens
                generated_mask[:, step] = valid_mask
                sampled_logprobs[:, step] = token_logp.to(torch.float32) * valid_mask

                if self.eos_token_id is not None:
                    active = active & (next_tokens != self.eos_token_id)
                if self.eos_token_id is not None:
                    next_tokens = torch.where(
                        active,
                        next_tokens,
                        torch.full_like(next_tokens, self.eos_token_id),
                    )

                past_key_values = outputs.past_key_values
                past_key_values, evicted = self.cache.prune(
                    past_key_values,
                    prompt_lengths=prompt_lengths,
                    window_size_override=window_override,
                )
                if evicted is not None and evicted.num_evicted > 0:
                    state = self.thought_encoder.update_state(
                        state,
                        evicted.keys.detach(),
                        evicted.values.detach(),
                        detach_prev_state=True,
                        evicted_tokens_per_layer=evicted.num_evicted,
                    )
                    num_eviction_events += 1
                    num_evicted_tokens += evicted.num_evicted

                input_ids = next_tokens.unsqueeze(1)
                current_seq_len += 1
                full_attention_mask[:, current_seq_len - 1] = 1
                attention_mask = full_attention_mask[:, :current_seq_len]

                if not bool(active.any().item()):
                    break

        decoded_texts = self._decode_batch(
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            generated_ids=generated_ids,
            generated_mask=generated_mask,
        )
        return RolloutBatch(
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_lengths=prompt_lengths,
            generated_ids=generated_ids,
            generated_mask=generated_mask,
            sampled_logprobs=sampled_logprobs,
            decoded_texts=decoded_texts,
            num_eviction_events=num_eviction_events,
            num_evicted_tokens=num_evicted_tokens,
        )

    def generate_group(
        self,
        prompt_ids: torch.Tensor,
        *,
        prompt_attention_mask: torch.Tensor | None = None,
        group_size: int,
        track_grad: bool = False,
    ) -> RolloutBatch:
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError(f"prompt_ids for generate_group must be [1, L], got {tuple(prompt_ids.shape)}")
        if group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {group_size}")
        prompt_ids = prompt_ids.repeat(group_size, 1)
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        else:
            prompt_attention_mask = prompt_attention_mask.repeat(group_size, 1)
        prompt_lengths = prompt_attention_mask.sum(dim=1)
        return self.generate_batch(
            prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_lengths=prompt_lengths,
            track_grad=track_grad,
        )

    def generate_one(self, prompt_ids: torch.Tensor) -> RolloutBatch:
        return self.generate_batch(prompt_ids, track_grad=False)
