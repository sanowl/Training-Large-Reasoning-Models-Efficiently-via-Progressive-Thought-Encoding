from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable, Sequence

import torch
from torch import nn

from .config import GRPOConfig, TrainConfig
from .dynamic_lora import dynamic_lora_signature
from .grpo import GRPOMetrics, grpo_objective, normalize_group_rewards, sequence_logprob
from .rollout import CacheAwareRolloutEngine, RolloutBatch
from .thought_state import ProgressiveThoughtEncoder

RewardFn = Callable[[Sequence[str], Sequence[str]], torch.Tensor]


@dataclass
class TrainStepOutput:
    metrics: GRPOMetrics
    reward_mean_raw: float
    reward_mean_norm: float
    num_samples: int


class ProgressiveThoughtTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer: object,
        rollout_engine: CacheAwareRolloutEngine,
        thought_encoder: ProgressiveThoughtEncoder,
        reward_fn: RewardFn,
        grpo_config: GRPOConfig,
        train_config: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        if self.ref_model is None:
            raise ValueError("ref_model must not be None for GRPO training")
        self.tokenizer = tokenizer
        self.rollout_engine = rollout_engine
        self.thought_encoder = thought_encoder
        self.reward_fn = reward_fn
        self.grpo_config = grpo_config
        self.train_config = train_config
        self.device = device

        self.model.train().to(self.device)
        self.ref_model.eval().to(self.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False

        model_params = [p for p in self.model.parameters() if p.requires_grad]
        thought_params = [p for p in self.thought_encoder.parameters() if p.requires_grad]
        seen: set[int] = set()
        trainable: list[torch.nn.Parameter] = []
        for param in model_params + thought_params:
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            trainable.append(param)
        self._trainable_params = trainable
        if not trainable:
            raise ValueError("model has no trainable parameters")
        self.optimizer = self._build_optimizer(trainable)
        amp_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if train_config.amp_dtype not in amp_dtype_map:
            raise ValueError(f"unsupported amp_dtype={train_config.amp_dtype}")
        self._amp_dtype = amp_dtype_map[train_config.amp_dtype]
        self._use_amp = bool(train_config.use_amp and self.device.type == "cuda" and self._amp_dtype != torch.float32)
        self._scaler = None
        if self._use_amp and self._amp_dtype == torch.float16:
            self._scaler = torch.cuda.amp.GradScaler(enabled=True)
        self._step = 0

    def _build_optimizer(self, params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
        kwargs: dict[str, object] = {
            "lr": self.train_config.learning_rate,
            "weight_decay": self.train_config.weight_decay,
        }
        if self.train_config.fused_adamw and self.device.type == "cuda":
            try:
                return torch.optim.AdamW(params, fused=True, **kwargs)
            except TypeError:
                # Older torch versions may not expose fused AdamW.
                pass
        return torch.optim.AdamW(params, **kwargs)

    @staticmethod
    def _chunked_indices(indices: list[int], chunk_size: int) -> list[list[int]]:
        if chunk_size <= 0 or chunk_size >= len(indices):
            return [indices]
        return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

    def _encode_prompts(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
        )
        prompt_ids = encoded["input_ids"].to(self.device)
        prompt_attention_mask = encoded.get("attention_mask")
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        prompt_attention_mask = prompt_attention_mask.to(self.device)
        prompt_lengths = prompt_attention_mask.sum(dim=1).to(torch.long)
        return prompt_ids, prompt_attention_mask, prompt_lengths

    @torch.no_grad()
    def _reference_logprob(self, rollout: RolloutBatch) -> torch.Tensor:
        full_ids = torch.cat([rollout.prompt_ids, rollout.generated_ids], dim=1)
        full_mask = torch.cat([rollout.prompt_attention_mask, rollout.generated_mask.to(torch.long)], dim=1)

        if full_ids.shape[1] <= 1:
            return torch.zeros(full_ids.shape[0], device=full_ids.device)

        inputs = full_ids[:, :-1]
        labels = full_ids[:, 1:]
        completion_mask = torch.zeros_like(labels, dtype=torch.float32)
        prompt_len = int(rollout.prompt_ids.shape[1])
        completion_mask[:, prompt_len - 1 : prompt_len - 1 + rollout.generated_ids.shape[1]] = rollout.generated_mask

        bsz = int(inputs.shape[0])
        sub_bsz = int(self.train_config.reference_sub_batch_size)
        if sub_bsz <= 0 or sub_bsz >= bsz:
            out = self.ref_model(input_ids=inputs, attention_mask=full_mask[:, :-1], use_cache=False)
            return sequence_logprob(out.logits, labels, completion_mask)

        chunks: list[torch.Tensor] = []
        for start in range(0, bsz, sub_bsz):
            end = min(start + sub_bsz, bsz)
            out = self.ref_model(
                input_ids=inputs[start:end],
                attention_mask=full_mask[start:end, :-1],
                use_cache=False,
            )
            chunk = sequence_logprob(
                out.logits,
                labels[start:end],
                completion_mask[start:end],
            )
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)

    def _rollout_batched(self, prompts: Sequence[str]) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor]]:
        prompt_ids, prompt_mask, prompt_lengths = self._encode_prompts(prompts)
        index_buckets: dict[int, list[int]] = defaultdict(list)
        for idx, length in enumerate(prompt_lengths.tolist()):
            index_buckets[int(length)].append(idx)

        total = len(prompts)
        texts: list[str] = [""] * total
        logp_entries: list[torch.Tensor | None] = [None] * total
        ref_entries: list[torch.Tensor | None] = [None] * total

        rollout_chunk = int(self.train_config.rollout_sub_batch_size)
        for _, indices in index_buckets.items():
            for chunk_indices in self._chunked_indices(indices, rollout_chunk):
                idx_tensor = torch.tensor(chunk_indices, device=self.device, dtype=torch.long)
                batch_prompt_ids = prompt_ids.index_select(0, idx_tensor)
                batch_prompt_mask = prompt_mask.index_select(0, idx_tensor)
                batch_prompt_lengths = prompt_lengths.index_select(0, idx_tensor)

                rollout = self.rollout_engine.generate_batch(
                    batch_prompt_ids,
                    prompt_attention_mask=batch_prompt_mask,
                    prompt_lengths=batch_prompt_lengths,
                    track_grad=True,
                )
                logp = (rollout.sampled_logprobs * rollout.generated_mask).sum(dim=1)
                ref_logp = self._reference_logprob(rollout)

                for local_idx, global_idx in enumerate(chunk_indices):
                    texts[global_idx] = rollout.decoded_texts[local_idx]
                    logp_entries[global_idx] = logp[local_idx]
                    ref_entries[global_idx] = ref_logp[local_idx]

        if any(entry is None for entry in logp_entries) or any(entry is None for entry in ref_entries):
            raise RuntimeError("rollout collection failed to fill all entries")

        logp_list = [entry for entry in logp_entries if entry is not None]
        ref_list = [entry for entry in ref_entries if entry is not None]
        return texts, logp_list, ref_list

    def train_step(self, prompts: Sequence[str], references: Sequence[str]) -> TrainStepOutput:
        if len(prompts) != len(references):
            raise ValueError("prompts and references must have the same length")
        if not prompts:
            raise ValueError("empty batch")

        expanded_prompts: list[str] = []
        expanded_refs: list[str] = []
        for prompt, reference in zip(prompts, references):
            for _ in range(self.grpo_config.group_size):
                expanded_prompts.append(prompt)
                expanded_refs.append(reference)

        autocast_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._use_amp,
        )
        with autocast_ctx:
            texts, logp_list, ref_list = self._rollout_batched(expanded_prompts)
            logp_t = torch.stack(logp_list, dim=0)
            logp_ref_t = torch.stack(ref_list, dim=0)

            raw_rewards = self.reward_fn(texts, expanded_refs).to(self.device)
            group_raw = raw_rewards.view(len(prompts), self.grpo_config.group_size)
            rewards = normalize_group_rewards(group_raw, eps=self.grpo_config.reward_eps).reshape(-1)

            loss, metrics = grpo_objective(
                logp_t,
                logp_ref_t,
                rewards,
                beta_kl=self.grpo_config.beta_kl,
            )

        self.optimizer.zero_grad(set_to_none=True)
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            if self.train_config.max_grad_norm > 0:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self._trainable_params, self.train_config.max_grad_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            if self.train_config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self._trainable_params, self.train_config.max_grad_norm)
            self.optimizer.step()
        self._step += 1

        return TrainStepOutput(
            metrics=metrics,
            reward_mean_raw=float(raw_rewards.mean().item()),
            reward_mean_norm=float(rewards.mean().item()),
            num_samples=len(expanded_prompts),
        )

    def fit(
        self,
        *,
        prompts: Sequence[str],
        references: Sequence[str],
        output_dir: str | Path,
        log_enabled: bool = True,
        save_enabled: bool = True,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(prompts) != len(references):
            raise ValueError("prompts and references must have the same length")
        if not prompts:
            raise ValueError("no training samples provided")

        random.seed(self.train_config.seed)
        torch.manual_seed(self.train_config.seed)
        dataset_size = len(prompts)
        batch_size = min(max(int(self.train_config.batch_size), 1), dataset_size)
        profiler = None
        if self.train_config.profile_steps > 0:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profile_dir = output_dir / self.train_config.profile_dir
            profile_dir.mkdir(parents=True, exist_ok=True)
            profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=self.train_config.profile_steps, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            profiler.start()

        for step in range(1, self.train_config.max_steps + 1):
            batch_indices = random.sample(range(dataset_size), k=batch_size)
            batch_prompts = [prompts[idx] for idx in batch_indices]
            batch_references = [references[idx] for idx in batch_indices]
            out = self.train_step(batch_prompts, batch_references)
            if log_enabled and self.train_config.log_interval > 0 and step % self.train_config.log_interval == 0:
                print(
                    f"[step {step:05d}] loss={out.metrics.loss:.4f} "
                    f"reward(raw/norm)=({out.reward_mean_raw:.4f}/{out.reward_mean_norm:.4f})"
                )
            if save_enabled and self.train_config.save_interval > 0 and step % self.train_config.save_interval == 0:
                self.save_checkpoint(output_dir / f"pte_step_{step:05d}.pt")
            if profiler is not None:
                profiler.step()

        if profiler is not None:
            profiler.stop()
        if save_enabled:
            self.save_checkpoint(output_dir / "pte_final.pt")

    def save_checkpoint(self, path: str | Path) -> None:
        model_to_save = self.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module
        if hasattr(model_to_save, "_orig_mod"):
            model_to_save = model_to_save._orig_mod
        runtime_contract = {
            "dynamic_lora": dynamic_lora_signature(model_to_save),
            "thought_config": asdict(self.thought_encoder.config),
        }
        payload = {
            "model": model_to_save.state_dict(),
            "thought_encoder": self.thought_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self._scaler.state_dict() if self._scaler is not None else None,
            "step": self._step,
            "metadata": {
                "format_version": 2,
                "runtime_contract": runtime_contract,
            },
        }
        torch.save(payload, Path(path))
