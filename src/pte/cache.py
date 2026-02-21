from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .config import CacheConfig

PastKeyValues = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class EvictedKV:
    keys: torch.Tensor  # [B, E, H]
    values: torch.Tensor  # [B, E, H]
    num_evicted: int


class SlidingWindowCache:
    """
    Prompt-anchored sliding-window cache with eviction ratio.
    """

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        if config.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {config.window_size}")
        if not 0.0 < config.evict_ratio <= 1.0:
            raise ValueError(f"evict_ratio must be in (0, 1], got {config.evict_ratio}")
        if config.min_evict_tokens <= 0:
            raise ValueError(f"min_evict_tokens must be > 0, got {config.min_evict_tokens}")

    def _evict_count(self, reasoning_tokens: int, *, window_size: int | None = None) -> int:
        effective_window = int(window_size if window_size is not None else self.config.window_size)
        if effective_window <= 0:
            raise ValueError(f"effective window size must be > 0, got {effective_window}")
        overflow = reasoning_tokens - effective_window
        if overflow <= 0:
            return 0
        ratio_count = int(math.ceil(effective_window * self.config.evict_ratio))
        return max(overflow, ratio_count, self.config.min_evict_tokens)

    @staticmethod
    def _summarize_evicted(layers: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> EvictedKV:
        if not layers:
            raise ValueError("cannot summarize empty evicted layer list")
        key_list: list[torch.Tensor] = []
        value_list: list[torch.Tensor] = []
        for key, value in layers:
            # [B, Heads, E, Dim] -> [B, E, Heads*Dim]
            key_list.append(key.permute(0, 2, 1, 3).flatten(start_dim=2))
            value_list.append(value.permute(0, 2, 1, 3).flatten(start_dim=2))
        keys = torch.stack(key_list, dim=0).mean(dim=0)
        values = torch.stack(value_list, dim=0).mean(dim=0)
        num_evicted = int(keys.shape[1])
        return EvictedKV(keys=keys, values=values, num_evicted=num_evicted)

    @staticmethod
    def _to_legacy_cache(past_key_values: Any) -> tuple[PastKeyValues, Any | None]:
        if isinstance(past_key_values, tuple):
            return past_key_values, None
        if hasattr(past_key_values, "to_legacy_cache"):
            legacy = past_key_values.to_legacy_cache()
            return legacy, type(past_key_values)
        raise TypeError(
            "unsupported past_key_values type; expected tuple or cache object with to_legacy_cache()"
        )

    @staticmethod
    def _from_legacy_cache(legacy_cache: PastKeyValues, cache_type: Any | None) -> Any:
        if cache_type is None:
            return legacy_cache
        if hasattr(cache_type, "from_legacy_cache"):
            return cache_type.from_legacy_cache(legacy_cache)
        return legacy_cache

    def prune(
        self,
        past_key_values: Any,
        *,
        prompt_lengths: int | Sequence[int] | torch.Tensor,
        window_size_override: int | None = None,
    ) -> tuple[Any, EvictedKV | None]:
        if not past_key_values:
            return past_key_values, None

        legacy_cache, cache_type = self._to_legacy_cache(past_key_values)

        sample_key = legacy_cache[0][0]
        if sample_key.ndim != 4:
            raise ValueError(f"expected cache key shape [B,H,T,D], got {tuple(sample_key.shape)}")
        batch_size, _, seq_len, _ = sample_key.shape

        if isinstance(prompt_lengths, int):
            prompt_tensor = torch.full((batch_size,), int(prompt_lengths), dtype=torch.long, device=sample_key.device)
        elif isinstance(prompt_lengths, torch.Tensor):
            prompt_tensor = prompt_lengths.to(device=sample_key.device, dtype=torch.long)
        else:
            prompt_tensor = torch.tensor(list(prompt_lengths), dtype=torch.long, device=sample_key.device)

        if prompt_tensor.ndim != 1 or prompt_tensor.shape[0] != batch_size:
            raise ValueError(
                f"prompt_lengths must be shape [{batch_size}], got {tuple(prompt_tensor.shape)}"
            )
        if torch.any(prompt_tensor < 0):
            raise ValueError(f"prompt_lengths must be >= 0, got {prompt_tensor.tolist()}")

        if self.config.keep_prompt_tokens:
            prompt_kept = torch.clamp(prompt_tensor, min=0, max=seq_len)
        else:
            prompt_kept = torch.zeros_like(prompt_tensor)

        reasoning_tokens = seq_len - prompt_kept
        evict_counts = torch.tensor(
            [
                self._evict_count(
                    int(v.item()),
                    window_size=window_size_override,
                )
                for v in reasoning_tokens
            ],
            dtype=torch.long,
            device=sample_key.device,
        )
        max_evict = int(evict_counts.max().item())
        if max_evict <= 0:
            return past_key_values, None
        if int(evict_counts.min().item()) != max_evict:
            raise ValueError(
                "batch has different eviction counts per sample; bucket prompts by length before batched rollout. "
                f"evict_counts={evict_counts.tolist()}"
            )

        start = prompt_kept
        end = prompt_kept + max_evict
        new_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        evicted_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for key, value in legacy_cache:
            if key.shape != value.shape:
                raise ValueError(f"key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}")
            kept_keys: list[torch.Tensor] = []
            kept_values: list[torch.Tensor] = []
            evicted_keys: list[torch.Tensor] = []
            evicted_values: list[torch.Tensor] = []
            for batch_idx in range(batch_size):
                sample_start = int(start[batch_idx].item())
                sample_end = int(end[batch_idx].item())
                key_b = key[batch_idx : batch_idx + 1]
                value_b = value[batch_idx : batch_idx + 1]
                evicted_key = key_b[:, :, sample_start:sample_end, :]
                evicted_value = value_b[:, :, sample_start:sample_end, :]
                kept_key = torch.cat(
                    [key_b[:, :, :sample_start, :], key_b[:, :, sample_end:, :]],
                    dim=2,
                )
                kept_value = torch.cat(
                    [value_b[:, :, :sample_start, :], value_b[:, :, sample_end:, :]],
                    dim=2,
                )
                kept_keys.append(kept_key)
                kept_values.append(kept_value)
                evicted_keys.append(evicted_key)
                evicted_values.append(evicted_value)

            new_layers.append((torch.cat(kept_keys, dim=0), torch.cat(kept_values, dim=0)))
            evicted_layers.append((torch.cat(evicted_keys, dim=0), torch.cat(evicted_values, dim=0)))

        summary = self._summarize_evicted(evicted_layers)
        return self._from_legacy_cache(tuple(new_layers), cache_type), summary
