from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheConfig:
    window_size: int = 1024
    evict_ratio: float = 0.25
    keep_prompt_tokens: bool = True
    min_evict_tokens: int = 1
    dynamic_window_from_prompt_max: bool = False


@dataclass(frozen=True)
class ThoughtConfig:
    hidden_size: int
    rank: int = 32
    num_global_tokens: int = 32
    normalize_eps: float = 1e-6
    adapter_scale: float = 1.0


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 4
    beta_kl: float = 0.02
    reward_eps: float = 1e-8
    temperature: float = 1.0
    max_new_tokens: int = 1024


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_steps: int = 1_000
    batch_size: int = 8
    rollout_sub_batch_size: int = 0
    reference_sub_batch_size: int = 0
    seed: int = 42
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    fused_adamw: bool = True
    profile_steps: int = 0
    profile_dir: str = "profiles"
    log_interval: int = 10
    save_interval: int = 100
