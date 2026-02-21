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
    use_layer_attention: bool = True
    layer_attn_temperature: float = 1.0
    normalize_eps: float = 1e-6
    adapter_scale: float = 1.0
    max_state_norm: float = 4.0
    state_decay: float = 0.01
    min_update_gate: float = 0.05
    max_update_gate: float = 0.30
    evidence_tokens_scale: float = 8.0
    use_token_confidence_gate: bool = True
    min_token_confidence: float = 0.05
    track_state_stats: bool = True
    state_stats_momentum: float = 0.01
    ood_guard_enabled: bool = True
    ood_threshold: float = 3.0
    ood_temperature: float = 0.5
    ood_min_confidence: float = 0.2
    apply_ood_guard_in_train: bool = False


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
