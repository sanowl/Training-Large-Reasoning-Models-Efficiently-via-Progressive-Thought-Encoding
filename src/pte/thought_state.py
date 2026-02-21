from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ThoughtConfig


class ProgressiveThoughtEncoder(nn.Module):
    """
    Computes progressive evicted-context state S_e used to modulate LoRA weights.
    Runtime state is explicitly provided by callers through a scoped context.
    """

    def __init__(self, config: ThoughtConfig) -> None:
        super().__init__()
        if config.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {config.hidden_size}")
        if config.rank <= 0:
            raise ValueError(f"rank must be positive, got {config.rank}")
        if config.num_global_tokens <= 0:
            raise ValueError(f"num_global_tokens must be positive, got {config.num_global_tokens}")
        if config.layer_attn_temperature <= 0:
            raise ValueError(f"layer_attn_temperature must be > 0, got {config.layer_attn_temperature}")
        if not 0.0 <= config.state_decay < 1.0:
            raise ValueError(f"state_decay must be in [0, 1), got {config.state_decay}")
        if config.max_state_norm <= 0:
            raise ValueError(f"max_state_norm must be > 0, got {config.max_state_norm}")
        if not 0.0 < config.state_stats_momentum <= 1.0:
            raise ValueError(
                f"state_stats_momentum must be in (0, 1], got {config.state_stats_momentum}"
            )
        if config.ood_threshold < 0:
            raise ValueError(f"ood_threshold must be >= 0, got {config.ood_threshold}")
        if config.ood_temperature <= 0:
            raise ValueError(f"ood_temperature must be > 0, got {config.ood_temperature}")
        if not 0.0 <= config.ood_min_confidence <= 1.0:
            raise ValueError(
                f"ood_min_confidence must be in [0, 1], got {config.ood_min_confidence}"
            )
        if not 0.0 <= config.min_update_gate <= config.max_update_gate <= 1.0:
            raise ValueError(
                "update gates must satisfy 0 <= min_update_gate <= max_update_gate <= 1, "
                f"got min={config.min_update_gate}, max={config.max_update_gate}"
            )
        if config.evidence_tokens_scale <= 0:
            raise ValueError(f"evidence_tokens_scale must be > 0, got {config.evidence_tokens_scale}")

        self.config = config
        self.rank = config.rank
        self.hidden_size = config.hidden_size

        self.w_q_a = nn.Linear(config.hidden_size, config.rank, bias=False)
        self.w_k_a = nn.Linear(config.hidden_size, config.rank, bias=False)
        self.w_v_a = nn.Linear(config.hidden_size, config.rank, bias=False)

        self.global_q = nn.Parameter(torch.randn(config.num_global_tokens, config.hidden_size) * 0.02)
        self.global_k = nn.Parameter(torch.randn(config.num_global_tokens, config.hidden_size) * 0.02)
        self.global_v = nn.Parameter(torch.randn(config.num_global_tokens, config.hidden_size) * 0.02)
        if config.use_layer_attention:
            self.layer_query = nn.Parameter(torch.randn(config.rank) * 0.02)
        else:
            self.register_parameter("layer_query", None)
        self.register_buffer("state_mean", torch.zeros(config.rank, dtype=torch.float32))
        self.register_buffer("state_var", torch.ones(config.rank, dtype=torch.float32))
        self.register_buffer("state_stats_initialized", torch.tensor(False, dtype=torch.bool))

        self._runtime_state: torch.Tensor | None = None

    def runtime_state(self) -> torch.Tensor:
        if self._runtime_state is None:
            raise RuntimeError("runtime thought state is not set; use thought_encoder.use_runtime_state(state)")
        return self._runtime_state

    @contextmanager
    def use_runtime_state(self, state: torch.Tensor):
        previous = self._runtime_state
        self._runtime_state = state
        try:
            yield
        finally:
            self._runtime_state = previous

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        device = device or self.global_q.device
        dtype = dtype or self.global_q.dtype

        q = self.w_q_a(self.global_q.to(device=device, dtype=dtype))  # [G, R]
        k = self.w_k_a(self.global_k.to(device=device, dtype=dtype))  # [G, R]
        v = self.w_v_a(self.global_v.to(device=device, dtype=dtype))  # [G, R]

        # Initialization from learnable global tokens before any eviction event.
        s0 = (q @ k.transpose(0, 1)) @ v  # [G, R]
        s0 = s0.mean(dim=0)  # [R]
        s0 = F.normalize(s0, p=2, dim=0, eps=self.config.normalize_eps)
        return s0.unsqueeze(0).repeat(batch_size, 1)

    def _state_from_evicted(self, evicted_k: torch.Tensor, evicted_v: torch.Tensor) -> torch.Tensor:
        if evicted_k.ndim not in (3, 4) or evicted_v.ndim not in (3, 4):
            raise ValueError(
                "expected evicted tensors [B, E, H] or [B, L, E, H], "
                f"got {tuple(evicted_k.shape)} and {tuple(evicted_v.shape)}"
            )
        if evicted_k.shape != evicted_v.shape:
            raise ValueError(
                f"evicted key/value shape mismatch: {tuple(evicted_k.shape)} vs {tuple(evicted_v.shape)}"
            )

        if evicted_k.ndim == 4:
            batch, num_layers, evicted_tokens, hidden_size = evicted_k.shape
            if num_layers <= 0:
                raise ValueError(f"num_layers must be > 0, got {num_layers}")
            if self.config.use_layer_attention:
                layer_k = evicted_k.mean(dim=2)  # [B, L, H]
                layer_scores = torch.einsum(
                    "blr,r->bl",
                    self.w_k_a(layer_k),
                    self.layer_query.to(device=evicted_k.device, dtype=evicted_k.dtype),
                ) / (math.sqrt(float(self.rank)) * self.config.layer_attn_temperature)
                layer_weights = torch.softmax(layer_scores, dim=1)  # [B, L]
            else:
                layer_weights = torch.full(
                    (batch, num_layers),
                    1.0 / float(num_layers),
                    device=evicted_k.device,
                    dtype=evicted_k.dtype,
                )
            evicted_k = (evicted_k * layer_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [B, E, H]
            evicted_v = (evicted_v * layer_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [B, E, H]
        batch, evicted_tokens, hidden_size = evicted_k.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"evicted hidden size {hidden_size} does not match encoder hidden size {self.hidden_size}"
            )
        if evicted_tokens == 0:
            return torch.zeros(batch, self.rank, device=evicted_k.device, dtype=evicted_k.dtype)

        q = self.w_q_a(self.global_q.to(device=evicted_k.device, dtype=evicted_k.dtype))  # [G, R]
        q = q.unsqueeze(0).expand(batch, -1, -1)  # [B, G, R]
        k = self.w_k_a(evicted_k)  # [B, E, R]
        v = self.w_v_a(evicted_v)  # [B, E, R]

        # Compact state from evicted key/value context.
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(float(self.rank))  # [B, G, E]
        attn = torch.softmax(scores, dim=-1)
        context = attn @ v  # [B, G, R]
        return context.mean(dim=1)  # [B, R]

    def _compute_update_gate(
        self,
        state: torch.Tensor,
        s_new: torch.Tensor,
        *,
        evicted_tokens_per_layer: int | None = None,
    ) -> torch.Tensor:
        state_n = F.normalize(state, p=2, dim=-1, eps=self.config.normalize_eps)
        s_new_n = F.normalize(s_new, p=2, dim=-1, eps=self.config.normalize_eps)
        similarity = F.cosine_similarity(state_n, s_new_n, dim=-1, eps=self.config.normalize_eps).clamp(-1.0, 1.0)
        sim_gate = 0.5 * (similarity + 1.0)  # map [-1, 1] -> [0, 1]

        gate = self.config.min_update_gate + (self.config.max_update_gate - self.config.min_update_gate) * sim_gate
        if evicted_tokens_per_layer is not None:
            evidence = min(1.0, float(evicted_tokens_per_layer) / float(self.config.evidence_tokens_scale))
            gate = gate * evidence
        return gate.unsqueeze(-1)  # [B, 1]

    def _update_state_stats(self, state: torch.Tensor) -> None:
        if not self.config.track_state_stats:
            return
        x = state.detach().to(dtype=torch.float32)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False).clamp_min(self.config.normalize_eps)
        if not bool(self.state_stats_initialized.item()):
            self.state_mean.copy_(batch_mean)
            self.state_var.copy_(batch_var)
            self.state_stats_initialized.fill_(True)
            return
        momentum = self.config.state_stats_momentum
        self.state_mean.mul_(1.0 - momentum).add_(batch_mean, alpha=momentum)
        self.state_var.mul_(1.0 - momentum).add_(batch_var, alpha=momentum)

    def state_confidence(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.rank:
            raise ValueError(f"state must be [B, {self.rank}], got {tuple(state.shape)}")
        if not self.config.ood_guard_enabled or not bool(self.state_stats_initialized.item()):
            return torch.ones((state.shape[0], 1), device=state.device, dtype=state.dtype)

        mean = self.state_mean.to(device=state.device, dtype=torch.float32)
        var = self.state_var.to(device=state.device, dtype=torch.float32).clamp_min(self.config.normalize_eps)
        z = (state.to(dtype=torch.float32) - mean).abs() / torch.sqrt(var)
        score = z.mean(dim=-1)  # [B]
        raw = torch.sigmoid((self.config.ood_threshold - score) / self.config.ood_temperature)
        conf = self.config.ood_min_confidence + (1.0 - self.config.ood_min_confidence) * raw
        return conf.unsqueeze(-1).to(dtype=state.dtype)

    def adaptation_confidence(self, state: torch.Tensor) -> torch.Tensor:
        if self.training and not self.config.apply_ood_guard_in_train:
            return torch.ones((state.shape[0], 1), device=state.device, dtype=state.dtype)
        return self.state_confidence(state)

    def update_state(
        self,
        state: torch.Tensor,
        evicted_k: torch.Tensor,
        evicted_v: torch.Tensor,
        *,
        detach_prev_state: bool = True,
        evicted_tokens_per_layer: int | None = None,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.rank:
            raise ValueError(f"state must be [B, {self.rank}], got {tuple(state.shape)}")
        s_new = self._state_from_evicted(evicted_k, evicted_v)
        if s_new.shape != state.shape:
            raise ValueError(
                f"state batch mismatch: current state {tuple(state.shape)} vs new state {tuple(s_new.shape)}"
            )
        base = state.detach() if detach_prev_state else state
        gate = self._compute_update_gate(
            base,
            s_new,
            evicted_tokens_per_layer=evicted_tokens_per_layer,
        )
        base_scaled = (1.0 - self.config.state_decay) * base
        updated = base_scaled + gate * s_new

        # Preserve confidence magnitude while preventing exploding state norms.
        norm = torch.norm(updated, dim=-1, keepdim=True).clamp_min(self.config.normalize_eps)
        scale = torch.clamp(norm / self.config.max_state_norm, min=1.0)
        bounded = updated / scale
        if self.training:
            self._update_state_stats(bounded)
        return bounded
