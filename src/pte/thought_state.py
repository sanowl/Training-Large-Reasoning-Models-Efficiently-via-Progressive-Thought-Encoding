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

        self.config = config
        self.rank = config.rank
        self.hidden_size = config.hidden_size

        self.w_q_a = nn.Linear(config.hidden_size, config.rank, bias=False)
        self.w_k_a = nn.Linear(config.hidden_size, config.rank, bias=False)
        self.w_v_a = nn.Linear(config.hidden_size, config.rank, bias=False)

        self.global_q = nn.Parameter(torch.randn(config.num_global_tokens, config.hidden_size) * 0.02)
        self.global_k = nn.Parameter(torch.randn(config.num_global_tokens, config.hidden_size) * 0.02)
        self.global_v = nn.Parameter(torch.randn(config.num_global_tokens, config.hidden_size) * 0.02)

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
        if evicted_k.ndim != 3 or evicted_v.ndim != 3:
            raise ValueError(
                f"expected evicted tensors [B, E, H], got {tuple(evicted_k.shape)} and {tuple(evicted_v.shape)}"
            )
        if evicted_k.shape != evicted_v.shape:
            raise ValueError(
                f"evicted key/value shape mismatch: {tuple(evicted_k.shape)} vs {tuple(evicted_v.shape)}"
            )
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

    def update_state(
        self,
        state: torch.Tensor,
        evicted_k: torch.Tensor,
        evicted_v: torch.Tensor,
        *,
        detach_prev_state: bool = True,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.rank:
            raise ValueError(f"state must be [B, {self.rank}], got {tuple(state.shape)}")
        s_new = self._state_from_evicted(evicted_k, evicted_v)
        if s_new.shape != state.shape:
            raise ValueError(
                f"state batch mismatch: current state {tuple(state.shape)} vs new state {tuple(s_new.shape)}"
            )
        base = state.detach() if detach_prev_state else state
        updated = base + s_new
        return F.normalize(updated, p=2, dim=-1, eps=self.config.normalize_eps)
