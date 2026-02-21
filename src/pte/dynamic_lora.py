from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .thought_state import ProgressiveThoughtEncoder


@dataclass(frozen=True)
class DynamicLoRAConfig:
    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.0
    freeze_base: bool = True


class DynamicLoRALinear(nn.Module):
    """
    y = xW + x(B^T) * S_e -> A^T
    Equivalent to low-rank update with dynamic middle state S_e from evicted context.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        thought_encoder: ProgressiveThoughtEncoder,
        *,
        config: DynamicLoRAConfig,
    ) -> None:
        super().__init__()
        if config.rank <= 0:
            raise ValueError(f"rank must be > 0, got {config.rank}")
        self.base = base_layer
        self.thought_encoder = thought_encoder
        self.rank = config.rank
        self.alpha = config.alpha
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_a = nn.Parameter(
            torch.empty(
                out_features,
                config.rank,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.empty(
                config.rank,
                in_features,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )
        self.reset_parameters()

        if config.freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_b, a=math.sqrt(5))
        nn.init.zeros_(self.lora_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_in = F.linear(self.dropout(x), self.lora_b)  # [..., R]
        state = self.thought_encoder.runtime_state()  # [B, R]

        if x.ndim == 2:
            if state.shape[0] != x.shape[0]:
                raise RuntimeError(
                    f"state batch {state.shape[0]} does not match input batch {x.shape[0]}"
                )
            scaled = lora_in * state
        elif x.ndim == 3:
            if state.shape[0] != x.shape[0]:
                raise RuntimeError(
                    f"state batch {state.shape[0]} does not match input batch {x.shape[0]}"
                )
            scaled = lora_in * state.unsqueeze(1)
        else:
            raise ValueError(f"unsupported input shape for DynamicLoRALinear: {tuple(x.shape)}")

        delta = F.linear(scaled, self.lora_a)
        return base_out + (self.scaling * delta)


def _split_parent(module_name: str) -> tuple[str, str]:
    if "." not in module_name:
        return "", module_name
    parent, _, child = module_name.rpartition(".")
    return parent, child


def _resolve_module(root: nn.Module, module_path: str) -> nn.Module:
    current = root
    if not module_path:
        return current
    for part in module_path.split("."):
        current = getattr(current, part)
    return current


def attach_dynamic_lora(
    model: nn.Module,
    thought_encoder: ProgressiveThoughtEncoder,
    *,
    target_module_suffixes: Iterable[str],
    config: DynamicLoRAConfig,
) -> list[str]:
    """
    Replace selected nn.Linear modules with DynamicLoRALinear.
    """
    suffixes = tuple(target_module_suffixes)
    replaced: list[str] = []
    module_names = [name for name, _ in model.named_modules()]
    for name in module_names:
        if not name:
            continue
        if not name.endswith(suffixes):
            continue
        parent_name, child_name = _split_parent(name)
        parent = _resolve_module(model, parent_name)
        child = getattr(parent, child_name)
        if not isinstance(child, nn.Linear):
            continue
        wrapped = DynamicLoRALinear(child, thought_encoder, config=config)
        setattr(parent, child_name, wrapped)
        replaced.append(name)
    return replaced


def dynamic_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, DynamicLoRALinear):
            params.extend([module.lora_a, module.lora_b])
    return params
