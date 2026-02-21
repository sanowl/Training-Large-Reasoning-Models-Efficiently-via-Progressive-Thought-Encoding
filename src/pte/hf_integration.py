from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Iterable

import torch

from .config import ThoughtConfig
from .dynamic_lora import DynamicLoRAConfig, attach_dynamic_lora, dynamic_lora_signature
from .thought_state import ProgressiveThoughtEncoder


@dataclass(frozen=True)
class HFSetup:
    model: torch.nn.Module
    ref_model: torch.nn.Module | None
    tokenizer: object
    thought_encoder: ProgressiveThoughtEncoder
    replaced_modules: list[str]


def _require_hf() -> tuple[object, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "HuggingFace integration requires transformers. Install with: pip install -e '.[hf]'"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def _infer_kv_hidden_size(config: object) -> int:
    hidden_size = int(getattr(config, "hidden_size"))
    num_heads = int(getattr(config, "num_attention_heads"))
    kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))
    return kv_heads * head_dim


def _runtime_contract(model: torch.nn.Module, thought_encoder: ProgressiveThoughtEncoder) -> dict[str, object]:
    return {
        "dynamic_lora": dynamic_lora_signature(model),
        "thought_config": asdict(thought_encoder.config),
    }


def build_hf_setup(
    *,
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    lora_rank: int = 32,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    lora_interaction: str = "outer",
    lora_outer_scale: float = 1.0,
    global_tokens: int = 32,
    track_state_stats: bool = True,
    state_stats_momentum: float = 0.01,
    ood_guard_enabled: bool = True,
    ood_threshold: float = 3.0,
    ood_temperature: float = 0.5,
    ood_min_confidence: float = 0.2,
    apply_ood_guard_in_train: bool = False,
    load_reference: bool = True,
    target_module_suffixes: Iterable[str] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ),
) -> HFSetup:
    AutoModelForCausalLM, AutoTokenizer = _require_hf()

    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
    ).to(device)
    ref_model = None
    if load_reference:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
        ).to(device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    kv_hidden_size = _infer_kv_hidden_size(model.config)
    thought_encoder = ProgressiveThoughtEncoder(
        ThoughtConfig(
            hidden_size=kv_hidden_size,
            rank=lora_rank,
            num_global_tokens=global_tokens,
            track_state_stats=track_state_stats,
            state_stats_momentum=state_stats_momentum,
            ood_guard_enabled=ood_guard_enabled,
            ood_threshold=ood_threshold,
            ood_temperature=ood_temperature,
            ood_min_confidence=ood_min_confidence,
            apply_ood_guard_in_train=apply_ood_guard_in_train,
        )
    ).to(device=device, dtype=dtype)

    lora_cfg = DynamicLoRAConfig(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        interaction_mode=lora_interaction,
        outer_scale=lora_outer_scale,
        freeze_base=True,
    )
    replaced = attach_dynamic_lora(
        model,
        thought_encoder,
        target_module_suffixes=target_module_suffixes,
        config=lora_cfg,
    )
    if not replaced:
        raise RuntimeError(
            "no modules were wrapped with DynamicLoRA; check target_module_suffixes for this model architecture"
        )

    return HFSetup(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        thought_encoder=thought_encoder,
        replaced_modules=replaced,
    )


def load_pte_checkpoint(
    model: torch.nn.Module,
    thought_encoder: ProgressiveThoughtEncoder,
    checkpoint_path: str,
    *,
    allow_partial: bool = False,
) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_incompat = model.load_state_dict(payload["model"], strict=False)
    thought_incompat = thought_encoder.load_state_dict(payload["thought_encoder"], strict=False)
    if not allow_partial:
        model_missing = list(model_incompat.missing_keys)
        model_unexpected = list(model_incompat.unexpected_keys)
        optional_thought_keys = {"state_mean", "state_var", "state_stats_initialized"}
        thought_missing = [k for k in thought_incompat.missing_keys if k not in optional_thought_keys]
        thought_unexpected = [k for k in thought_incompat.unexpected_keys if k not in optional_thought_keys]
        if model_missing or model_unexpected or thought_missing or thought_unexpected:
            raise RuntimeError(
                "checkpoint mismatch detected: "
                f"model_missing={len(model_missing)}, model_unexpected={len(model_unexpected)}, "
                f"thought_missing={len(thought_missing)}, thought_unexpected={len(thought_unexpected)}"
            )
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "checkpoint missing metadata runtime contract; "
                "retrain with current code or load with allow_partial=True"
            )
        runtime_saved = metadata.get("runtime_contract")
        if not isinstance(runtime_saved, dict):
            raise RuntimeError(
                "checkpoint metadata missing runtime_contract; "
                "retrain with current code or load with allow_partial=True"
            )
        runtime_expected = _runtime_contract(model, thought_encoder)
        if runtime_saved != runtime_expected:
            raise RuntimeError(
                "checkpoint runtime contract mismatch: "
                f"saved={runtime_saved} expected={runtime_expected}. "
                "Use matching CLI/config or load with allow_partial=True."
            )
    return int(payload.get("step", 0))
