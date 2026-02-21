from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .config import ThoughtConfig
from .dynamic_lora import DynamicLoRAConfig, attach_dynamic_lora
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


def build_hf_setup(
    *,
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    lora_rank: int = 32,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    global_tokens: int = 32,
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
        )
    ).to(device=device, dtype=dtype)

    lora_cfg = DynamicLoRAConfig(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
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
        thought_missing = list(thought_incompat.missing_keys)
        thought_unexpected = list(thought_incompat.unexpected_keys)
        if model_missing or model_unexpected or thought_missing or thought_unexpected:
            raise RuntimeError(
                "checkpoint mismatch detected: "
                f"model_missing={len(model_missing)}, model_unexpected={len(model_unexpected)}, "
                f"thought_missing={len(thought_missing)}, thought_unexpected={len(thought_unexpected)}"
            )
    return int(payload.get("step", 0))
