#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from pte.cache import SlidingWindowCache
from pte.config import CacheConfig
from pte.hf_integration import build_hf_setup, load_pte_checkpoint
from pte.rollout import CacheAwareRolloutEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cache-constrained inference with Progressive Thought Encoding.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--cache_window", type=int, default=1024)
    parser.add_argument("--evict_ratio", type=float, default=0.25)
    parser.add_argument("--global_tokens", type=int, default=32)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)

    setup = build_hf_setup(
        model_name_or_path=args.model_name_or_path,
        device=device,
        dtype=dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        global_tokens=args.global_tokens,
        load_reference=False,
    )
    step = load_pte_checkpoint(setup.model, setup.thought_encoder, args.checkpoint)
    print(f"loaded checkpoint step={step}")

    cache = SlidingWindowCache(
        CacheConfig(
            window_size=args.cache_window,
            evict_ratio=args.evict_ratio,
            keep_prompt_tokens=True,
        )
    )
    engine = CacheAwareRolloutEngine(
        model=setup.model,
        tokenizer=setup.tokenizer,
        cache=cache,
        thought_encoder=setup.thought_encoder,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        eos_token_id=setup.tokenizer.eos_token_id,
        pad_token_id=int(setup.tokenizer.pad_token_id or 0),
    )

    prompt_ids = setup.tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    out = engine.generate_one(prompt_ids)
    print("--- prompt ---")
    print(args.prompt)
    print("--- completion ---")
    completion_ids = out.generated_ids[0][out.generated_mask[0].bool()].tolist()
    print(setup.tokenizer.decode(completion_ids, skip_special_tokens=True))
    print("--- full output ---")
    print(out.decoded_texts[0])
    print(
        f"eviction_events={out.num_eviction_events} "
        f"evicted_tokens={out.num_evicted_tokens}"
    )


if __name__ == "__main__":
    main()
