#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from functools import partial

import torch
import torch.distributed as dist

from pte.cache import SlidingWindowCache
from pte.config import CacheConfig, GRPOConfig, TrainConfig
from pte.data import load_prompt_answer_jsonl
from pte.hf_integration import build_hf_setup
from pte.rewards import exact_match_reward, math_equivalence_reward
from pte.rollout import CacheAwareRolloutEngine
from pte.trainer import ProgressiveThoughtTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Progressive Thought Encoding with GRPO.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--cache_window", type=int, default=1024)
    parser.add_argument("--evict_ratio", type=float, default=0.25)
    parser.add_argument("--disable_dynamic_window", action="store_true")
    parser.add_argument("--global_tokens", type=int, default=32)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--beta_kl", type=float, default=0.02)
    parser.add_argument("--max_new_tokens", type=int, default=3072)

    parser.add_argument("--reward_mode", type=str, default="math", choices=["math", "exact"])
    parser.add_argument("--reward_rtol", type=float, default=1e-4)
    parser.add_argument("--reward_atol", type=float, default=1e-8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--rollout_sub_batch_size", type=int, default=0)
    parser.add_argument("--reference_sub_batch_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--profile_steps", type=int, default=0)
    parser.add_argument("--profile_dir", type=str, default="profiles")

    parser.add_argument("--limit_samples", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_fused_adamw", action="store_true")
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--matmul_precision", type=str, default="high", choices=["highest", "high", "medium"])

    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_distributed = world_size > 1
    if use_distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return use_distributed, rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def maybe_wrap_ddp(model: torch.nn.Module, *, use_distributed: bool, local_rank: int) -> torch.nn.Module:
    if not use_distributed:
        return model
    if torch.cuda.is_available():
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    return torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)


def maybe_compile_model(model: torch.nn.Module, *, enabled: bool, mode: str) -> torch.nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    return torch.compile(model, mode=mode)  # type: ignore[arg-type]


def shard_data_by_rank(prompts: list[str], answers: list[str], rank: int, world_size: int) -> tuple[list[str], list[str]]:
    if world_size <= 1:
        return prompts, answers
    shard_prompts = prompts[rank::world_size]
    shard_answers = answers[rank::world_size]
    if not shard_prompts:
        raise RuntimeError(
            f"rank {rank} received zero samples after sharding; increase dataset size or reduce world size"
        )
    return shard_prompts, shard_answers


def main() -> None:
    args = parse_args()
    use_distributed, rank, world_size, local_rank = setup_distributed()
    main_process = is_main_process(rank)

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if use_distributed else 0)
        if use_distributed:
            torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    dtype = resolve_dtype(args.dtype)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if torch.cuda.is_available():
        allow_tf32 = not args.disable_tf32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = True

    setup = build_hf_setup(
        model_name_or_path=args.model_name_or_path,
        device=device,
        dtype=dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        global_tokens=args.global_tokens,
        load_reference=True,
    )
    if main_process:
        print(f"wrapped {len(setup.replaced_modules)} modules with DynamicLoRA")

    setup.model = maybe_compile_model(setup.model, enabled=args.compile_model, mode=args.compile_mode)
    setup.model = maybe_wrap_ddp(setup.model, use_distributed=use_distributed, local_rank=local_rank)

    prompts, answers = load_prompt_answer_jsonl(args.dataset_jsonl)
    if args.limit_samples > 0:
        prompts = prompts[: args.limit_samples]
        answers = answers[: args.limit_samples]
    prompts, answers = shard_data_by_rank(prompts, answers, rank, world_size)
    if main_process:
        print(f"loaded {len(prompts)} training samples on rank {rank} (world_size={world_size})")

    cache = SlidingWindowCache(
        CacheConfig(
            window_size=args.cache_window,
            evict_ratio=args.evict_ratio,
            keep_prompt_tokens=True,
            dynamic_window_from_prompt_max=not args.disable_dynamic_window,
        )
    )
    rollout = CacheAwareRolloutEngine(
        model=setup.model,
        tokenizer=setup.tokenizer,
        cache=cache,
        thought_encoder=setup.thought_encoder,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        eos_token_id=setup.tokenizer.eos_token_id,
        pad_token_id=int(setup.tokenizer.pad_token_id or 0),
    )

    reward_fn = (
        partial(math_equivalence_reward, rtol=args.reward_rtol, atol=args.reward_atol)
        if args.reward_mode == "math"
        else exact_match_reward
    )
    trainer = ProgressiveThoughtTrainer(
        model=setup.model,
        ref_model=setup.ref_model,
        tokenizer=setup.tokenizer,
        rollout_engine=rollout,
        thought_encoder=setup.thought_encoder,
        reward_fn=reward_fn,
        grpo_config=GRPOConfig(
            group_size=args.group_size,
            beta_kl=args.beta_kl,
            max_new_tokens=args.max_new_tokens,
        ),
        train_config=TrainConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            rollout_sub_batch_size=args.rollout_sub_batch_size,
            reference_sub_batch_size=args.reference_sub_batch_size,
            seed=args.seed + rank,
            use_amp=not args.no_amp,
            amp_dtype=args.amp_dtype,
            fused_adamw=not args.no_fused_adamw,
            profile_steps=args.profile_steps,
            profile_dir=args.profile_dir,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
        ),
        device=device,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.fit(
        prompts=prompts,
        references=answers,
        output_dir=output_dir,
        log_enabled=main_process,
        save_enabled=main_process,
    )
    if use_distributed:
        dist.barrier()
    if main_process:
        print(f"training completed, checkpoints written to {output_dir}")
    if use_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
