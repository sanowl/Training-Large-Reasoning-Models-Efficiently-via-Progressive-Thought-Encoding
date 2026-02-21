# Progressive Thought Encoding

Reference implementation of:
**Training Large Reasoning Models Efficiently via Progressive Thought Encoding** (arXiv:2602.16839v1).

## Overview

This project provides a PyTorch implementation of cache-constrained reinforcement learning for reasoning language models.  
The method preserves long-range reasoning signals by converting evicted KV-cache information into a compact latent thought state that dynamically modulates LoRA updates during decoding.

## Key Features

1. Cache-constrained autoregressive rollout with prompt-token retention.
2. Progressive thought-state updates from evicted key/value tensors.
3. Dynamic LoRA integration conditioned on runtime thought state.
4. Math-aware reward normalization (`\\boxed{}`, fractions, percents, equation-side extraction).
5. GRPO training with grouped reward normalization and KL regularization.
6. Distributed training (DDP), AMP policy control, optional `torch.compile`, and profiler export.
7. Throughput controls for long rollouts (dynamic cache windowing, rollout/ref sub-batching, fused AdamW).
8. Training and inference CLI entry points with unit tests for critical core components.

## Repository Structure

```text
.
├── docs/
│   └── ARCHITECTURE.md
├── examples/
│   └── sample_math.jsonl
├── scripts/
│   ├── train_grpo_pte.py
│   └── run_inference.py
├── src/pte/
│   ├── cache.py
│   ├── config.py
│   ├── data.py
│   ├── dynamic_lora.py
│   ├── grpo.py
│   ├── hf_integration.py
│   ├── rewards.py
│   ├── rollout.py
│   ├── thought_state.py
│   └── trainer.py
└── tests/
    └── test_core.py
```

## Installation

### Core

```bash
pip install -e .
```

### With Hugging Face stack

```bash
pip install -e ".[hf]"
```

## Quick Validation

Run unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
```

## Data Format

Training expects JSONL rows with:

1. `prompt` (string)
2. `answer` (string)

Example file: `examples/sample_math.jsonl`.

## Training

```bash
python3 scripts/train_grpo_pte.py \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --dataset_jsonl examples/sample_math.jsonl \
  --output_dir /tmp/pte_out \
  --cache_window 1024 \
  --evict_ratio 0.25 \
  --reward_mode math \
  --global_tokens 32 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --batch_size 8 \
  --rollout_sub_batch_size 16 \
  --reference_sub_batch_size 16 \
  --group_size 4 \
  --max_new_tokens 3072 \
  --amp_dtype bfloat16 \
  --max_steps 100
```

### Distributed training (8 GPUs)

```bash
torchrun --nproc_per_node=8 scripts/train_grpo_pte.py \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --dataset_jsonl /path/to/train.jsonl \
  --output_dir /tmp/pte_out \
  --batch_size 64 \
  --group_size 4 \
  --max_new_tokens 3072 \
  --reward_mode math \
  --compile_model
```

### Profiling

```bash
python3 scripts/train_grpo_pte.py \
  ... \
  --profile_steps 20 \
  --profile_dir traces
```

TensorBoard traces are written under `output_dir/traces`.

## Inference

```bash
python3 scripts/run_inference.py \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --checkpoint /tmp/pte_out/pte_final.pt \
  --prompt "Solve for x: 2x + 3 = 11" \
  --cache_window 1024 \
  --evict_ratio 0.25
```

## Method Mapping to Paper

1. **Cache-aware policy** (`π^D`) is implemented via constrained rollout and online cache pruning.
2. **Evicted-context encoding** is implemented in `ProgressiveThoughtEncoder`.
3. **Dynamic parameter update path** is implemented via `DynamicLoRALinear`.
4. **Grouped reward objective** is implemented with GRPO utilities in `grpo.py`.
5. **Protocol-aligned cache windowing** uses per-micro-batch prompt max (`--disable_dynamic_window` to opt out).

## Design Notes

1. Prompt lengths are tracked explicitly for cache operations.
2. Trainer buckets prompts by effective length to avoid unsafe mixed-eviction batch behavior.
3. Inference skips loading a reference model to reduce memory footprint.
4. Rollout and reference-model passes support sub-batching to manage long-sequence memory pressure.
5. Checkpoints unwrap DDP/compiled wrappers before serialization for load compatibility.
6. Checkpoint loading is strict by default to catch architecture mismatches early.

## Limitations

1. Current reward normalization is math-centric and not a general verifier for symbolic proof equivalence.
2. The implementation is a research-oriented baseline; full benchmark replication still requires large curated datasets and substantial GPU compute.
