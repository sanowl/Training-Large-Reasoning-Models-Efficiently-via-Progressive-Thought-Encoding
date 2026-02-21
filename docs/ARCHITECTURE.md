# Architecture Overview

This document summarizes how the implementation maps to the paper:
`Training Large Reasoning Models Efficiently via Progressive Thought Encoding` (arXiv:2602.16839v1).

## Core Design

The project has four major subsystems:

1. **Progressive Thought State**
Path: `src/pte/thought_state.py`  
Maintains a compact latent state `S_e` derived from evicted KV tensors and initialized with learnable global tokens.

2. **Dynamic LoRA Path**
Path: `src/pte/dynamic_lora.py`  
Wraps selected `nn.Linear` modules and injects a low-rank update modulated by runtime thought state.

3. **Cache-Constrained Rollout**
Path: `src/pte/cache.py`, `src/pte/rollout.py`  
Performs autoregressive decoding with bounded cache, prompt-token retention, and eviction-based state updates.

4. **GRPO Training**
Path: `src/pte/grpo.py`, `src/pte/trainer.py`  
Implements grouped reward normalization and KL-regularized policy updates against a frozen reference model, with rollout/reference sub-batching and AMP support.

## Data Flow Per Training Step

1. Tokenize a prompt batch.
2. Expand each prompt into a group of rollouts (`group_size`).
3. Generate completions under constrained cache.
4. When eviction happens, update `S_e` using evicted KV summary.
5. Compute sampled log-probabilities from rollout directly (no replay pass).
6. Compute reference log-probabilities from the frozen model.
7. Normalize group rewards and apply GRPO objective.
8. Backpropagate to Dynamic LoRA + thought encoder parameters.

## Notes on Batch Handling

- Cache pruning accepts per-sample prompt lengths.
- If mixed prompt lengths produce different eviction counts at the same step, the cache layer raises a clear error.
- Trainer handles this by bucketing prompts by effective prompt length before rollout.
- Dynamic cache window mode can set window size to the maximum prompt length in each micro-batch.

## Entry Points

- Train: `scripts/train_grpo_pte.py`
- Inference: `scripts/run_inference.py`

## Known Limits

- Reward normalization is tuned for math-answer equivalence and does not replace a full symbolic verifier.
- Benchmark-scale replication still depends on multi-GPU hardware and external datasets.
