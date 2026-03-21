# BESPOKE -- Training Strategy

This document is read by the Claude Code training agent. It defines what to optimize, what constraints to respect, and what hypotheses to test.

## Current Objective

Train a single generalist DoRA adapter on all curated interaction data. The goal is to validate that fine-tuning on frontier model reasoning traces produces a measurable quality improvement over the naked base model.

## Adapter: general-v1

- **Domain:** All domains (code, strategy, planning, organizing, general)
- **Minimum quality:** medium
- **Base model:** Qwen3.5-4B (Q4_K_M GGUF from `unsloth/Qwen3.5-4B-GGUF`, 4-bit MLX via `Qwen/Qwen3.5-4B` for training)

## Training Configuration

- **Phase 1 -- SFT:**
  - DoRA rank 16, learning rate 2e-4, 2 epochs
  - Target: all attention layers (q_proj, k_proj, v_proj, o_proj)
  - rsLoRA scaling enabled

- **Phase 2 -- DPO:** (when enough preference pairs exist)
  - DoRA rank 8, learning rate 5e-6, 1 epoch
  - Same target modules

## Evaluation

Compare the trained adapter against the naked base model on held-out evaluation prompts. Metrics:
- Benchmark quality scores (when benchmark is initialized)
- Manual inspection of responses to 10 diverse held-out prompts
- Training loss convergence

## Constraints

- Time budget: 30 minutes per training run
- Memory: must fit in 16GB UMA (base model Q4 + adapter overhead)
- Keep/revert: if eval doesn't improve, revert to previous best

## Current Hypotheses

These hypotheses are now tested automatically by the search grid (`bespoke train --deadline`) rather than requiring manual config changes. The search space covers all three:

1. With <100 training examples, focus on highest-quality examples only (quality_score = 'high') — tested via `min_quality: ["high", "medium"]` grid
2. If training loss plateaus quickly, try increasing rank from 16 to 32 — tested via `rank: [8, 16]` grid
3. If overfitting (eval loss rises while train loss drops), reduce epochs to 1 — tested via `epochs: [1, 2]` grid
