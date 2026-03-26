# Experiment 003: LoRA Rank Sweep

## Status: NOT STARTED

## Hypothesis

**[CLIP]** LoRA rank-4 captured 96% of full FT gains on CLIP with 0.6% of params. For generative VLMs (more complex task demands than contrastive alignment), rank-8 may be the sweet spot — rank-4 might underfit on generation tasks, rank-16 may overfit.

## Changes from Exp 002

Before this experiment, we fixed two issues found in Exp 002:
1. **Training now randomly samples from all references** (was always using first reference, causing format overfitting)
2. **Eval normalization is more forgiving** for ChartQA (strips `%`, trailing periods)

These fixes should reduce the accuracy degradation seen in Exp 002 where all LRs hurt zero-shot performance.

## Variable

LoRA rank ∈ {4, 8, 16}

## Fixed Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | Qwen2.5-VL-3B-Instruct | Exp 001 |
| LR | 1e-4 | Exp 002 (best generation accuracy) |
| LoRA alpha | 2x rank | Convention |
| Target modules | q_proj, v_proj | Default |
| Training task | ChartQA | |
| Training samples | 2,000 | |
| Eval samples | 500 | |
| Epochs | 3 | |
| Batch size | 2 x 8 accum = 16 effective | |
| Max length | 2048 | |

## Results

_To be filled after running._

| Rank | Trainable Params | Train Loss | Eval Loss | Gen Accuracy | VRAM (GB) | Time (s) |
|------|-----------------|-----------|-----------|--------------|-----------|----------|
| 4 | | | | | | |
| 8 | | | | | | |
| 16 | | | | | | |

Zero-shot baseline: relaxed_accuracy = 0.770 (Exp 001)
Exp 002 rank-8 w/ old data: 0.728

## CLIP Lesson Check

- [ ] Does rank-4 capture most of rank-8's gains? (CLIP: 96%)
- [ ] Does rank-16 offer meaningful improvement or just overfit?
- [ ] With the data fix, does fine-tuning now IMPROVE over zero-shot?
