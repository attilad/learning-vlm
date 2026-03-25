# Experiment 002: LoRA Learning Rate Sweep

## Status: RUNNING

## Hypothesis

Optimal LoRA LR for generative VLMs will be in the 1e-4–2e-4 range (higher than full FT), similar to what worked for CLIP LoRA. Too-high LR will cause loss spikes; too-low will underfit within the small training budget.

## Variable

Learning rate ∈ {5e-5, 1e-4, 2e-4}

## Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Qwen2.5-VL-3B-Instruct | Winner from Exp 001 |
| LoRA rank | 8 | Middle of planned sweep range |
| LoRA alpha | 16 | Convention: 2x rank |
| Target modules | q_proj, v_proj | Attention-only (default) |
| Training task | ChartQA | Baseline 0.770 — room for improvement |
| Training samples | 2,000 | From ChartQA train split |
| Eval samples | 500 | From ChartQA test split |
| Epochs | 3 | |
| Batch size | 2 x 8 accum = 16 effective | |
| Max length | 512 tokens | |
| Warmup | 10% of steps | |
| LR schedule | Cosine | |
| Precision | BF16 | |
| Gradient checkpointing | Yes | |

## Method

1. Load Qwen2.5-VL-3B in BF16
2. For each LR: attach LoRA → train 3 epochs → eval (generation-based relaxed accuracy)
3. Compare: training loss curves (TensorBoard), eval loss, generation accuracy, VRAM, wall time
4. Fresh model load per LR to avoid contamination

## Results

_To be filled after running._

| LR | Train Loss | Eval Loss | Gen Accuracy | VRAM (GB) | Time (s) |
|----|-----------|-----------|--------------|-----------|----------|
| 5e-5 | | | | | |
| 1e-4 | | | | | |
| 2e-4 | | | | | |

Zero-shot baseline (Exp 001): relaxed_accuracy = 0.770

## Conclusions

_To be filled after analysis._

## CLIP Lesson Check

- [ ] Does the "LoRA needs higher LR" pattern hold for generative VLMs?
- [ ] Is 5e-5 visibly underfitting (flat loss curve, no accuracy improvement)?
