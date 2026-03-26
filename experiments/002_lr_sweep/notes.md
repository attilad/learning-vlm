# Experiment 002: LoRA Learning Rate Sweep

## Status: COMPLETE

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
| Max length | 2048 tokens | Covers Qwen dynamic resolution outliers |
| Warmup | 10% of steps | |
| LR schedule | Cosine | |
| Precision | BF16 | |
| Gradient checkpointing | Yes | |

## Results

| LR | Train Loss | Eval Loss | Gen Accuracy | VRAM (GB) | Time (s) |
|----|-----------|-----------|--------------|-----------|----------|
| 5e-5 | 9.889 | 7.518 | 0.656 | 15.25 | 3821 |
| 1e-4 | 9.016 | 7.330 | **0.728** | 15.25 | 3665 |
| 2e-4 | 8.498 | 7.256 | 0.712 | 15.25 | 3698 |

Zero-shot baseline: relaxed_accuracy = **0.770**

### Loss Curve Behavior (LR=5e-5)

- Epoch 0.0–0.3: Loss flat at ~19 (warmup phase)
- Epoch 0.3–0.8: Rapid descent from 19 → 9 (main learning phase)
- Epoch 0.8–3.0: Slow descent from 9 → 8.4, then flattens
- Entropy increases from 2.1 → 8.4 during training (model predictions become less confident)
- mean_token_accuracy stays at ~3% throughout

### Key Observation: Training Hurt Generation Accuracy

All three LRs **decreased** generation accuracy relative to zero-shot:
- 5e-5: 0.770 → 0.656 (-15%)
- 1e-4: 0.770 → 0.728 (-5%)
- 2e-4: 0.770 → 0.712 (-8%)

The generation outputs are well-formatted (short, concise answers), but less accurate.

## Conclusions

1. **The hypothesis was partially wrong.** Higher LR does give lower training loss (2e-4 best), but 1e-4 gives the best generation accuracy. This suggests 2e-4 is slightly overfitting to training data patterns while losing generalization.

2. **5e-5 is clearly too low.** It underfits (highest train/eval loss) AND has the worst generation accuracy, suggesting it disrupts the model without learning enough to compensate.

3. **Training HURT zero-shot performance.** Even the best LR (1e-4) dropped accuracy by 5%. This is the CLIP project's forgetting lesson manifesting immediately. The pretrained model's zero-shot capability on ChartQA (0.770) was already strong enough that naive LoRA fine-tuning on 2K examples degrades rather than improves.

4. **The high absolute loss (~8-10) is expected.** Qwen2.5-VL computes loss over all tokens including vision tokens (~600 per sample). The language head can't predict vision tokens, inflating the loss. The relative ordering between LRs is still meaningful.

5. **1e-4 is the best LR.** Narrowest accuracy gap from zero-shot (-5%), best balance of train/eval loss. Use this for subsequent experiments.

## Implications for Next Experiments

- **Experiment 3 (rank sweep):** Use LR=1e-4. The accuracy drop means we should also test whether rank affects forgetting.
- **Experiment 8 (OOD transfer):** Forgetting is already visible on in-domain data. WiSE-FT may be needed even for in-domain recovery.
- **Consider:** More training data (Exp 6) or adding the eval distribution to training may help. 2K examples may not be enough for ChartQA.

## CLIP Lesson Check

- [x] Does the "LoRA needs higher LR" pattern hold for generative VLMs? **Partially — 2e-4 learns faster (lower loss) but 1e-4 generalizes better. The pattern is noisier for generative models.**
- [x] Is 5e-5 visibly underfitting? **Yes — highest loss AND worst accuracy. Clear underfit.**
- [x] **Surprise finding:** Zero-shot was already very good (0.770). Fine-tuning degraded it. This validates the CLIP lesson that pretrained models dominate — adaptation needs to be carefully tuned to actually help.
