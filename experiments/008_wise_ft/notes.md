# Experiment 008: WiSE-FT + OOD Transfer

## Status: NOT STARTED

## Hypothesis

**[CLIP]** Fine-tuning degrades OOD performance. WiSE-FT (interpolating fine-tuned and pretrained weights) at α=0.5–0.75 will recover most OOD performance while retaining in-domain gains — just as it did for CLIP.

Given Experiments 002-003 showed fine-tuning HURT even in-domain accuracy, WiSE-FT may also recover in-domain performance at intermediate alpha values.

## Variable

WiSE-FT interpolation coefficient α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}

- α=0.0: Pure pretrained model (no adapter contribution)
- α=1.0: Full fine-tuned model (full adapter contribution)
- Intermediate: Blended

## Fixed Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | Qwen2.5-VL-3B-Instruct | Exp 001 |
| Adapter | Rank-8, LR=1e-4 (Exp 003) | experiments/003_rank_sweep/rank_r8/final |
| Eval samples | 500 per task | |

## Three-Tier Evaluation

| Tier | Task | Dataset | Baseline (Exp 001) | Why |
|------|------|---------|-------------------|-----|
| In-domain | ChartQA | ahmed-masry/ChartQA test | 0.770 (0.796 w/ new eval) | Same distribution as training |
| Shifted-domain | DocVQA | lmms-lab/DocVQA val | 0.893 | Related document task, different format |
| Distant OOD | VQA | merve/vqav2-small val | 0.712 | Genuinely different domain (natural images) |

**[CLIP lesson]:** CIFAR-100 eval was too close to training distribution to detect forgetting. This three-tier design uses genuinely different domains.

## Method

1. Load base model + rank-8 adapter (once, reuse across alphas)
2. For each alpha: scale adapter via set_scale(), eval on all 3 tiers
3. No model reload needed — set_scale is a reversible float multiply

## Results

_To be filled after running._

| Alpha | ChartQA (in-domain) | DocVQA (shifted) | VQA (distant OOD) |
|-------|--------------------|--------------------|-------------------|
| 0.00 | | | |
| 0.25 | | | |
| 0.50 | | | |
| 0.75 | | | |
| 1.00 | | | |

## CLIP Lesson Check

- [ ] Does WiSE-FT recover OOD performance?
- [ ] Is there a "sweet spot" alpha that improves in-domain AND preserves OOD?
- [ ] Does the three-tier eval reveal forgetting that same-domain holdouts miss?
