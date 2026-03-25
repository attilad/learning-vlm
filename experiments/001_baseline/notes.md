# Experiment 001: Zero-Shot Baseline

## Status: NOT STARTED

## Hypothesis

Qwen2.5-VL-3B-Instruct will outperform PaliGemma 3B on open-ended tasks (captioning, VQA) due to its chat-tuned decoder, while PaliGemma may win on structured tasks (OCR/doc QA) due to task-specific pretraining.

## Variable

Model choice (this is the only experiment where we vary the model itself).

## Method

1. Load each model in BF16 (and optionally 4-bit for VRAM comparison)
2. Evaluate on 4 task types:
   - **Image captioning** — dataset TBD (COCO captions subset?)
   - **VQA** — dataset TBD (VQAv2 subset?)
   - **OCR / Document QA** — dataset TBD (DocVQA subset?)
   - **Domain-specific extraction** — dataset TBD
3. Metrics per model per task:
   - Accuracy (task-appropriate: CIDEr for captioning, exact-match/F1 for QA)
   - Hallucination rate (100-sample manual audit per model/task)
   - Latency (ms/sample, median over eval set)
   - VRAM (peak allocated GB)
   - Throughput (tokens/sec)

## Models

| Model | Params | Type | Notes |
|-------|--------|------|-------|
| Qwen2.5-VL-3B-Instruct | 3B | Chat-tuned VLM | Strong general baseline |
| PaliGemma 3B (224px) | 3B | Task-specific VLM | Pretrained for captioning/VQA/OCR |

## Datasets

TBD — select before running. Requirements:
- Public, HuggingFace-hosted for reproducibility
- Small enough for fast eval (~500-1000 samples per task)
- Diverse enough to expose model differences

## Results

_To be filled after running._

| Model | Task | Accuracy | Hallucination % | Latency (ms) | VRAM (GB) | Tokens/s |
|-------|------|----------|-----------------|---------------|-----------|----------|
| | | | | | | |

## Conclusions

_To be filled after analysis._

## Decision

_Which model becomes the focus for Phase 1? Why?_

## CLIP Lesson Check

- [ ] Did the zero-shot baseline surprise us? (CLIP lesson: pretrained models dominate)
- [ ] Is the gap between models large enough to matter, or should we just pick the cheaper one?
