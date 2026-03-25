# Experiment 001: Zero-Shot Baseline

## Status: COMPLETE

## Hypothesis

Qwen2.5-VL-3B-Instruct will outperform PaliGemma 3B on open-ended tasks (captioning, VQA) due to its chat-tuned decoder, while PaliGemma may win on structured tasks (OCR/doc QA) due to task-specific pretraining.

## Variable

Model choice (this is the only experiment where we vary the model itself).

## Method

1. Loaded each model in BF16 on RTX 4090
2. Evaluated on 4 tasks, 500 samples each (fixed seed=42):
   - **Image captioning** — Flickr8k test split (jxie/flickr8k)
   - **VQA** — VQAv2-small validation (merve/vqav2-small)
   - **OCR / Document QA** — DocVQA validation (lmms-lab/DocVQA)
   - **Chart QA** — ChartQA test (ahmed-masry/ChartQA)
3. Greedy decoding (do_sample=False) for reproducibility
4. Task-appropriate max_new_tokens: captioning=80, VQA=30, DocQA=50, ChartQA=30

## Models

| Model | Params | Type | Notes |
|-------|--------|------|-------|
| Qwen2.5-VL-3B-Instruct | 3B | Chat-tuned VLM | Dynamic resolution, chat template |
| PaliGemma 3B (pt-224) | 3B | Pretrained VLM | Fixed 224px, flat prompt |

## Results

| Model | Task | Metric | Score | Med Latency (ms) | P95 Latency (ms) | VRAM (GB) | Tok/s |
|-------|------|--------|-------|-------------------|-------------------|-----------|-------|
| Qwen2.5-VL-3B | Captioning | ROUGE-L | **0.493** | 430 | 682 | 7.07 | 39.3 |
| Qwen2.5-VL-3B | VQA | VQA acc | **0.712** | 107 | 150 | 7.11 | 21.2 |
| Qwen2.5-VL-3B | DocQA | ANLS | **0.893** | 1023 | 1487 | 10.20 | 6.7 |
| Qwen2.5-VL-3B | ChartQA | Relaxed acc | **0.770** | 199 | 357 | 7.34 | 23.6 |
| PaliGemma 3B | Captioning | ROUGE-L | 0.317 | 61 | 133 | 5.49 | 55.5 |
| PaliGemma 3B | VQA | VQA acc | 0.470 | 46 | 58 | 5.49 | 43.3 |
| PaliGemma 3B | DocQA | ANLS | 0.065 | 47 | 167 | 5.49 | 54.5 |
| PaliGemma 3B | ChartQA | Relaxed acc | 0.046 | 47 | 82 | 5.49 | 46.2 |

## Conclusions

1. **Qwen dominates on every task.** The hypothesis that PaliGemma would win on structured tasks was wrong. PaliGemma's `pt-224` (pretrained, not instruction-tuned) variant doesn't understand the QA format — it outputs fragmented text rather than answers for DocQA and ChartQA.

2. **The gap is largest on structured tasks.** Qwen beats PaliGemma by ~50% on captioning/VQA but by 14–17x on DocQA/ChartQA. This is likely instruction tuning, not just model capability — PaliGemma-pt hasn't been taught to answer questions.

3. **Qwen's zero-shot DocQA (ANLS 0.893) is remarkably strong.** This validates the CLIP lesson: pretrained models dominate. Fine-tuning on DocQA may have limited upside.

4. **PaliGemma is 2x faster and uses 1.5–5GB less VRAM**, but accuracy is too low for the gap to matter. Speed advantage is meaningless without quality.

5. **Qwen's DocQA latency (1023ms median) is 5–10x higher** than other tasks. This is because DocVQA images are high-resolution documents (1787x2286px) and Qwen2.5-VL processes them at native resolution with dynamic token counts. This is a real-world concern for serving.

## Decision

**Qwen2.5-VL-3B-Instruct is the focus model for Phase 1.**

Rationale: It wins every task by a large margin. The 7–10GB VRAM range leaves ample headroom on a 24GB 4090 for LoRA training with gradient checkpointing.

PaliGemma-pt-224 is not a fair comparison — using PaliGemma-mix (instruction-tuned) would have been more appropriate. However, Qwen's instruction-tuning advantage is exactly what we'd be building on top of in Phase 1, so it's the right choice regardless.

## CLIP Lesson Check

- [x] Did the zero-shot baseline surprise us? **Yes — Qwen's DocQA ANLS of 0.893 zero-shot is very strong. Fine-tuning may have limited upside on this task.**
- [x] Is the gap between models large enough to matter? **Yes, decisively. Qwen wins on all tasks.**
