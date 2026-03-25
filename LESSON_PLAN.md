# Lesson Plan: Parameter-Efficient VLM Adaptation

8 experiments on a single RTX 4090. Each isolates one variable.
Lessons from the CLIP project are marked with **[CLIP]**.

---

## Phase 0 — Hard Baseline (No Training)

### Experiment 1: Zero-Shot Evaluation

**Hypothesis:** Qwen2.5-VL-3B-Instruct will outperform PaliGemma 3B on open-ended tasks (captioning, VQA) due to its chat-tuned decoder, while PaliGemma may win on structured tasks (OCR/doc QA) due to its task-specific pretraining.

**Method:**
- Load Qwen2.5-VL-3B-Instruct and PaliGemma 3B in BF16 (and 4-bit if needed for comparison)
- Evaluate on 4 tasks: image captioning, VQA, OCR/doc QA, domain-specific extraction
- Metrics: accuracy (task-appropriate), hallucination rate (100-sample manual audit per model/task), latency (ms/sample), VRAM (peak GB), throughput (tokens/sec)

**What this teaches:**
- Which model to focus adaptation experiments on
- Baseline numbers to measure improvement against
- Whether 4-bit quantization costs accuracy at this scale

**Expected outcome:** One model clearly dominates or they split by task type. The winner becomes the focus for Phase 1.

**CLIP lesson applied:** **[CLIP]** Pretrained models dominate — the zero-shot baseline may already be surprisingly strong. Don't assume fine-tuning will help until you measure.

---

## Phase 1 — Supervised Fine-Tuning (One Variable at a Time)

### Experiment 2: LoRA Learning Rate Sweep

**Hypothesis:** Optimal LoRA LR for generative VLMs will be in the 1e-4–2e-4 range (higher than full FT), similar to what worked for CLIP LoRA. Too-high LR will cause loss spikes; too-low will underfit within the small training budget.

**Method:**
- Fix: best model from Exp 1, LoRA rank=8, attention-only, 2K training examples
- Sweep: LR ∈ {5e-5, 1e-4, 2e-4}
- Train for same number of steps, log loss curves, eval on validation set

**What this teaches:**
- Sensitivity of VLM LoRA to learning rate
- Whether the "LoRA needs higher LR" pattern from CLIP transfers

**Expected outcome:** 1e-4 or 2e-4 wins. 5e-5 underfits. Loss curves reveal stability characteristics.

---

### Experiment 3: LoRA Rank Sweep

**Hypothesis:** **[CLIP]** LoRA rank-4 captured 96% of full FT gains on CLIP with 0.6% of params. For generative VLMs (which have more complex task demands than contrastive alignment), rank-8 may be the sweet spot — rank-4 might underfit on generation tasks.

**Method:**
- Fix: best LR from Exp 2, attention-only LoRA, 2K examples
- Sweep: rank ∈ {4, 8, 16}
- Compare: validation accuracy, parameter count, VRAM delta, training speed

**What this teaches:**
- Whether the "rank-4 is enough" finding from CLIP generalizes to generative VLMs
- The parameter-efficiency curve for VLMs

**Expected outcome:** Rank-8 slightly beats rank-4; rank-16 offers diminishing returns. The gap between ranks is smaller than the gap between LoRA and no-LoRA.

---

### Experiment 4: Freeze Strategy

**Hypothesis:** Attention-only LoRA will be the best default (highest accuracy per parameter), but attention+MLP LoRA may help on tasks requiring more expressive adaptation. Projector-only tuning will underperform since it can't modify the language model's generation behavior.

**Method:**
- Fix: best LR and rank from Exp 2–3, 2K examples
- Compare: (a) projector-only, (b) attention-only LoRA, (c) attention+MLP LoRA
- Track parameter count, VRAM, accuracy, and training speed for each

**What this teaches:**
- Where adaptation capacity is most valuable in a VLM
- Whether the visual projector is a bottleneck or already well-aligned

**Expected outcome:** Attention-only LoRA wins on efficiency. Attention+MLP LoRA wins on absolute accuracy but costs more VRAM. Projector-only is cheapest but worst.

---

### Experiment 5: Image Resolution

**Hypothesis:** 448px will meaningfully improve OCR/doc QA accuracy (where spatial detail matters) but offer marginal gains on captioning/VQA. The VRAM cost will roughly 4x the vision encoder's activation memory.

**Method:**
- Fix: best LoRA config from Exp 2–4, 2K examples
- Compare: 224px vs 448px input resolution
- Measure: accuracy by task, VRAM, throughput, training time

**What this teaches:**
- Whether resolution is a lever worth pulling for different task types
- The actual VRAM cost of higher resolution on a 4090

**Expected outcome:** 448px helps OCR tasks significantly (+5-10%), marginal for captioning. VRAM increase is manageable with gradient checkpointing.

---

### Experiment 6: Training Set Size Sweep

**Hypothesis:** VLM adaptation will show diminishing returns after ~2K examples for simple tasks (captioning) but continue improving up to 5K for complex tasks (structured extraction). This is because the pretrained model already "knows" captioning but needs more examples to learn new output formats.

**Method:**
- Fix: best config from Exp 2–5
- Sweep: 500 / 2,000 / 5,000 training examples
- Measure: accuracy, loss curves, signs of overfitting (train-val gap)

**What this teaches:**
- Data efficiency of VLM adaptation — how much data do you actually need?
- Whether overfitting is a practical concern at these dataset sizes

**Expected outcome:** 2K is the sweet spot for most tasks. 500 works for simple tasks. 5K helps complex extraction but shows signs of overfitting on simple tasks.

---

## Phase 2 — Transfer and Robustness

### Experiment 7: Structured Output (JSON) vs Free-Form

**Hypothesis:** Fine-tuning for JSON extraction will produce higher precision on in-domain data (because the format constrains hallucination) but may hurt free-form generation quality. The model may "forget" how to produce natural language responses.

**Method:**
- Fix: best config from Phase 1
- Compare: (a) train on free-form answers, (b) train on JSON-structured answers, for the same underlying task
- Eval: in-domain accuracy, JSON parse rate, hallucination rate, free-form quality on held-out prompts

**What this teaches:**
- Whether structured output training is complementary or competing with general capability
- Practical reliability of JSON extraction from fine-tuned VLMs

**Expected outcome:** JSON training achieves near-100% parse rate and lower hallucination (format constrains output), but degrades free-form quality. This motivates WiSE-FT as a mitigation.

---

### Experiment 8: OOD Transfer and Forgetting

**Hypothesis:** **[CLIP]** Fine-tuning will improve in-domain metrics but degrade OOD performance. WiSE-FT (interpolating fine-tuned and pretrained weights) will recover most OOD performance while retaining most in-domain gains — just as it did for CLIP.

**Method:**
- Train on one data source (e.g., document receipts)
- Eval on three tiers:
  - **In-domain holdout:** same distribution as training data
  - **Shifted-domain:** related but different (e.g., invoices instead of receipts)
  - **Distant OOD:** genuinely different domain (e.g., natural image captioning, medical VQA)
- Apply WiSE-FT: interpolate fine-tuned and pretrained weights at α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
- Track: in-domain accuracy, shifted-domain accuracy, distant-OOD accuracy, hallucination rate

**What this teaches:**
- How much forgetting occurs during VLM adaptation
- Whether WiSE-FT generalizes from CLIP (contrastive) to generative VLMs
- The right OOD evaluation protocol

**CLIP lesson applied:** **[CLIP]** CIFAR-100 eval was too close to training distribution to detect forgetting. This experiment uses a three-tier evaluation with genuinely distant OOD benchmarks.

**Expected outcome:** Fine-tuning hurts distant OOD. WiSE-FT at α=0.5–0.75 recovers most OOD performance while keeping 80%+ of in-domain gains. This validates the technique's generality beyond CLIP.

---

## Summary Table

| # | Experiment | Variable | Fixed | Key Metric |
|---|-----------|----------|-------|------------|
| 1 | Zero-shot baseline | Model choice | — | Accuracy, hallucination, VRAM |
| 2 | LR sweep | Learning rate | Rank=8, attn-only | Val accuracy, loss stability |
| 3 | Rank sweep | LoRA rank | Best LR, attn-only | Accuracy per parameter |
| 4 | Freeze strategy | What to adapt | Best LR+rank | Accuracy vs VRAM |
| 5 | Resolution | Image size | Best LoRA config | Accuracy by task, VRAM |
| 6 | Dataset size | N examples | Best config | Data efficiency curve |
| 7 | Structured output | Output format | Best config | Parse rate, hallucination |
| 8 | OOD transfer | Eval domain | Best config + WiSE-FT | Forgetting vs retention |
