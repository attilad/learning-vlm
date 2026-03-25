# Prompt: Scaffold a VLM Adaptation Learning Project

Use this prompt in a fresh Claude Code session in an empty directory (e.g., `~/dev/vlm`).

---

I'm starting a new self-directed MLE learning project: **parameter-efficient adaptation of compact vision-language models on a single RTX 4090**. This is the successor to a CLIP contrastive training project where I ran 7 experiments studying training dynamics, adaptation methods (LoRA, WiSE-FT, frozen backbone), and catastrophic forgetting. Key lessons from that project that should inform this one:

- **Pretrained models dominate.** Don't train from scratch — adapt.
- **LoRA rank-4 captured 96% of full FT's gains with 0.6% of params** on CLIP. Test whether this holds for generative VLMs.
- **WiSE-FT (weight interpolation) is free lunch** — always try it after fine-tuning.
- **Forgetting evaluation must use distant OOD benchmarks**, not just same-domain holdouts. My CLIP project's CIFAR-100 eval was too close to the training distribution to detect forgetting.
- **Temperature/calibration is a high-value diagnostic** but should be read alongside other signals.
- **Isolate one variable per experiment.** Hypothesis → method → results → conclusions.

## About me

I'm an AI Architect with background in ML infrastructure, model serving, and platform architecture. I work forensically — I want to understand *why* things behave the way they do, not just that they work. Prioritize explanatory comments and observable behavior over cleverness. I like structured experiment tracking and clear diagnostic signals.

## Environment

- **OS:** WSL2 (Ubuntu 24.04) on Windows 11
- **GPU:** NVIDIA RTX 4090 (24GB VRAM, Ada Lovelace, CUDA 12.x)
- **Python:** 3.12 via `uv`
- **Package manager:** `uv` — always use `uv run` or `uv pip install`, never bare `pip`
- **Shell:** Bash inside WSL2
- **HF cache:** Use `HF_HOME=.cache/huggingface` to avoid permission issues with system cache

## The research question

**"Can a compact pretrained VLM be adapted on a single 4090 for robust structured extraction without losing OOD behavior?"**

## Model stack (don't try all at once)

**Tier 1 (4090-friendly, use these):**
- PaliGemma 3B — task-specific (captioning, VQA, doc QA, OCR). Best for disciplined, structured experiments.
- SmolVLM2-2.2B — cheapest/fastest iteration loop. Good for chat-style and later video.
- Qwen2.5-VL-3B — strongest "general VLM" baseline at this size.

**Tier 2 (borderline, cloud later):**
- Qwen2.5-VL-7B, Idefics2 8B — only with aggressive quantization or cloud hardware.

## Planned experiment sequence (8 experiments)

**Phase 0 — Hard baseline (no training)**
1. Zero-shot eval on Qwen2.5-VL-3B-Instruct and PaliGemma 3B across 4 tasks: image captioning, VQA, OCR/doc QA, domain-specific extraction. Measure accuracy, hallucination rate (100-sample manual audit), latency, VRAM, tokens/sec.

**Phase 1 — Small supervised fine-tuning (isolate one variable at a time)**
2. LoRA LR sweep on best model/task: 5e-5, 1e-4, 2e-4
3. LoRA rank sweep: r=4, 8, 16
4. Freeze strategy: projector-only vs attention-only LoRA vs attention+MLP LoRA
5. Image resolution: 224 vs 448
6. Train-set size sweep: 500 / 2K / 5K examples

**Phase 2 — Transfer and robustness**
7. Structured output (JSON extraction) vs free-form answer
8. OOD transfer test: train on one data source, eval on same-domain holdout + shifted-domain + genuinely different domain. Track in-domain score, shifted-domain score, hallucination rate.

## What to scaffold

Please set up:

1. **CLAUDE.md** — project instructions following the same style as above. Include: who I am, environment details, code style expectations (comments explain WHY, readable over clever, fail loudly with shape assertions, type hints on all signatures, no silent CPU fallback), observability standards, project structure, and what "done" looks like.

2. **pyproject.toml** — uv-managed, with: transformers, trl, peft, bitsandbytes, accelerate, torch, torchvision, datasets, Pillow, tensorboard, evaluate. Use CUDA 12.x PyTorch wheels.

3. **Project structure:**
```
vlm/
├── CLAUDE.md
├── LESSON_PLAN.md          ← 8-experiment curriculum with hypotheses
├── pyproject.toml
├── src/
│   ├── model.py            ← model loading (HF, quantization config)
│   ├── data.py             ← dataset loading/formatting for VLM input
│   ├── train.py            ← LoRA/QLoRA training loop via TRL SFTTrainer
│   ├── eval.py             ← evaluation framework (accuracy, F1, hallucination)
│   └── adapt.py            ← adaptation utilities (freeze strategies, WiSE-FT)
├── scripts/
│   ├── smoke_test.py       ← GPU + model + forward pass + generation sanity check
│   └── postmortem.py       ← training analysis / forensics
├── experiments/            ← numbered experiment dirs with notes.md
├── checkpoints/
├── runs/                   ← TensorBoard
└── data/                   ← task datasets
```

4. **LESSON_PLAN.md** — detailed curriculum with the 8-experiment sequence above, including hypotheses, what each experiment teaches, and expected outcomes. Reference CLIP project lessons where relevant.

5. **src/model.py** — model loading utility that supports:
   - Loading any HF VLM with `AutoModelForVision2Seq` or equivalent
   - BF16 and 4-bit quantization (BitsAndBytesConfig)
   - VRAM reporting
   - No silent CPU fallback

6. **scripts/smoke_test.py** — verify GPU, load a small model, run a forward pass with a dummy image, generate a few tokens. Confirm the full pipeline works before any experiment.

7. **experiments/001_baseline/notes.md** — template for the zero-shot baseline experiment.

Don't implement the full training loop or eval framework yet — just the scaffolding, model loading, and smoke test. I want to verify the environment works before writing training code.
