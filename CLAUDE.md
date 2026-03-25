# VLM Parameter-Efficient Adaptation Project

## Who

AI Architect studying parameter-efficient adaptation of compact vision-language models.
Successor to a CLIP contrastive training project (7 experiments on training dynamics, LoRA, WiSE-FT, catastrophic forgetting).

## Research Question

"Can a compact pretrained VLM be adapted on a single 4090 for robust structured extraction without losing OOD behavior?"

## Environment

- **OS:** WSL2 (Ubuntu 24.04) on Windows 11
- **GPU:** NVIDIA RTX 4090 (24GB VRAM, Ada Lovelace, CUDA 12.x)
- **Python:** 3.12 via `uv`
- **Package manager:** `uv` only — use `uv run`, `uv pip install`, never bare `pip`
- **HF cache:** Always set `HF_HOME=.cache/huggingface` (relative to project root) to avoid WSL permission issues
- **PyTorch:** CUDA 12.x wheels via `--extra-index-url`

## Code Style

- **Comments explain WHY**, not what. No boilerplate docstrings.
- **Readable over clever.** No one-liner comprehensions that need a diagram to parse.
- **Fail loudly.** Shape assertions at tensor boundaries. Never silently reshape.
- **No silent CPU fallback.** If CUDA is unavailable, crash with an explicit error message.
- **Type hints on all function signatures.** No `Any` unless genuinely unavoidable.
- **Imports:** Group stdlib → third-party → local. One blank line between groups.

## Observability Standards

- Every training run logs to TensorBoard under `runs/`.
- Log VRAM usage, tokens/sec, and wall-clock time for every experiment.
- Temperature/calibration is a diagnostic — read alongside accuracy and hallucination rate.
- Forgetting evaluation uses distant OOD benchmarks, not same-domain holdouts.

## Experiment Discipline

- **Isolate one variable per experiment.** Hypothesis → method → results → conclusions.
- Each experiment lives in `experiments/NNN_name/` with a `notes.md`.
- Never train from scratch — always adapt pretrained models.
- Always try WiSE-FT (weight interpolation) after fine-tuning.

## Project Structure

```
vlm/
├── CLAUDE.md               ← you are here
├── LESSON_PLAN.md           ← 8-experiment curriculum
├── pyproject.toml
├── src/
│   ├── model.py             ← model loading (HF, quantization config)
│   ├── data.py              ← dataset loading/formatting for VLM input
│   ├── train.py             ← LoRA/QLoRA training loop via TRL SFTTrainer
│   ├── eval.py              ← evaluation framework
│   └── adapt.py             ← adaptation utilities (freeze, WiSE-FT)
├── scripts/
│   ├── smoke_test.py        ← GPU + model + forward pass sanity check
│   └── postmortem.py        ← training analysis / forensics
├── experiments/             ← numbered experiment dirs
├── checkpoints/
├── runs/                    ← TensorBoard logs
└── data/                    ← task datasets
```

## What "Done" Looks Like

1. All 8 experiments completed with clear conclusions in each `notes.md`.
2. A final summary comparing adaptation methods across tasks.
3. Quantified answers: Does LoRA rank-4 generalize from CLIP to generative VLMs? Does WiSE-FT help? What freezing strategy works? Does structured output hurt OOD transfer?
4. Reproducible — any experiment can be re-run from its config.

## Key Lessons from CLIP Project (Carry Forward)

- Pretrained models dominate. Don't train from scratch — adapt.
- LoRA rank-4 captured 96% of full FT's gains with 0.6% of params on CLIP.
- WiSE-FT (weight interpolation) is free lunch.
- Forgetting eval needs distant OOD benchmarks, not same-domain holdouts.
- Temperature/calibration is high-value but read alongside other signals.
- Isolate one variable per experiment.
