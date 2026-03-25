# VLM: Parameter-Efficient Adaptation of Compact Vision-Language Models

A structured learning project studying parameter-efficient fine-tuning of compact vision-language models (VLMs) on a single NVIDIA RTX 4090. Builds on lessons from a prior CLIP contrastive training project, extending techniques like LoRA, QLoRA, WiSE-FT, and freeze strategies to generative VLMs.

## Research Question

> Can a compact pretrained VLM be adapted on a single 4090 for robust structured extraction without losing out-of-distribution (OOD) behavior?

## Models

| Model | Params | Strengths |
|-------|--------|-----------|
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3B | Strongest general VLM at this size |
| [PaliGemma 3B](https://huggingface.co/google/paligemma-3b-pt-224) | 3B | Task-specific (captioning, VQA, OCR) |
| [SmolVLM2-2.2B](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | 2.2B | Fastest iteration loop |

## Experiment Curriculum

Eight experiments, each isolating a single variable:

| # | Experiment | Variable | Phase |
|---|-----------|----------|-------|
| 1 | Zero-shot baseline | Model choice | Baseline |
| 2 | LR sweep | Learning rate (5e-5, 1e-4, 2e-4) | Fine-tuning |
| 3 | Rank sweep | LoRA rank (4, 8, 16) | Fine-tuning |
| 4 | Freeze strategy | Projector-only vs attention vs attention+MLP | Fine-tuning |
| 5 | Resolution | 224px vs 448px | Fine-tuning |
| 6 | Dataset size | 500 / 2K / 5K examples | Fine-tuning |
| 7 | Structured output | JSON extraction vs free-form | Transfer |
| 8 | OOD transfer | In-domain → shifted → distant OOD | Robustness |

See [LESSON_PLAN.md](LESSON_PLAN.md) for detailed hypotheses and methodology.

## Project Structure

```
vlm/
├── CLAUDE.md               # Project instructions for Claude Code
├── LESSON_PLAN.md           # 8-experiment curriculum with hypotheses
├── pyproject.toml           # uv-managed dependencies
├── src/
│   ├── model.py             # Model loading (HF, quantization, LoRA)
│   ├── data.py              # Dataset loading/formatting
│   ├── train.py             # Training loop (TRL SFTTrainer)
│   ├── eval.py              # Evaluation framework
│   └── adapt.py             # Adaptation utilities (freeze, WiSE-FT)
├── scripts/
│   ├── smoke_test.py        # GPU + model + generation sanity check
│   └── postmortem.py        # Post-training analysis
├── experiments/             # Numbered experiment directories
│   └── 001_baseline/
├── checkpoints/             # Model checkpoints
├── runs/                    # TensorBoard logs
└── data/                    # Task datasets
```

## Setup

Requires an NVIDIA GPU with CUDA 12.x. Developed on an RTX 4090 (24GB) under WSL2.

```bash
# Install dependencies (uses uv — do not use pip directly)
uv sync

# Verify GPU and model loading
uv run python scripts/smoke_test.py

# With 4-bit quantization
uv run python scripts/smoke_test.py --quantize

# With a specific model
uv run python scripts/smoke_test.py --model "google/paligemma-3b-pt-224"
```

## Key Principles

Carried forward from the CLIP project:

- **Adapt, don't train from scratch.** Pretrained models dominate.
- **LoRA rank-4 captured 96% of full FT gains on CLIP.** Does this hold for generative VLMs?
- **WiSE-FT is free lunch.** Always try weight interpolation after fine-tuning.
- **OOD eval needs distant benchmarks.** Same-domain holdouts miss forgetting.
- **One variable per experiment.** Hypothesis → method → results → conclusions.

## Hardware

- NVIDIA RTX 4090 (24GB VRAM, Ada Lovelace)
- CUDA 12.x, PyTorch 2.6+
- Python 3.12 via [uv](https://docs.astral.sh/uv/)
