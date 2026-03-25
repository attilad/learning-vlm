"""Model loading utilities for VLM experiments.

Supports HuggingFace VLMs with BF16 and 4-bit quantization.
Crashes loudly if CUDA is unavailable — no silent CPU fallback.
"""

import os
from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    PreTrainedModel,
)

# Force HF cache to project-local directory to avoid WSL permission issues
os.environ.setdefault("HF_HOME", ".cache/huggingface")


@dataclass
class VRAMReport:
    """Snapshot of GPU memory state."""

    allocated_gb: float
    reserved_gb: float
    total_gb: float

    def __str__(self) -> str:
        return (
            f"VRAM: {self.allocated_gb:.2f}GB allocated, "
            f"{self.reserved_gb:.2f}GB reserved, "
            f"{self.total_gb:.2f}GB total"
        )


def require_cuda() -> torch.device:
    """Return CUDA device or crash. No silent CPU fallback."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This project requires a GPU. "
            "Check your CUDA installation and driver."
        )
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
    return device


def get_vram_report() -> VRAMReport:
    """Snapshot current GPU memory usage."""
    return VRAMReport(
        allocated_gb=torch.cuda.memory_allocated() / 1024**3,
        reserved_gb=torch.cuda.memory_reserved() / 1024**3,
        total_gb=torch.cuda.get_device_properties(0).total_memory / 1024**3,
    )


def make_quantization_config(bits: int = 4) -> BitsAndBytesConfig:
    """Create a BitsAndBytesConfig for quantized loading.

    Only 4-bit (NF4) is supported — it's the standard for QLoRA
    and the only config that fits 3B+ models comfortably on a 4090.
    """
    if bits != 4:
        raise ValueError(f"Only 4-bit quantization is supported, got {bits}")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_vlm(
    model_id: str,
    *,
    quantize: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> tuple[PreTrainedModel, AutoProcessor]:
    """Load a VLM and its processor from HuggingFace.

    Args:
        model_id: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct").
        quantize: If True, load in 4-bit NF4 quantization for QLoRA.
        dtype: Precision for non-quantized loading. Default BF16.
        trust_remote_code: Required by some models (e.g., Qwen).

    Returns:
        (model, processor) tuple ready for inference or LoRA attachment.
    """
    device = require_cuda()

    quantization_config = make_quantization_config() if quantize else None

    print(f"Loading model: {model_id}")
    print(f"  dtype: {dtype}, quantize: {quantize}")

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        quantization_config=quantization_config,
        trust_remote_code=trust_remote_code,
        # If not quantized, explicitly place on GPU. BnB handles placement itself.
        device_map="auto" if quantize else {"": device},
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    vram = get_vram_report()
    print(f"  Model loaded. {vram}")

    return model, processor


def attach_lora(
    model: PreTrainedModel,
    *,
    rank: int = 8,
    alpha: int = 16,
    target_modules: list[str] | None = None,
    dropout: float = 0.05,
) -> PreTrainedModel:
    """Wrap a model with LoRA adapters.

    Args:
        model: Base model (typically from load_vlm).
        rank: LoRA rank. Start with 8, sweep in Experiment 3.
        alpha: LoRA alpha (scaling). Convention: alpha = 2 * rank.
        target_modules: Which modules to adapt. None = auto-detect attention layers.
        dropout: LoRA dropout for regularization.

    Returns:
        PEFT-wrapped model with trainable LoRA parameters.
    """
    # Default to attention projections — the standard starting point.
    # Experiment 4 will compare this against attention+MLP and projector-only.
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, config)

    trainable, total = peft_model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    print(f"  LoRA attached: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")

    return peft_model
