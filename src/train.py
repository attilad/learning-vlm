"""LoRA/QLoRA training loop via TRL SFTTrainer.

SFTTrainer auto-detects VLMs when given a ProcessorMixin as processing_class.
It handles image collation, chat template application, and label masking
internally — we just provide a dataset with "messages" + "images" keys.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import PreTrainedModel
from trl import SFTConfig, SFTTrainer

from src.model import get_vram_report, require_cuda

# Ensure HF cache is set
os.environ.setdefault("HF_HOME", ".cache/huggingface")


@dataclass
class TrainResult:
    """Summary of a training run for experiment logging."""

    output_dir: str
    train_loss: float
    eval_loss: float | None
    train_samples: int
    train_steps: int
    peak_vram_gb: float
    wall_time_s: float


def make_lora_config(
    rank: int = 8,
    alpha: int | None = None,
    target_modules: list[str] | None = None,
    dropout: float = 0.05,
) -> LoraConfig:
    """Create a LoRA config for SFTTrainer.

    Convention: alpha = 2 * rank unless overridden.
    Default targets: attention Q and V projections (Experiment 4 will sweep this).
    """
    if alpha is None:
        alpha = 2 * rank
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_vlm(
    model: PreTrainedModel,
    processor: "AutoProcessor",
    train_dataset: "Dataset",
    eval_dataset: "Dataset | None" = None,
    *,
    output_dir: str = "checkpoints/run",
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_length: int = 1536,
    lora_config: LoraConfig | None = None,
    warmup_ratio: float = 0.1,
    logging_steps: int = 5,
    eval_steps: int | None = None,
    save_strategy: str = "epoch",
    tb_run_name: str | None = None,
) -> TrainResult:
    """Train a VLM with LoRA using TRL SFTTrainer.

    The dataset must have "messages" (list of chat dicts) and "images" (list of PIL Images).
    SFTTrainer's DataCollatorForVisionLanguageModeling handles the rest:
    - Applies the processor's chat template
    - Tokenizes text + encodes images
    - Creates labels (masking padding with -100)

    Args:
        model: Base VLM (will be wrapped with LoRA if lora_config provided).
        processor: Model's processor (AutoProcessor). Passed as processing_class to SFTTrainer.
        train_dataset: HF Dataset with "messages" and "images" columns.
        eval_dataset: Optional eval dataset, same format.
        output_dir: Where to save checkpoints.
        learning_rate: Peak LR for the cosine schedule.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size. Keep small (1-2) for VLMs on 4090.
        gradient_accumulation_steps: Effective batch = batch_size * grad_accum.
        max_length: Max sequence length (tokens). Truncates longer sequences.
        lora_config: LoRA configuration. If None, trains without LoRA (not recommended).
        warmup_ratio: Fraction of steps for linear LR warmup.
        logging_steps: Log training loss every N steps.
        eval_steps: Run eval every N steps. None = eval at end of each epoch.
        save_strategy: When to save checkpoints ("epoch", "steps", "no").
        tb_run_name: TensorBoard run name. Logs go to runs/{tb_run_name}.

    Returns:
        TrainResult with loss, timing, and VRAM info.
    """
    device = require_cuda()

    # TensorBoard logging directory
    logging_dir = f"runs/{tb_run_name}" if tb_run_name else "runs/default"

    sft_config = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        eval_strategy="steps" if eval_steps else "epoch",
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        bf16=True,
        gradient_checkpointing=True,
        # VLM constraints — these are NOT supported and will error if True
        packing=False,
        padding_free=False,
        # Determinism
        seed=42,
        data_seed=42,
        # Avoid OOM from caching during eval
        eval_do_concat_batches=False,
        # Report to tensorboard
        report_to="tensorboard",
        run_name=tb_run_name,
        # Don't push to hub
        push_to_hub=False,
    )

    # Reset VRAM tracking
    torch.cuda.reset_peak_memory_stats()

    import time
    t0 = time.perf_counter()

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=lora_config,
    )

    # Log trainable params
    if lora_config is not None:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    print(f"  Training for {num_epochs} epochs, LR={learning_rate}, batch={batch_size}x{gradient_accumulation_steps}")
    print(f"  {get_vram_report()}")

    train_output = trainer.train()

    wall_time = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    # Run final eval if we have an eval dataset
    eval_loss = None
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss")
        print(f"  Eval loss: {eval_loss:.4f}" if eval_loss else "  Eval loss: N/A")

    train_loss = train_output.training_loss
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Wall time: {wall_time:.0f}s, Peak VRAM: {peak_vram:.2f}GB")

    # Save the final adapter
    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    print(f"  Model saved to {final_dir}")

    return TrainResult(
        output_dir=output_dir,
        train_loss=train_loss,
        eval_loss=eval_loss,
        train_samples=len(train_dataset),
        train_steps=train_output.global_step,
        peak_vram_gb=peak_vram,
        wall_time_s=wall_time,
    )
