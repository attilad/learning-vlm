"""Experiment 002: LoRA Learning Rate Sweep.

Trains Qwen2.5-VL-3B-Instruct with LoRA on ChartQA at 3 learning rates.
Fixed: rank=8, attention-only (q_proj, v_proj), 2K training examples, 3 epochs.

Usage:
    uv run python scripts/run_lr_sweep.py
    uv run python scripts/run_lr_sweep.py --n-train 500 --n-eval 100  # quick test
    uv run python scripts/run_lr_sweep.py --lrs 1e-4               # single LR
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, ".")

from src.data import TaskType, load_task, load_training_dataset
from src.eval import evaluate_task
from src.model import load_vlm, require_cuda
from src.train import TrainResult, make_lora_config, train_vlm

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TASK = TaskType.CHART_QA
DEFAULT_LRS = [5e-5, 1e-4, 2e-4]

# Fixed hyperparameters for this sweep
RANK = 8
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8  # Effective batch size = 16
# ChartQA images produce 145-1534 tokens with Qwen's dynamic resolution.
# P95 is ~1060, max is ~1534. Use 1536 to avoid truncating any samples.
MAX_LENGTH = 1536


def run_lr_sweep(
    lrs: list[float],
    n_train: int,
    n_eval: int,
    output_dir: Path,
) -> dict:
    device = require_cuda()

    # Load training + eval data once (shared across all LRs)
    print("\n=== Loading datasets ===")
    train_dataset, eval_dataset = load_training_dataset(TASK, n_train=n_train, n_eval=n_eval)

    # Also load eval samples in TaskSample format for generation-based eval
    eval_samples = load_task(TASK, n_samples=n_eval)

    results: dict = {
        "experiment": "002_lr_sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "task": TASK.value,
        "fixed": {
            "rank": RANK,
            "target_modules": ["q_proj", "v_proj"],
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "max_length": MAX_LENGTH,
            "n_train": n_train,
            "n_eval": n_eval,
        },
        "runs": {},
    }

    for lr in lrs:
        lr_str = f"{lr:.0e}"
        run_name = f"lr_{lr_str}"
        run_dir = str(output_dir / run_name)

        print(f"\n{'='*60}")
        print(f"LR SWEEP: {lr_str}")
        print(f"{'='*60}")

        # Load a fresh model for each LR to avoid contamination
        model, processor = load_vlm(MODEL_ID)
        model.eval()

        # Zero-shot eval before training (only on first LR — same model, same result)
        if lr == lrs[0]:
            print("\n--- Zero-shot eval (before training) ---")
            zs_metrics, _ = evaluate_task(model, processor, eval_samples, MODEL_ID, device, max_new_tokens=30)
            results["zero_shot"] = asdict(zs_metrics)
            print(f"  {zs_metrics.metric_name}: {zs_metrics.accuracy:.4f}")

        # Train
        lora_config = make_lora_config(rank=RANK)
        train_result = train_vlm(
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=run_dir,
            learning_rate=lr,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            max_length=MAX_LENGTH,
            lora_config=lora_config,
            tb_run_name=f"002_lr_{lr_str}",
        )

        # Generation-based eval after training
        print(f"\n--- Post-training eval (LR={lr_str}) ---")
        model.eval()
        post_metrics, gen_results = evaluate_task(model, processor, eval_samples, MODEL_ID, device, max_new_tokens=30)
        print(f"  {post_metrics.metric_name}: {post_metrics.accuracy:.4f}")

        results["runs"][lr_str] = {
            "learning_rate": lr,
            "train": asdict(train_result),
            "eval_generation": asdict(post_metrics),
        }

        # Save per-sample generations for inspection
        gen_path = output_dir / f"generations_{lr_str}.jsonl"
        with open(gen_path, "w") as f:
            for r in gen_results:
                record = {
                    "prediction": r.text,
                    "references": r.sample.references,
                    "prompt": r.sample.prompt,
                    "n_tokens": r.n_tokens,
                    "latency_s": round(r.latency_s, 4),
                }
                f.write(json.dumps(record) + "\n")

        # Free VRAM
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    return results


def print_summary(results: dict) -> None:
    """Print a comparison table of all LRs."""
    print(f"\n{'='*80}")
    print("SUMMARY: LR SWEEP")
    print(f"{'='*80}")

    zs = results.get("zero_shot", {})
    if zs:
        print(f"Zero-shot baseline: {zs.get('metric_name', 'N/A')} = {zs.get('accuracy', 0):.4f}")
    print()

    header = f"{'LR':<12} {'Train Loss':>12} {'Eval Loss':>12} {'Gen Accuracy':>14} {'VRAM (GB)':>10} {'Time (s)':>10}"
    print(header)
    print("-" * len(header))

    for lr_str, run in results["runs"].items():
        t = run["train"]
        g = run["eval_generation"]
        eval_loss = f"{t['eval_loss']:.4f}" if t.get("eval_loss") is not None else "N/A"
        print(
            f"{lr_str:<12} {t['train_loss']:>12.4f} {eval_loss:>12} "
            f"{g['accuracy']:>14.4f} {t['peak_vram_gb']:>10.2f} {t['wall_time_s']:>10.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 002: LoRA LR sweep")
    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=DEFAULT_LRS,
        help="Learning rates to sweep",
    )
    parser.add_argument("--n-train", type=int, default=2000, help="Training samples")
    parser.add_argument("--n-eval", type=int, default=500, help="Eval samples")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/002_lr_sweep",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_lr_sweep(args.lrs, args.n_train, args.n_eval, output_dir)

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
