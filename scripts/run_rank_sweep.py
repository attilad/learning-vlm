"""Experiment 003: LoRA Rank Sweep.

Trains Qwen2.5-VL-3B-Instruct with LoRA on ChartQA at 3 ranks.
Fixed: LR=1e-4 (winner from Exp 002), attention-only, 2K examples, 3 epochs.

Usage:
    uv run python scripts/run_rank_sweep.py
    uv run python scripts/run_rank_sweep.py --n-train 50 --n-eval 10  # quick test
    uv run python scripts/run_rank_sweep.py --ranks 8                 # single rank
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
from src.train import make_lora_config, train_vlm

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TASK = TaskType.CHART_QA
DEFAULT_RANKS = [4, 8, 16]

# Fixed hyperparameters (LR from Exp 002)
LR = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8  # Effective batch size = 16
MAX_LENGTH = 2048


def run_rank_sweep(
    ranks: list[int],
    n_train: int,
    n_eval: int,
    output_dir: Path,
) -> dict:
    device = require_cuda()

    print("\n=== Loading datasets ===")
    train_dataset, eval_dataset = load_training_dataset(TASK, n_train=n_train, n_eval=n_eval)
    eval_samples = load_task(TASK, n_samples=n_eval)

    results: dict = {
        "experiment": "003_rank_sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "task": TASK.value,
        "fixed": {
            "learning_rate": LR,
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

    for rank in ranks:
        rank_str = f"r{rank}"
        run_dir = str(output_dir / f"rank_{rank_str}")

        print(f"\n{'='*60}")
        print(f"RANK SWEEP: rank={rank}")
        print(f"{'='*60}")

        model, processor = load_vlm(MODEL_ID)
        model.eval()

        # Zero-shot eval once
        if rank == ranks[0]:
            print("\n--- Zero-shot eval (before training) ---")
            zs_metrics, _ = evaluate_task(model, processor, eval_samples, MODEL_ID, device, max_new_tokens=30)
            results["zero_shot"] = asdict(zs_metrics)
            print(f"  {zs_metrics.metric_name}: {zs_metrics.accuracy:.4f}")

        # Train with this rank
        lora_config = make_lora_config(rank=rank)
        train_result = train_vlm(
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=run_dir,
            learning_rate=LR,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            max_length=MAX_LENGTH,
            lora_config=lora_config,
            tb_run_name=f"003_rank_{rank_str}",
        )

        # Generation-based eval after training
        print(f"\n--- Post-training eval (rank={rank}) ---")
        model.eval()
        post_metrics, gen_results = evaluate_task(model, processor, eval_samples, MODEL_ID, device, max_new_tokens=30)
        print(f"  {post_metrics.metric_name}: {post_metrics.accuracy:.4f}")

        # Count trainable params for comparison
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        results["runs"][rank_str] = {
            "rank": rank,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100 * trainable / total,
            "train": asdict(train_result),
            "eval_generation": asdict(post_metrics),
        }

        # Save generations
        gen_path = output_dir / f"generations_{rank_str}.jsonl"
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

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    return results


def print_summary(results: dict) -> None:
    print(f"\n{'='*80}")
    print("SUMMARY: RANK SWEEP")
    print(f"{'='*80}")

    zs = results.get("zero_shot", {})
    if zs:
        print(f"Zero-shot baseline: {zs.get('metric_name', 'N/A')} = {zs.get('accuracy', 0):.4f}")
    print()

    header = f"{'Rank':<8} {'Params':>12} {'Train Loss':>12} {'Eval Loss':>12} {'Gen Accuracy':>14} {'VRAM (GB)':>10} {'Time (s)':>10}"
    print(header)
    print("-" * len(header))

    for rank_str, run in results["runs"].items():
        t = run["train"]
        g = run["eval_generation"]
        params = f"{run['trainable_params']:,}"
        eval_loss = f"{t['eval_loss']:.4f}" if t.get("eval_loss") is not None else "N/A"
        print(
            f"{rank_str:<8} {params:>12} {t['train_loss']:>12.4f} {eval_loss:>12} "
            f"{g['accuracy']:>14.4f} {t['peak_vram_gb']:>10.2f} {t['wall_time_s']:>10.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 003: LoRA rank sweep")
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=DEFAULT_RANKS,
        help="LoRA ranks to sweep",
    )
    parser.add_argument("--n-train", type=int, default=2000, help="Training samples")
    parser.add_argument("--n-eval", type=int, default=500, help="Eval samples")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/003_rank_sweep",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_rank_sweep(args.ranks, args.n_train, args.n_eval, output_dir)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
