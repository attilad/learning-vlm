"""Experiment 008: WiSE-FT (weight interpolation) + OOD Transfer.

Loads the rank-8 adapter from Exp 003, sweeps interpolation alpha,
and evaluates on 3 domain tiers: in-domain (ChartQA), shifted (DocVQA),
distant OOD (VQA).

Usage:
    uv run python scripts/run_wise_ft.py
    uv run python scripts/run_wise_ft.py --n-samples 50           # quick test
    uv run python scripts/run_wise_ft.py --alphas 0.0 0.5 1.0     # fewer points
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, ".")

from src.adapt import apply_wise_ft, load_adapter
from src.data import TaskType, load_task
from src.eval import evaluate_task
from src.model import load_vlm, require_cuda

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "experiments/003_rank_sweep/rank_r8/final"
DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Three evaluation tiers: in-domain → shifted → distant OOD
EVAL_TASKS = [TaskType.CHART_QA, TaskType.DOC_QA, TaskType.VQA]
TASK_MAX_TOKENS = {
    TaskType.CHART_QA: 30,
    TaskType.DOC_QA: 50,
    TaskType.VQA: 30,
}


def run_wise_ft(
    alphas: list[float],
    n_samples: int,
    adapter_path: str,
    output_dir: Path,
) -> dict:
    device = require_cuda()

    # Pre-load all eval datasets
    print("\n=== Loading datasets ===")
    task_samples: dict = {}
    for task in EVAL_TASKS:
        task_samples[task] = load_task(task, n_samples=n_samples)

    # Load model + adapter ONCE. set_scale is reversible so we reuse
    # across all alpha values — no contamination, saves ~30s per alpha.
    print("\n=== Loading model + adapter ===")
    model, processor = load_vlm(MODEL_ID)
    model = load_adapter(model, adapter_path)
    model.eval()

    results: dict = {
        "experiment": "008_wise_ft",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "adapter_path": adapter_path,
        "config": {
            "alphas": alphas,
            "n_samples": n_samples,
            "eval_tasks": [t.value for t in EVAL_TASKS],
        },
        "runs": {},
    }

    for alpha in alphas:
        alpha_str = f"{alpha:.2f}"

        print(f"\n{'='*60}")
        print(f"WiSE-FT: alpha={alpha_str}")
        print(f"{'='*60}")

        apply_wise_ft(model, alpha)

        alpha_results: dict = {}
        for task in EVAL_TASKS:
            samples = task_samples[task]
            max_tokens = TASK_MAX_TOKENS[task]

            print(f"\n--- {task.value} ({len(samples)} samples) ---")
            torch.cuda.reset_peak_memory_stats()

            metrics, gen_results = evaluate_task(
                model, processor, samples, MODEL_ID, device, max_tokens
            )

            alpha_results[task.value] = asdict(metrics)

            print(f"  {metrics.metric_name}: {metrics.accuracy:.4f}")
            print(f"  latency: {metrics.median_latency_ms:.0f}ms median")

            # Save generations for hallucination audit
            gen_path = output_dir / f"generations_a{alpha_str}_{task.value}.jsonl"
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

        results["runs"][alpha_str] = alpha_results

    return results


def print_summary(results: dict) -> None:
    print(f"\n{'='*80}")
    print("SUMMARY: WiSE-FT INTERPOLATION")
    print(f"{'='*80}")

    # Per-alpha, per-task table
    header = f"{'Alpha':<8} {'Task':<12} {'Metric':<18} {'Score':>7} {'Med ms':>7} {'Tok/s':>7}"
    print(header)
    print("-" * len(header))

    for alpha_str, tasks in results["runs"].items():
        for task_name, m in tasks.items():
            print(
                f"{alpha_str:<8} {task_name:<12} {m['metric_name']:<18} "
                f"{m['accuracy']:>7.4f} {m['median_latency_ms']:>7.0f} "
                f"{m['tokens_per_sec']:>7.1f}"
            )
        print()

    # WiSE-FT curve: ChartQA accuracy vs alpha
    print("WiSE-FT CURVE (ChartQA in-domain):")
    print(f"  {'Alpha':<8} {'Accuracy':>10}")
    print(f"  {'-'*20}")
    for alpha_str, tasks in results["runs"].items():
        if "chart_qa" in tasks:
            acc = tasks["chart_qa"]["accuracy"]
            bar = "█" * int(acc * 40)
            print(f"  {alpha_str:<8} {acc:>10.4f}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 008: WiSE-FT + OOD transfer")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
        help="WiSE-FT interpolation coefficients (0=pretrained, 1=fine-tuned)",
    )
    parser.add_argument("--n-samples", type=int, default=500, help="Eval samples per task")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=ADAPTER_PATH,
        help="Path to saved LoRA adapter",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/008_wise_ft",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_wise_ft(args.alphas, args.n_samples, args.adapter_path, output_dir)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
