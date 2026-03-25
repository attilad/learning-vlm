"""Experiment 001: Zero-shot baseline evaluation.

Runs two models across four tasks, collects metrics, and saves results.

Usage:
    uv run python scripts/run_baseline.py
    uv run python scripts/run_baseline.py --n-samples 50  # quick test
    uv run python scripts/run_baseline.py --tasks captioning vqa
    uv run python scripts/run_baseline.py --models "Qwen/Qwen2.5-VL-3B-Instruct"
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

from src.data import TaskType, load_task
from src.eval import evaluate_task
from src.model import load_vlm, require_cuda

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "google/paligemma-3b-pt-224",
]

TASK_MAP = {
    "captioning": TaskType.CAPTIONING,
    "vqa": TaskType.VQA,
    "doc_qa": TaskType.DOC_QA,
    "chart_qa": TaskType.CHART_QA,
}

# Max new tokens per task — no need to generate 256 tokens for a VQA answer
TASK_MAX_TOKENS = {
    TaskType.CAPTIONING: 80,
    TaskType.VQA: 30,
    TaskType.DOC_QA: 50,
    TaskType.CHART_QA: 30,
}


def run_baseline(
    models: list[str],
    tasks: list[TaskType],
    n_samples: int,
    output_dir: Path,
) -> dict:
    device = require_cuda()

    # Pre-load all task datasets so download time doesn't pollute model timing
    print("\n=== Loading datasets ===")
    task_samples = {}
    for task in tasks:
        task_samples[task] = load_task(task, n_samples)

    results: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples_per_task": n_samples,
        "models": {},
    }

    for model_id in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_id}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        model, processor = load_vlm(model_id)
        load_time = time.perf_counter() - t0
        print(f"  Loaded in {load_time:.1f}s")

        model.eval()
        model_results: dict = {}

        for task in tasks:
            samples = task_samples[task]
            max_tokens = TASK_MAX_TOKENS[task]

            print(f"\n--- {task.value} ({len(samples)} samples, max_tokens={max_tokens}) ---")

            metrics, gen_results = evaluate_task(
                model, processor, samples, model_id, device, max_tokens
            )

            model_results[task.value] = asdict(metrics)

            print(f"  {metrics.metric_name}: {metrics.accuracy:.4f}")
            print(f"  latency: {metrics.median_latency_ms:.0f}ms median, "
                  f"{metrics.p95_latency_ms:.0f}ms p95")
            print(f"  throughput: {metrics.tokens_per_sec:.1f} tok/s")
            print(f"  peak VRAM: {metrics.peak_vram_gb:.2f}GB")

            # Save per-sample outputs for hallucination audit
            _save_generations(gen_results, model_id, task, output_dir)

        results["models"][model_id] = model_results

        # Free VRAM before loading next model
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    return results


def _save_generations(
    results: list,
    model_id: str,
    task: TaskType,
    output_dir: Path,
) -> None:
    """Save individual generation outputs as JSONL for later hallucination audit."""
    # Sanitize model name for filename
    safe_name = model_id.replace("/", "_")
    path = output_dir / f"generations_{safe_name}_{task.value}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for r in results:
            record = {
                "prediction": r.text,
                "references": r.sample.references,
                "prompt": r.sample.prompt,
                "n_tokens": r.n_tokens,
                "latency_s": round(r.latency_s, 4),
            }
            f.write(json.dumps(record) + "\n")


def print_summary(results: dict) -> None:
    """Print a comparison table of all models and tasks."""
    models = results["models"]
    if not models:
        return

    # Gather all tasks across models
    all_tasks = set()
    for model_data in models.values():
        all_tasks.update(model_data.keys())
    all_tasks = sorted(all_tasks)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Header
    header = f"{'Model':<40} {'Task':<12} {'Metric':<18} {'Score':>7} {'Med ms':>7} {'Tok/s':>7} {'VRAM':>6}"
    print(header)
    print("-" * len(header))

    for model_id, model_data in models.items():
        short_name = model_id.split("/")[-1]
        for task_name in all_tasks:
            if task_name not in model_data:
                continue
            m = model_data[task_name]
            print(
                f"{short_name:<40} {task_name:<12} {m['metric_name']:<18} "
                f"{m['accuracy']:>7.4f} {m['median_latency_ms']:>7.0f} "
                f"{m['tokens_per_sec']:>7.1f} {m['peak_vram_gb']:>5.1f}G"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 001: Zero-shot baseline")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="HuggingFace model IDs to evaluate",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_MAP.keys()),
        choices=list(TASK_MAP.keys()),
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of samples per task",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/001_baseline",
        help="Directory to save results",
    )
    args = parser.parse_args()

    tasks = [TASK_MAP[t] for t in args.tasks]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_baseline(args.models, tasks, args.n_samples, output_dir)

    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
