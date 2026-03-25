"""Evaluation framework for VLM experiments.

Handles generation, timing, VRAM tracking, and task-specific metrics.
Single-sample generation (no batching) — VLMs with variable-size images
make batching fragile, and 500 samples is fast enough sequentially.
"""

import re
import time
from dataclasses import dataclass

import torch
from transformers import AutoProcessor, PreTrainedModel

from src.data import TaskSample, TaskType, format_for_model


@dataclass
class GenerationResult:
    """Output from a single model.generate() call with timing."""

    text: str
    n_tokens: int
    latency_s: float
    sample: TaskSample


@dataclass
class TaskMetrics:
    """Aggregated metrics for one model on one task."""

    accuracy: float
    metric_name: str
    median_latency_ms: float
    mean_latency_ms: float
    p95_latency_ms: float
    peak_vram_gb: float
    tokens_per_sec: float
    n_samples: int


def generate_one(
    model: PreTrainedModel,
    processor: AutoProcessor,
    sample: TaskSample,
    model_id: str,
    device: torch.device,
    max_new_tokens: int = 256,
) -> GenerationResult:
    """Generate a response for a single sample. Greedy decoding for reproducibility."""
    inputs = format_for_model(sample, processor, model_id)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy — deterministic and reproducible
        )

    torch.cuda.synchronize()  # Wait for GPU to finish before timing
    latency = time.perf_counter() - t0

    new_tokens = generated[0][input_len:]
    text = processor.decode(new_tokens, skip_special_tokens=True)

    return GenerationResult(
        text=text,
        n_tokens=len(new_tokens),
        latency_s=latency,
        sample=sample,
    )


def evaluate_task(
    model: PreTrainedModel,
    processor: AutoProcessor,
    samples: list[TaskSample],
    model_id: str,
    device: torch.device,
    max_new_tokens: int = 256,
) -> tuple[TaskMetrics, list[GenerationResult]]:
    """Run generation + metrics for a list of samples.

    Returns both aggregated metrics and individual results
    (for the hallucination audit and per-sample analysis).
    """
    assert len(samples) > 0, "No samples to evaluate"
    task = samples[0].task

    # Reset peak VRAM tracking for this task
    torch.cuda.reset_peak_memory_stats()

    results: list[GenerationResult] = []
    for i, sample in enumerate(samples):
        result = generate_one(model, processor, sample, model_id, device, max_new_tokens)
        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(samples)}] last output: {result.text[:80]!r}")

    # Compute timing stats
    latencies_ms = [r.latency_s * 1000 for r in results]
    latencies_ms.sort()
    total_tokens = sum(r.n_tokens for r in results)
    total_time = sum(r.latency_s for r in results)

    p95_idx = int(len(latencies_ms) * 0.95)
    median_idx = len(latencies_ms) // 2

    # Compute task-specific accuracy
    accuracy, metric_name = compute_accuracy(results, task)

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    metrics = TaskMetrics(
        accuracy=accuracy,
        metric_name=metric_name,
        median_latency_ms=latencies_ms[median_idx],
        mean_latency_ms=sum(latencies_ms) / len(latencies_ms),
        p95_latency_ms=latencies_ms[p95_idx],
        peak_vram_gb=peak_vram_gb,
        tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
        n_samples=len(results),
    )

    return metrics, results


# ---------------------------------------------------------------------------
# Task-specific accuracy metrics
# ---------------------------------------------------------------------------


def compute_accuracy(
    results: list[GenerationResult], task: TaskType
) -> tuple[float, str]:
    """Dispatch to the right metric based on task type."""
    if task == TaskType.CAPTIONING:
        return compute_rouge_l(results), "rouge_l"
    elif task == TaskType.VQA:
        return compute_vqa_accuracy(results), "vqa_accuracy"
    elif task == TaskType.DOC_QA:
        return compute_anls(results), "anls"
    elif task == TaskType.CHART_QA:
        return compute_relaxed_accuracy(results), "relaxed_accuracy"
    else:
        raise ValueError(f"Unknown task type: {task}")


def _normalize_text(text: str) -> str:
    """Lowercase, strip, remove punctuation. Used for VQA-style matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_rouge_l(results: list[GenerationResult]) -> float:
    """ROUGE-L F1 for captioning.

    Using ROUGE-L instead of CIDEr because CIDEr requires corpus-level
    TF-IDF which adds complexity. ROUGE-L is a reasonable proxy for
    zero-shot baseline comparison — we're comparing models, not chasing
    leaderboard numbers.
    """
    scores = []
    for r in results:
        pred_tokens = _normalize_text(r.text).split()
        # Compare against each reference, take the best
        best_score = 0.0
        for ref in r.sample.references:
            ref_tokens = _normalize_text(ref).split()
            if not ref_tokens or not pred_tokens:
                continue
            lcs_len = _lcs_length(pred_tokens, ref_tokens)
            precision = lcs_len / len(pred_tokens) if pred_tokens else 0
            recall = lcs_len / len(ref_tokens) if ref_tokens else 0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_score = max(best_score, f1)
        scores.append(best_score)

    return sum(scores) / len(scores) if scores else 0.0


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of longest common subsequence. O(n*m) DP — fine for short sequences."""
    m, n = len(a), len(b)
    # Space-optimized: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_vqa_accuracy(results: list[GenerationResult]) -> float:
    """Standard VQA accuracy: min(matching_annotators / 3, 1.0).

    If there's only one reference (e.g. from merve/vqav2-small),
    falls back to exact match.
    """
    scores = []
    for r in results:
        pred = _normalize_text(r.text)
        refs = [_normalize_text(ref) for ref in r.sample.references]
        if len(refs) > 1:
            # Standard VQA accuracy with 10 annotator answers
            match_count = sum(1 for ref in refs if ref == pred)
            score = min(match_count / 3.0, 1.0)
        else:
            # Single reference — exact match
            score = 1.0 if pred == refs[0] else 0.0
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def compute_anls(results: list[GenerationResult]) -> float:
    """Average Normalized Levenshtein Similarity for DocVQA.

    ANLS = 1 - NLD if NLD < 0.5, else 0. Takes max over references.
    """
    scores = []
    for r in results:
        pred = _normalize_text(r.text)
        best_score = 0.0
        for ref in r.sample.references:
            ref_norm = _normalize_text(ref)
            if not ref_norm and not pred:
                best_score = 1.0
                continue
            nld = _levenshtein(pred, ref_norm) / max(len(pred), len(ref_norm), 1)
            score = 1.0 - nld if nld < 0.5 else 0.0
            best_score = max(best_score, score)
        scores.append(best_score)

    return sum(scores) / len(scores) if scores else 0.0


def _levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance. Pure Python — only called ~500 times per eval."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, deletion, substitution
            curr_row.append(min(
                prev_row[j + 1] + 1,
                curr_row[j] + 1,
                prev_row[j] + (0 if c1 == c2 else 1),
            ))
        prev_row = curr_row

    return prev_row[-1]


def compute_relaxed_accuracy(results: list[GenerationResult], tolerance: float = 0.05) -> float:
    """Relaxed accuracy for ChartQA.

    Exact string match, but if both values parse as numbers,
    allow `tolerance` relative error. Standard ChartQA metric.
    """
    scores = []
    for r in results:
        pred = _normalize_text(r.text)
        matched = False
        for ref in r.sample.references:
            ref_norm = _normalize_text(ref)
            if pred == ref_norm:
                matched = True
                break
            # Try numeric comparison
            try:
                pred_num = float(pred.replace(",", "").replace("%", ""))
                ref_num = float(ref_norm.replace(",", "").replace("%", ""))
                if ref_num != 0 and abs(pred_num - ref_num) / abs(ref_num) <= tolerance:
                    matched = True
                    break
                elif ref_num == 0 and abs(pred_num) <= tolerance:
                    matched = True
                    break
            except ValueError:
                continue
        scores.append(1.0 if matched else 0.0)

    return sum(scores) / len(scores) if scores else 0.0
