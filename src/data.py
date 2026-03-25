"""Dataset loading and formatting for VLM experiments.

Each task has a registry entry mapping to an HF dataset, prompt template,
and column names. format_for_model() bridges the gap between task-agnostic
samples and model-specific input formats (Qwen chat template vs PaliGemma flat).
"""

import os
from dataclasses import dataclass, field
from enum import Enum

from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor

# Ensure HF cache is set before any dataset downloads
os.environ.setdefault("HF_HOME", ".cache/huggingface")

SEED = 42


class TaskType(Enum):
    CAPTIONING = "captioning"
    VQA = "vqa"
    DOC_QA = "doc_qa"
    CHART_QA = "chart_qa"


@dataclass
class TaskSample:
    """One evaluation example: an image, a prompt, and reference answers."""

    image: Image.Image
    prompt: str
    references: list[str]
    task: TaskType
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskConfig:
    """How to load and format a task from HuggingFace."""

    dataset_id: str
    split: str
    subset: str | None
    prompt_template: str
    # Column names vary across datasets — these map to TaskSample fields
    image_column: str
    question_column: str | None  # None for captioning (no question)
    reference_column: str


# ---------------------------------------------------------------------------
# Task registry — one entry per task type.
# These are small, public, HF-hosted datasets suitable for fast eval.
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[TaskType, TaskConfig] = {
    TaskType.CAPTIONING: TaskConfig(
        # Flickr8k has embedded PIL images and 5 captions per image.
        # Flickr30k's HF dataset uses a legacy loading script (broken in datasets 4.x).
        dataset_id="jxie/flickr8k",
        split="test",
        subset=None,
        prompt_template="Describe this image in one sentence.",
        image_column="image",
        question_column=None,
        reference_column="_captions",  # Synthetic — built from caption_0..caption_4
    ),
    TaskType.VQA: TaskConfig(
        dataset_id="merve/vqav2-small",
        split="validation",
        subset=None,
        prompt_template="Answer the following question about the image briefly: {question}",
        image_column="image",
        question_column="question",
        reference_column="multiple_choice_answer",
    ),
    TaskType.DOC_QA: TaskConfig(
        dataset_id="lmms-lab/DocVQA",
        split="validation",
        subset="DocVQA",
        prompt_template="Look at this document image and answer briefly: {question}",
        image_column="image",
        question_column="question",
        reference_column="answers",
    ),
    TaskType.CHART_QA: TaskConfig(
        dataset_id="ahmed-masry/ChartQA",
        split="test",
        subset=None,
        prompt_template="Based on the chart, answer briefly: {question}",
        image_column="image",
        question_column="query",
        reference_column="label",
    ),
}


def load_task(task: TaskType, n_samples: int = 500) -> list[TaskSample]:
    """Load and format n_samples from a task dataset.

    Uses a fixed seed for reproducible subsampling. All images are
    converted to RGB PIL Images regardless of source format.
    """
    config = TASK_REGISTRY[task]

    print(f"Loading {task.value}: {config.dataset_id} ({config.split})")

    ds = load_dataset(
        config.dataset_id,
        name=config.subset,
        split=config.split,
    )

    # Subsample with fixed seed for reproducibility
    if len(ds) > n_samples:
        ds = ds.shuffle(seed=SEED).select(range(n_samples))
    print(f"  {len(ds)} samples loaded")

    samples = []
    for row in ds:
        image = row[config.image_column]
        # Some datasets (e.g. ChartQA) store images as raw bytes
        if isinstance(image, bytes):
            import io
            image = Image.open(io.BytesIO(image))
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected PIL Image in column '{config.image_column}', "
                f"got {type(image)}. Dataset may need a custom adapter."
            )
        image = image.convert("RGB")

        # Build the prompt — captioning has no question, others do
        if config.question_column is not None:
            prompt = config.prompt_template.format(question=row[config.question_column])
        else:
            prompt = config.prompt_template

        # Normalize references to a list of strings.
        # Flickr8k stores captions in separate columns (caption_0..caption_4),
        # so we handle the synthetic "_captions" column specially.
        if config.reference_column == "_captions":
            refs = [row[f"caption_{i}"] for i in range(5) if f"caption_{i}" in row]
        else:
            refs = row[config.reference_column]
        if isinstance(refs, str):
            refs = [refs]
        elif isinstance(refs, list):
            # Flickr30k has list of captions, DocVQA has list of answers
            refs = [str(r) for r in refs]
        else:
            refs = [str(refs)]

        samples.append(TaskSample(
            image=image,
            prompt=prompt,
            references=refs,
            task=task,
            metadata={"dataset_id": config.dataset_id},
        ))

    return samples


def load_training_dataset(
    task: TaskType,
    n_train: int = 2000,
    n_eval: int = 500,
) -> tuple["Dataset", "Dataset"]:
    """Load a task dataset formatted for TRL SFTTrainer (messages + images).

    SFTTrainer's VLM collator expects each row to have:
      - "messages": list of chat message dicts (user + assistant)
      - "images": list of PIL Images

    The collator handles chat template application, tokenization, and label creation.

    Returns:
        (train_dataset, eval_dataset) — both HuggingFace Datasets.
    """
    from datasets import Dataset as HFDataset

    config = TASK_REGISTRY[task]
    print(f"Loading training data for {task.value}: {config.dataset_id}")

    # Load the train split for training, val/test split for eval.
    # ChartQA has train/val/test; VQA has train/validation; DocVQA has train/validation/test.
    train_split = "train"
    eval_split = config.split  # Reuse the eval split from the registry

    train_ds = load_dataset(config.dataset_id, name=config.subset, split=train_split)
    eval_ds = load_dataset(config.dataset_id, name=config.subset, split=eval_split)

    # Subsample
    if len(train_ds) > n_train:
        train_ds = train_ds.shuffle(seed=SEED).select(range(n_train))
    if len(eval_ds) > n_eval:
        eval_ds = eval_ds.shuffle(seed=SEED).select(range(n_eval))

    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    def _to_sft_format(row: dict) -> dict:
        """Convert a dataset row to SFTTrainer's messages + images format."""
        # Get the image
        img = row[config.image_column]
        if isinstance(img, bytes):
            import io
            img = Image.open(io.BytesIO(img))
        if isinstance(img, Image.Image):
            img = img.convert("RGB")

        # Build the question prompt
        if config.question_column is not None:
            prompt = config.prompt_template.format(question=row[config.question_column])
        else:
            prompt = config.prompt_template

        # Get the answer — use first reference for training
        if config.reference_column == "_captions":
            answer = row.get("caption_0", "")
        else:
            ref = row[config.reference_column]
            if isinstance(ref, list):
                answer = str(ref[0]) if ref else ""
            else:
                answer = str(ref)

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        return {"messages": messages, "images": [img]}

    # Map to SFT format. We can't use .map() easily because PIL images
    # don't serialize well in Arrow, so we build lists and create new datasets.
    def _convert_split(ds) -> HFDataset:
        records = []
        for row in ds:
            records.append(_to_sft_format(row))
        return HFDataset.from_list(records)

    train_sft = _convert_split(train_ds)
    eval_sft = _convert_split(eval_ds)

    return train_sft, eval_sft


def format_for_model(
    sample: TaskSample,
    processor: AutoProcessor,
    model_id: str,
) -> dict:
    """Convert a TaskSample into model-ready input tensors.

    Dispatches on model family because Qwen uses a chat template
    while PaliGemma uses a flat text+image call. Returns tensors
    on CPU — caller moves to device.
    """
    model_id_lower = model_id.lower()

    if "qwen" in model_id_lower:
        return _format_qwen(sample, processor)
    elif "paligemma" in model_id_lower:
        return _format_paligemma(sample, processor)
    elif "smolvlm" in model_id_lower:
        return _format_smolvlm(sample, processor)
    else:
        # Fallback: try chat template first, then flat
        try:
            return _format_qwen(sample, processor)
        except Exception:
            return _format_paligemma(sample, processor)


def _format_qwen(sample: TaskSample, processor: AutoProcessor) -> dict:
    """Qwen2.5-VL uses a chat template with structured message dicts."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(text=[text], images=[sample.image], return_tensors="pt", padding=True)


def _format_paligemma(sample: TaskSample, processor: AutoProcessor) -> dict:
    """PaliGemma uses a flat text prompt — the processor prepends image tokens."""
    return processor(text=sample.prompt, images=sample.image, return_tensors="pt")


def _format_smolvlm(sample: TaskSample, processor: AutoProcessor) -> dict:
    """SmolVLM2 uses a chat template similar to Qwen."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(text=[text], images=[sample.image], return_tensors="pt", padding=True)
