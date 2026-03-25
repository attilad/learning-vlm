"""Smoke test: verify GPU, load a model, run forward pass, generate tokens.

Run this before any experiment to confirm the full pipeline works.

Usage:
    uv run python scripts/smoke_test.py
    uv run python scripts/smoke_test.py --model "google/paligemma-3b-pt-224"
    uv run python scripts/smoke_test.py --quantize
"""

import argparse
import sys
import time

import torch
from PIL import Image

# Ensure project root is on path so we can import src/
sys.path.insert(0, ".")

from src.model import attach_lora, get_vram_report, load_vlm, require_cuda


def create_dummy_image(size: int = 224) -> Image.Image:
    """Create a solid-color test image. Not realistic, but sufficient for a smoke test."""
    return Image.new("RGB", (size, size), color=(128, 64, 192))


def run_smoke_test(model_id: str, quantize: bool) -> None:
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    # Step 1: GPU check
    print("\n[1/5] Checking GPU...")
    device = require_cuda()

    # Step 2: Load model
    print(f"\n[2/5] Loading model: {model_id} (quantize={quantize})...")
    t0 = time.perf_counter()
    model, processor = load_vlm(model_id, quantize=quantize)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Step 3: Forward pass with dummy image
    print("\n[3/5] Running forward pass...")
    image = create_dummy_image()

    # Build inputs using the processor's chat template if available,
    # otherwise fall back to a simple text prompt
    prompt = "Describe this image."
    try:
        # Qwen2.5-VL and similar chat-style models
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    except Exception:
        # PaliGemma and other simpler models
        inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits_shape = outputs.logits.shape
    print(f"  Logits shape: {logits_shape}")
    assert len(logits_shape) == 3, f"Expected 3D logits (batch, seq, vocab), got {logits_shape}"

    # Step 4: Generate tokens
    print("\n[4/5] Generating tokens...")
    t0 = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=20)
    gen_time = time.perf_counter() - t0

    # Decode only the new tokens (skip the input portion)
    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated[0][input_len:]
    decoded = processor.decode(new_tokens, skip_special_tokens=True)
    n_tokens = len(new_tokens)
    tokens_per_sec = n_tokens / gen_time if gen_time > 0 else 0

    print(f"  Generated {n_tokens} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
    print(f"  Output: {decoded!r}")

    # Step 5: LoRA attachment (verify PEFT integration works)
    print("\n[5/5] Attaching LoRA adapters...")
    lora_model = attach_lora(model, rank=4)
    vram = get_vram_report()
    print(f"  Final {vram}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM smoke test")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model ID to test",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load in 4-bit quantization",
    )
    args = parser.parse_args()
    run_smoke_test(args.model, args.quantize)


if __name__ == "__main__":
    main()
