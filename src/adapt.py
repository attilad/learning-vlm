"""Adaptation utilities: WiSE-FT weight interpolation and adapter loading.

WiSE-FT for LoRA simplifies to scaling the adapter's contribution.
At alpha=0 you get the pretrained model; at alpha=1 the full fine-tuned model.
Intermediate values blend them. This is a zero-cost operation — just a float
multiply on the scaling factor per layer, no weight copying.
"""

import torch
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from transformers import PreTrainedModel


def load_adapter(
    model: PreTrainedModel,
    adapter_path: str,
    adapter_name: str = "default",
) -> PeftModel:
    """Load a saved LoRA adapter into a base model.

    Args:
        model: Base model from load_vlm().
        adapter_path: Directory containing adapter_config.json and adapter_model.safetensors.
        adapter_name: Name for the adapter (default "default").

    Returns:
        PeftModel with the adapter attached (in eval mode, not trainable).
    """
    peft_model = PeftModel.from_pretrained(
        model,
        adapter_path,
        adapter_name=adapter_name,
        is_trainable=False,
    )

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"  Adapter loaded from {adapter_path}")
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return peft_model


def apply_wise_ft(
    model: PeftModel,
    alpha: float,
    adapter_name: str = "default",
) -> None:
    """Apply WiSE-FT by scaling the LoRA adapter contribution.

    With LoRA, the forward pass computes:
        output = base(x) + scaling * B(A(x))
    where scaling = lora_alpha / rank.

    set_scale(adapter_name, alpha) sets:
        scaling = alpha * (lora_alpha / rank)

    So alpha=0 gives the pretrained model, alpha=1 gives the fine-tuned model,
    and intermediate values interpolate between them. This is reversible —
    calling with a different alpha just overwrites the scale. No model reload needed.

    Args:
        model: PeftModel with LoRA adapter attached.
        alpha: Interpolation coefficient. 0.0=pretrained, 1.0=fine-tuned.
        adapter_name: Which adapter to scale.
    """
    assert 0.0 <= alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"

    n_scaled = 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.set_scale(adapter_name, alpha)
            n_scaled += 1

    assert n_scaled > 0, (
        f"No LoRA layers found in model. Is adapter '{adapter_name}' attached?"
    )
    print(f"  WiSE-FT: alpha={alpha:.2f} applied to {n_scaled} LoRA layers")
