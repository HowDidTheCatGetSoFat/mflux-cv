from __future__ import annotations

from typing import Any

import mlx.nn as nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.training.lora.path_util import (
    expand_module_paths_from_targets,
    get_at_path,
    set_at_path,
)
from mflux.models.common.training.state.training_spec import LoraTargetSpec


def inject_lora_targets(transformer: Any, targets: list[LoraTargetSpec]) -> None:
    # Ideogram-4 ships fp8-quantized layers (Fp8Linear). We recognize them and wrap them
    # directly — LoRALinear.from_linear handles Fp8Linear (weight.shape is the true (out, in)),
    # and the frozen fp8 base is dequantized per-forward. We do NOT dequantize the whole base
    # to bf16 first: it is unnecessary (fp8 is not the source of the painterly texture — that is
    # overcooking) and on a 9B transformer the bf16 copy doubles resident memory, which causes
    # swap thrashing (~200s/step) on larger datasets.
    # Late, guarded import to avoid an import cycle / hard dependency on the ideogram package.
    try:
        from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear

        linear_types: tuple = (nn.Linear, nn.QuantizedLinear, Fp8Linear)
    except Exception:
        linear_types = (nn.Linear, nn.QuantizedLinear)

    expanded = expand_module_paths_from_targets(targets)
    for module_path, rank in expanded:
        current = get_at_path(transformer, module_path)

        # Skip if already has a trainable LoRA on this path
        if isinstance(current, LoRALinear):
            if getattr(current, "_mflux_lora_role", None) == "train":
                continue
        if isinstance(current, FusedLoRALinear):
            if any(getattr(lora, "_mflux_lora_role", None) == "train" for lora in current.loras):
                continue

        if isinstance(current, linear_types):
            wrapped = LoRALinear.from_linear(current, r=rank)
            wrapped._mflux_lora_role = "train"
            set_at_path(transformer, module_path, wrapped)
        elif isinstance(current, LoRALinear):
            # Fuse a new trainable LoRA on top of an existing LoRA (e.g. assistant adapter).
            train_lora = LoRALinear.from_linear(current.linear, r=rank)
            train_lora._mflux_lora_role = "train"
            fused = FusedLoRALinear(base_linear=current.linear, loras=[current, train_lora])
            set_at_path(transformer, module_path, fused)
        elif isinstance(current, FusedLoRALinear):
            # Add a new trainable LoRA to an existing fusion (e.g. multiple preloaded LoRAs).
            train_lora = LoRALinear.from_linear(current.base_linear, r=rank)
            train_lora._mflux_lora_role = "train"
            fused = FusedLoRALinear(base_linear=current.base_linear, loras=current.loras + [train_lora])
            set_at_path(transformer, module_path, fused)
        else:
            raise TypeError(
                f"LoRA target '{module_path}' must resolve to nn.Linear, nn.QuantizedLinear "
                f"or Fp8Linear, got {type(current)}"
            )
