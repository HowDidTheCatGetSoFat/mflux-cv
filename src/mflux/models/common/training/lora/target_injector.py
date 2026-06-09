from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.training.lora.path_util import (
    expand_module_paths_from_targets,
    get_at_path,
    set_at_path,
)
from mflux.models.common.training.state.training_spec import LoraTargetSpec


def _fp8_to_bf16_linear(fp8) -> nn.Linear:
    # Dequantize an Fp8Linear (e4m3 uint8 weights + per-row float32 scale) to a plain bf16
    # nn.Linear, exactly like ai-toolkit's _dequantize_fp8_state_dict. We train the LoRA over
    # this clean bf16 base instead of the raw fp8 layer: fp8's 3-bit-mantissa per-channel
    # rounding error on the frozen base is a source of painterly texture in the trained LoRA.
    w = mx.from_fp8(fp8.weight, dtype=mx.float32) * fp8.weight_scale.astype(mx.float32)[:, None]
    lin = nn.Linear(fp8.in_features, fp8.out_features, bias=fp8.bias is not None)
    lin.weight = w.astype(mx.bfloat16)
    if fp8.bias is not None:
        lin.bias = fp8.bias.astype(mx.bfloat16)
    return lin


def inject_lora_targets(transformer: Any, targets: list[LoraTargetSpec]) -> None:
    # Ideogram-4 ships fp8-quantized layers (Fp8Linear). We recognize them, and (below)
    # dequantize each LoRA-target layer to a bf16 nn.Linear before wrapping, so the LoRA
    # trains over a clean bf16 base like ai-toolkit (not over the lossy fp8 layer).
    # Late, guarded import to avoid an import cycle / hard dependency on the ideogram package.
    try:
        from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear

        linear_types: tuple = (nn.Linear, nn.QuantizedLinear, Fp8Linear)
    except Exception:
        Fp8Linear = None
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
            base = current
            if Fp8Linear is not None and isinstance(current, Fp8Linear):
                base = _fp8_to_bf16_linear(current)  # train over a clean bf16 base, not fp8
            wrapped = LoRALinear.from_linear(base, r=rank)
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
