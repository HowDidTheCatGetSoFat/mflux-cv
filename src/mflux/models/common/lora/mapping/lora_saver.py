import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear


def _is_fp8_base(linear) -> bool:
    # fp8 bases store raw uint8 codes in .weight plus a per-row weight_scale — a float delta
    # CANNOT be folded into the codes directly (`merged.astype(uint8)` rounds it away). They
    # are baked by dequantizing once and requantizing to MLX q8 (see _fold_fp8_delta_to_q8).
    weight = getattr(linear, "weight", None)
    return (
        weight is not None
        and weight.dtype == mx.uint8
        and hasattr(linear, "weight_scale")
        and not isinstance(linear, nn.QuantizedLinear)
    )


def _fold_fp8_delta_to_q8(base_linear, delta: mx.array) -> nn.Module:
    # Dequantize the fp8 base ONCE, add the LoRA delta in float, requantize to MLX q8
    # (group-64 affine ≈ more mantissa than fp8-e4m3, so no quality loss). Besides making
    # the bake CORRECT on fp8, this replaces Fp8Linear's per-forward full-matrix
    # dequantization with MLX's fused quantized matmul kernel — substantially faster.
    dense = mx.from_fp8(base_linear.weight, dtype=mx.float32) * base_linear.weight_scale[:, None]
    merged = dense + delta.astype(mx.float32)
    bias = getattr(base_linear, "bias", None)
    compute_dtype = getattr(base_linear, "compute_dtype", mx.bfloat16)
    linear = nn.Linear(merged.shape[1], merged.shape[0], bias=bias is not None)
    linear.weight = merged.astype(compute_dtype)
    if bias is not None:
        linear.bias = bias
    quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)
    mx.eval(quantized.parameters())
    return quantized


class LoRASaver:
    @staticmethod
    def bake_and_strip_lora(module: nn.Module) -> nn.Module:
        def _assign(parent, attr_name, idx, new_child):
            if parent is None:
                return
            if isinstance(parent, list) and idx is not None:
                parent[idx] = new_child
            elif isinstance(parent, dict) and attr_name is not None:
                parent[attr_name] = new_child
            elif attr_name is not None:
                setattr(parent, attr_name, new_child)

        def _bake_single(lora_layer: LoRALinear) -> nn.Module:
            return LoRASaver._bake_lora_into_linear(lora_layer.linear, lora_layer)

        def _bake_lokr(lokr_layer: LoKrLinear) -> nn.Module:
            return LoRASaver._bake_lokr_into_linear(lokr_layer.linear, lokr_layer)

        def _bake_fused(fused_layer: FusedLoRALinear) -> nn.Module:
            current = fused_layer.base_linear
            for lora in fused_layer.loras:
                if isinstance(lora, LoRALinear):
                    current = LoRASaver._bake_lora_into_linear(current, lora)
                elif isinstance(lora, LoKrLinear):
                    current = LoRASaver._bake_lokr_into_linear(current, lora)
            return current

        def _walk(obj, parent=None, attr_name=None, idx=None):
            # Replace wrappers first. fp8 bases are handled inside _bake_delta_into_linear
            # (dequantize once + fold + requantize to q8 — folding into the raw uint8 codes
            # would silently round the delta away).
            if isinstance(obj, FusedLoRALinear):
                new_child = _bake_fused(obj)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child
            elif isinstance(obj, LoKrLinear):
                new_child = _bake_lokr(obj)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child
            elif isinstance(obj, LoRALinear):
                new_child = _bake_single(obj)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child

            # Recurse into containers/modules
            if isinstance(obj, list):
                for i, child in enumerate(list(obj)):
                    _walk(child, obj, None, i)
            elif isinstance(obj, tuple):
                temp_list = list(obj)
                for i, child in enumerate(temp_list):
                    _walk(child, temp_list, None, i)
                if parent is not None:
                    _assign(parent, attr_name, idx, type(obj)(temp_list))
            elif isinstance(obj, dict):
                for key, child in list(obj.items()):
                    _walk(child, obj, key, None)
            elif isinstance(obj, nn.Module):
                for name, child in vars(obj).items():
                    if isinstance(child, (nn.Module, list, tuple, dict)):
                        _walk(child, obj, name, None)

        _walk(module, None, None, None)
        return module

    @staticmethod
    def _dense_weight(linear: nn.Linear | nn.QuantizedLinear) -> mx.array:
        if isinstance(linear, nn.QuantizedLinear):
            return mx.dequantize(
                linear.weight,
                linear.scales,
                biases=linear.biases,
                group_size=linear.group_size,
                bits=linear.bits,
                mode=linear.mode,
            )
        if _is_fp8_base(linear):
            return mx.from_fp8(linear.weight, dtype=mx.float32) * linear.weight_scale[:, None]
        return linear.weight

    @staticmethod
    def _bake_lora_into_linear(base_linear: nn.Linear | nn.QuantizedLinear, lora_layer: LoRALinear) -> nn.Module:
        if lora_layer.dora_scale is not None:
            # DoRA: the effective delta is weight-decomposed (magnitude x normalized direction), so
            # fold delta_weight() which already includes scale and the base-coupled normalization.
            dense_weight = LoRASaver._dense_weight(base_linear)
            delta = lora_layer.delta_weight(base_weight=dense_weight)
            return LoRASaver._bake_delta_into_linear(base_linear, delta)
        delta = mx.matmul(lora_layer.lora_A, lora_layer.lora_B)
        delta = mx.transpose(delta)
        delta = lora_layer.scale * delta
        return LoRASaver._bake_delta_into_linear(base_linear, delta)

    @staticmethod
    def _bake_lokr_into_linear(base_linear: nn.Linear | nn.QuantizedLinear, lokr_layer: LoKrLinear) -> nn.Module:
        dense_weight = LoRASaver._dense_weight(base_linear)
        delta = lokr_layer.scale * lokr_layer.delta_weight(base_weight=dense_weight)
        return LoRASaver._bake_delta_into_linear(base_linear, delta)

    @staticmethod
    def _bake_delta_into_linear(
        base_linear: nn.Linear | nn.QuantizedLinear,
        delta: mx.array,
    ) -> nn.Module:
        if not hasattr(base_linear, "weight"):
            return base_linear

        if _is_fp8_base(base_linear):
            return _fold_fp8_delta_to_q8(base_linear, delta)

        dense_weight = LoRASaver._dense_weight(base_linear)
        if dense_weight.shape != delta.shape:
            print(
                "⚠️  Skipping LoRA bake due to shape mismatch: "
                f"weight {dense_weight.shape} vs delta {delta.shape}"
            )
            return base_linear

        merged = dense_weight + delta.astype(dense_weight.dtype)

        try:
            if isinstance(base_linear, nn.QuantizedLinear):
                has_bias = hasattr(base_linear, "bias") and getattr(base_linear, "bias", None) is not None
                dense_linear = nn.Linear(merged.shape[1], merged.shape[0], bias=has_bias)
                dense_linear.weight = merged
                if has_bias:
                    dense_linear.bias = base_linear.bias
                return nn.QuantizedLinear.from_linear(
                    dense_linear,
                    group_size=base_linear.group_size,
                    bits=base_linear.bits,
                    mode=base_linear.mode,
                )

            base_linear.weight = merged.astype(base_linear.weight.dtype)
            return base_linear
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Failed to bake LoRA into base layer: {e}")
            return base_linear
