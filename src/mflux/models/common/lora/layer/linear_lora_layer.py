import math

import mlx.core as mx
from mlx import nn


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear | nn.QuantizedLinear,
        r: int = 16,
        scale: float = 1.0,
    ):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.scale = scale
        self.output_dims = output_dims
        # DoRA magnitude vector (per output channel); None = plain LoRA. Set at injection (or loaded
        # from an adapter) when DoRA is enabled, initialized to the base weight's per-output-row norm.
        self.dora_scale: mx.array | None = None
        init = 1 / math.sqrt(input_dims)

        self.lora_A = mx.random.uniform(
            low=-init,
            high=init,
            shape=(input_dims, r),
        )
        # Zero-init the up matrix so the LoRA is identity at step 0 (standard PEFT /
        # ai-toolkit convention). Random-B perturbs the frozen base off-distribution at
        # init, which (with few images) bakes in a washed/painterly look.
        self.lora_B = mx.zeros((r, output_dims))

    def _dense_base_weight(self) -> mx.array:
        # Materialize the frozen base weight (out, in), dequantizing / de-fp8-ing as needed.
        linear = self.linear
        if isinstance(linear, nn.QuantizedLinear):
            return mx.dequantize(
                linear.weight,
                linear.scales,
                biases=linear.biases,
                group_size=linear.group_size,
                bits=linear.bits,
                mode=linear.mode,
            )
        # Ideogram's Fp8Linear stores raw fp8 + a per-output scale.
        if hasattr(linear, "weight_scale"):
            return mx.from_fp8(linear.weight, dtype=mx.float32) * linear.weight_scale[:, None]
        return linear.weight

    def delta_weight(self, base_weight: mx.array | None = None) -> mx.array:
        # Scaled LoRA weight delta (out, in). For DoRA, decompose the combined weight into a trained
        # per-output magnitude times the normalized direction, and return the effective delta.
        delta = self.scale * mx.matmul(self.lora_A, self.lora_B).T
        if self.dora_scale is None:
            return delta
        if base_weight is None:
            base_weight = self._dense_base_weight()
        merged = base_weight + delta
        weight_norm = mx.linalg.norm(merged, axis=1, keepdims=True) + mx.finfo(merged.dtype).eps
        decomposed = merged * self.dora_scale.reshape((self.output_dims, 1)) / weight_norm
        return decomposed - base_weight

    def __call__(self, x):
        base_out = self.linear(x)
        if self.dora_scale is None:
            # Fast additive path (unchanged): plain LoRA never materializes the base weight.
            lora_out = mx.matmul(mx.matmul(x, self.lora_A), self.lora_B)
            return base_out + self.scale * lora_out
        # DoRA: base_out + x @ (W_dora - W0)^T  ==  x @ W_dora^T (+ bias).
        return base_out + mx.matmul(x, self.delta_weight().T)
