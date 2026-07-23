import mlx.core as mx
from mlx import nn


class MageFlowRMSNorm(nn.Module):
    """RMSNorm with the FP32 accumulation used by the reference model."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype
        hidden_states_f32 = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(hidden_states_f32), axis=-1, keepdims=True)
        normalized = hidden_states_f32 * mx.rsqrt(variance + self.eps)
        return (normalized * self.weight.astype(mx.float32)).astype(input_dtype)


class MageFlowLayerNorm(nn.Module):
    """Parameter-free LayerNorm with FP32 accumulation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype
        hidden_states_f32 = hidden_states.astype(mx.float32)
        mean = mx.mean(hidden_states_f32, axis=-1, keepdims=True)
        variance = mx.mean(mx.square(hidden_states_f32 - mean), axis=-1, keepdims=True)
        return ((hidden_states_f32 - mean) * mx.rsqrt(variance + self.eps)).astype(input_dtype)
