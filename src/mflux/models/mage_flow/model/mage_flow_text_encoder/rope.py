from collections.abc import Sequence

import mlx.core as mx
from mlx import nn


class MageFlowQwen3VLRotaryEmbedding(nn.Module):
    """Qwen3-VL interleaved multimodal rotary embeddings.

    Qwen3-VL assigns separate temporal, height, and width positions to visual
    tokens. The three axes are interleaved across the rotary frequencies rather
    than concatenated into contiguous chunks.
    """

    def __init__(
        self,
        dim: int = 128,
        max_position_embeddings: int = 262144,
        base: float = 5_000_000.0,
        scaling_factor: float = 1.0,
        mrope_section: Sequence[int] = (24, 20, 20),
    ):
        super().__init__()
        if dim % 2:
            raise ValueError("rotary head dimension must be even")
        if len(mrope_section) != 3 or sum(mrope_section) != dim // 2:
            raise ValueError("mrope_section must contain three values summing to half the head dimension")

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.mrope_section = tuple(mrope_section)

        # Keep this as Python metadata so it does not become a checkpoint
        # parameter. Axis 0 is temporal/text, 1 is height, and 2 is width.
        axis_codes = [0] * (dim // 2)
        for index in range(1, min(self.mrope_section[1] * 3, dim // 2), 3):
            axis_codes[index] = 1
        for index in range(2, min(self.mrope_section[2] * 3, dim // 2), 3):
            axis_codes[index] = 2
        self._axis_codes = tuple(axis_codes)

    def __call__(self, hidden_states: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(position_ids[None, ...], (3, *position_ids.shape))
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("position_ids must have shape [batch, sequence] or [3, batch, sequence]")

        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        frequencies = position_ids.astype(mx.float32)[..., None] * inv_freq[None, None, None, :]

        axis_codes = mx.array(self._axis_codes, dtype=mx.int32)[None, None, :]
        interleaved = mx.where(
            axis_codes == 1,
            frequencies[1],
            mx.where(axis_codes == 2, frequencies[2], frequencies[0]),
        )
        embeddings = mx.concatenate([interleaved, interleaved], axis=-1)
        cos = mx.cos(embeddings) * self.scaling_factor
        sin = mx.sin(embeddings) * self.scaling_factor
        return cos.astype(hidden_states.dtype), sin.astype(hidden_states.dtype)
