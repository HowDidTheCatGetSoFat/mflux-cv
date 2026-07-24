import math

import mlx.core as mx
from mlx import nn


class MageFlowTimesteps(nn.Module):
    """Mage's BF16-rounded sinusoidal timestep projection."""

    def __init__(
        self,
        num_channels: int = 256,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
        scale: float = 1000.0,
        max_period: int = 10000,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def __call__(self, timesteps: mx.array) -> mx.array:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be a one-dimensional array")

        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * mx.arange(half_dim, dtype=mx.float32)
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        # This cast is intentional. The released model was trained with its
        # frequency table rounded to the timestep dtype (normally BF16).
        frequencies = mx.exp(exponent).astype(timesteps.dtype)
        angles = timesteps[:, None].astype(mx.float32) * frequencies[None, :].astype(mx.float32)
        angles = self.scale * angles

        embedding = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)
        if self.flip_sin_to_cos:
            embedding = mx.concatenate([embedding[:, half_dim:], embedding[:, :half_dim]], axis=-1)
        if self.num_channels % 2 == 1:
            embedding = mx.pad(embedding, ((0, 0), (0, 1)))
        return embedding


class MageFlowTimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int = 256, time_embed_dim: int = 3072):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(sample)))


class MageFlowTimeTextEmbed(nn.Module):
    def __init__(self, embedding_dim: int = 3072):
        super().__init__()
        self.time_proj = MageFlowTimesteps()
        self.timestep_embedder = MageFlowTimestepEmbedding(time_embed_dim=embedding_dim)

    def __call__(self, timestep: mx.array, dtype: mx.Dtype) -> mx.array:
        projected = self.time_proj(timestep)
        return self.timestep_embedder(projected.astype(dtype))
