import mlx.core as mx
from mlx import nn


class MageFlowGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return nn.gelu_approx(self.proj(hidden_states))


class MageFlowFeedForward(nn.Module):
    """Diffusers-compatible GELU feed-forward module.

    The plain list deliberately preserves checkpoint paths ``net.0.proj`` and
    ``net.2`` instead of inserting an MLX ``Sequential.layers`` component.
    """

    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        self.net: list[nn.Module] = [
            MageFlowGELU(dim_in=dim, dim_out=inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=True),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states
