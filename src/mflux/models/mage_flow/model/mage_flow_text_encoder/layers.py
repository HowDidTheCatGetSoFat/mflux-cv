import mlx.core as mx
from mlx import nn


class MageFlowQwen3VLRMSNorm(nn.Module):
    """Qwen3-VL RMSNorm using MLX's fused, BF16-parity kernel."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return mx.fast.rms_norm(hidden_states, self.weight, self.eps)


class MageFlowQwen3VLMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MageFlowQwen3VLVisionMLP(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4096):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(hidden_states)))


class MageFlowQwen3VLVisionPatchMerger(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        spatial_merge_size: int = 2,
        out_hidden_size: int = 2560,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size * spatial_merge_size**2
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_width = self.hidden_size if use_postshuffle_norm else hidden_size
        self.norm = nn.LayerNorm(norm_width, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self.use_postshuffle_norm:
            hidden_states = self.norm(hidden_states.reshape(-1, self.hidden_size))
        else:
            hidden_states = self.norm(hidden_states).reshape(-1, self.hidden_size)
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(hidden_states)))
