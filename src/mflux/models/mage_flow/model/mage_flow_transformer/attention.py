import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.mage_flow.model.mage_flow_transformer.normalization import MageFlowRMSNorm


class MageFlowJointAttention(nn.Module):
    def __init__(
        self,
        dim: int = 3072,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.head_dim = attention_head_dim
        self.num_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.scale = 1.0 / math.sqrt(attention_head_dim)

        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=True)
        self.norm_q = MageFlowRMSNorm(self.head_dim, eps=eps)
        self.norm_k = MageFlowRMSNorm(self.head_dim, eps=eps)

        self.add_q_proj = nn.Linear(dim, self.inner_dim, bias=True)
        self.add_k_proj = nn.Linear(dim, self.inner_dim, bias=True)
        self.add_v_proj = nn.Linear(dim, self.inner_dim, bias=True)
        self.norm_added_q = MageFlowRMSNorm(self.head_dim, eps=eps)
        self.norm_added_k = MageFlowRMSNorm(self.head_dim, eps=eps)

        self.to_out: list[nn.Module] = [nn.Linear(self.inner_dim, dim, bias=True)]
        self.to_add_out = nn.Linear(self.inner_dim, dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        batch_size, image_length, _ = hidden_states.shape
        text_length = encoder_hidden_states.shape[1]

        image_query = self._split_heads(self.to_q(hidden_states))
        image_key = self._split_heads(self.to_k(hidden_states))
        image_value = self._split_heads(self.to_v(hidden_states))
        text_query = self._split_heads(self.add_q_proj(encoder_hidden_states))
        text_key = self._split_heads(self.add_k_proj(encoder_hidden_states))
        text_value = self._split_heads(self.add_v_proj(encoder_hidden_states))

        image_query = self.norm_q(image_query)
        image_key = self.norm_k(image_key)
        text_query = self.norm_added_q(text_query)
        text_key = self.norm_added_k(text_key)

        image_query = self.apply_rotary_emb(image_query, image_rotary_emb)
        image_key = self.apply_rotary_emb(image_key, image_rotary_emb)

        query = mx.concatenate([text_query, image_query], axis=1).transpose(0, 2, 1, 3)
        key = mx.concatenate([text_key, image_key], axis=1).transpose(0, 2, 1, 3)
        value = mx.concatenate([text_value, image_value], axis=1).transpose(0, 2, 1, 3)

        attended = scaled_dot_product_attention(query, key, value, scale=self.scale, mask=attention_mask)
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch_size,
            text_length + image_length,
            self.inner_dim,
        )
        text_attended = self.to_add_out(attended[:, :text_length])
        image_attended = self.to_out[0](attended[:, text_length:])
        return image_attended, text_attended

    def _split_heads(self, hidden_states: mx.array) -> mx.array:
        return hidden_states.reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.num_heads,
            self.head_dim,
        )

    @staticmethod
    def apply_rotary_emb(
        hidden_states: mx.array,
        rotary_emb: tuple[mx.array, mx.array],
    ) -> mx.array:
        """Apply adjacent-pair complex RoPE in FP32, matching ``view_as_complex``."""

        cos, sin = rotary_emb
        input_dtype = hidden_states.dtype
        pairs = hidden_states.astype(mx.float32).reshape(*hidden_states.shape[:-1], -1, 2)
        real = pairs[..., 0]
        imaginary = pairs[..., 1]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        rotated = mx.stack(
            [
                real * cos - imaginary * sin,
                real * sin + imaginary * cos,
            ],
            axis=-1,
        )
        return rotated.reshape(hidden_states.shape).astype(input_dtype)
