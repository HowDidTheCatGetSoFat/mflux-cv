import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.mage_flow.model.mage_flow_text_encoder.layers import MageFlowQwen3VLRMSNorm

MageFlowQwen3VLKVCache = tuple[mx.array, mx.array, int]


class MageFlowQwen3VLAttention(nn.Module):
    """Headless Qwen3-VL attention using MLX's native grouped-query kernel."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if num_attention_heads % num_key_value_heads:
            raise ValueError("query heads must be divisible by key/value heads")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scaling = 1.0 / math.sqrt(head_dim)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.q_norm = MageFlowQwen3VLRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = MageFlowQwen3VLRMSNorm(head_dim, eps=rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None,
        position_embeddings: tuple[mx.array, mx.array],
        *,
        use_cache: bool = False,
        past_key_value: MageFlowQwen3VLKVCache | None = None,
        max_cache_length: int | None = None,
    ) -> mx.array | tuple[mx.array, MageFlowQwen3VLKVCache]:
        batch_size, sequence_length, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        )
        key_states = self.k_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states = self.v_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )

        query_states = self.q_norm(query_states).transpose(0, 2, 1, 3)
        key_states = self.k_norm(key_states).transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)
        query_states, key_states = self._apply_rotary_pos_emb(
            query_states,
            key_states,
            *position_embeddings,
        )

        present_key_value = None
        if use_cache:
            key_states, value_states, present_key_value = self._update_cache(
                key_states,
                value_states,
                past_key_value=past_key_value,
                max_cache_length=max_cache_length,
            )

        # MLX SDPA performs the softmax in FP32 and directly supports grouped
        # query attention, so the 8 KV heads stay unexpanded for all 32 Q heads.
        attention_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scaling,
            mask=attention_mask,
        )
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size,
            sequence_length,
            self.num_attention_heads * self.head_dim,
        )
        attention_output = self.o_proj(attention_output)
        if present_key_value is not None:
            return attention_output, present_key_value
        return attention_output

    @staticmethod
    def _update_cache(
        key_states: mx.array,
        value_states: mx.array,
        *,
        past_key_value: MageFlowQwen3VLKVCache | None,
        max_cache_length: int | None,
    ) -> tuple[mx.array, mx.array, MageFlowQwen3VLKVCache]:
        sequence_length = key_states.shape[2]
        if past_key_value is None:
            cache_length = sequence_length
            if max_cache_length is None:
                cache_key_states = key_states
                cache_value_states = value_states
            else:
                if sequence_length > max_cache_length:
                    raise ValueError("the Qwen3-VL prefill exceeds max_cache_length")
                cache_shape = (
                    key_states.shape[0],
                    key_states.shape[1],
                    max_cache_length,
                    key_states.shape[3],
                )
                cache_key_states = mx.zeros(cache_shape, dtype=key_states.dtype)
                cache_value_states = mx.zeros(cache_shape, dtype=value_states.dtype)
                cache_key_states = mx.slice_update(
                    cache_key_states,
                    key_states,
                    start_indices=mx.array([0, 0, 0, 0], dtype=mx.int32),
                    axes=(0, 1, 2, 3),
                )
                cache_value_states = mx.slice_update(
                    cache_value_states,
                    value_states,
                    start_indices=mx.array([0, 0, 0, 0], dtype=mx.int32),
                    axes=(0, 1, 2, 3),
                )
        else:
            cache_key_states, cache_value_states, existing_length = past_key_value
            cache_length = existing_length + sequence_length
            if cache_length > cache_key_states.shape[2]:
                raise ValueError("the Qwen3-VL decode exceeds its preallocated KV cache")
            cache_key_states = mx.slice_update(
                cache_key_states,
                key_states,
                start_indices=mx.array([0, 0, existing_length, 0], dtype=mx.int32),
                axes=(0, 1, 2, 3),
            )
            cache_value_states = mx.slice_update(
                cache_value_states,
                value_states,
                start_indices=mx.array([0, 0, existing_length, 0], dtype=mx.int32),
                axes=(0, 1, 2, 3),
            )

        valid_key_states = cache_key_states[:, :, :cache_length]
        valid_value_states = cache_value_states[:, :, :cache_length]
        return (
            valid_key_states,
            valid_value_states,
            (cache_key_states, cache_value_states, cache_length),
        )

    @staticmethod
    def _apply_rotary_pos_emb(
        query_states: mx.array,
        key_states: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> tuple[mx.array, mx.array]:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        query_embed = query_states * cos + MageFlowQwen3VLAttention._rotate_half(query_states) * sin
        key_embed = key_states * cos + MageFlowQwen3VLAttention._rotate_half(key_states) * sin
        return query_embed, key_embed

    @staticmethod
    def _rotate_half(hidden_states: mx.array) -> mx.array:
        half = hidden_states.shape[-1] // 2
        return mx.concatenate([-hidden_states[..., half:], hidden_states[..., :half]], axis=-1)
