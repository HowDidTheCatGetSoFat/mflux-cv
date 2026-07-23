import mlx.core as mx
from mlx import nn

from mflux.models.mage_flow.model.mage_flow_text_encoder.attention import (
    MageFlowQwen3VLAttention,
    MageFlowQwen3VLKVCache,
)
from mflux.models.mage_flow.model.mage_flow_text_encoder.layers import (
    MageFlowQwen3VLMLP,
    MageFlowQwen3VLRMSNorm,
)


class MageFlowQwen3VLDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool,
        rms_norm_eps: float,
        intermediate_size: int,
    ):
        super().__init__()
        self.input_layernorm = MageFlowQwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = MageFlowQwen3VLAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
        )
        self.post_attention_layernorm = MageFlowQwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MageFlowQwen3VLMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        *,
        use_cache: bool = False,
        past_key_value: MageFlowQwen3VLKVCache | None = None,
        max_cache_length: int | None = None,
    ) -> mx.array | tuple[mx.array, MageFlowQwen3VLKVCache]:
        attention_output = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            past_key_value=past_key_value,
            max_cache_length=max_cache_length,
        )
        present_key_value = None
        if use_cache:
            attention_output, present_key_value = attention_output
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        if present_key_value is not None:
            return hidden_states, present_key_value
        return hidden_states
