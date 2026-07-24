import mlx.core as mx
from mlx import nn

from mflux.models.mage_flow.model.mage_flow_transformer.attention import MageFlowJointAttention
from mflux.models.mage_flow.model.mage_flow_transformer.feed_forward import MageFlowFeedForward
from mflux.models.mage_flow.model.mage_flow_transformer.normalization import MageFlowLayerNorm


class MageFlowTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 3072,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim

        self.img_mod: list[nn.Module] = [nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)]
        self.img_norm1 = MageFlowLayerNorm(dim, eps=eps)
        self.attn = MageFlowJointAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            eps=eps,
        )
        self.img_norm2 = MageFlowLayerNorm(dim, eps=eps)
        self.img_mlp = MageFlowFeedForward(dim=dim, dim_out=dim)

        self.txt_mod: list[nn.Module] = [nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)]
        self.txt_norm1 = MageFlowLayerNorm(dim, eps=eps)
        self.txt_norm2 = MageFlowLayerNorm(dim, eps=eps)
        self.txt_mlp = MageFlowFeedForward(dim=dim, dim_out=dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        image_modulation = self.img_mod[1](self.img_mod[0](temb))
        text_modulation = self.txt_mod[1](self.txt_mod[0](temb))
        image_mod_1, image_mod_2 = self._split_modulation(image_modulation)
        text_mod_1, text_mod_2 = self._split_modulation(text_modulation)

        image_input, image_gate_1 = self._modulate(self.img_norm1(hidden_states), image_mod_1)
        text_input, text_gate_1 = self._modulate(self.txt_norm1(encoder_hidden_states), text_mod_1)
        image_attention, text_attention = self.attn(
            hidden_states=image_input,
            encoder_hidden_states=text_input,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + image_gate_1[:, None, :] * image_attention
        encoder_hidden_states = encoder_hidden_states + text_gate_1[:, None, :] * text_attention

        image_input, image_gate_2 = self._modulate(self.img_norm2(hidden_states), image_mod_2)
        text_input, text_gate_2 = self._modulate(self.txt_norm2(encoder_hidden_states), text_mod_2)
        hidden_states = hidden_states + image_gate_2[:, None, :] * self.img_mlp(image_input)
        encoder_hidden_states = encoder_hidden_states + text_gate_2[:, None, :] * self.txt_mlp(text_input)

        return encoder_hidden_states, hidden_states

    @staticmethod
    def _split_modulation(modulation: mx.array) -> tuple[mx.array, mx.array]:
        split = modulation.shape[-1] // 2
        return modulation[..., :split], modulation[..., split:]

    @staticmethod
    def _modulate(
        hidden_states: mx.array,
        modulation: mx.array,
    ) -> tuple[mx.array, mx.array]:
        chunk = modulation.shape[-1] // 3
        shift = modulation[..., :chunk]
        scale = modulation[..., chunk : 2 * chunk]
        gate = modulation[..., 2 * chunk :]
        return hidden_states * (1 + scale[:, None, :]) + shift[:, None, :], gate
