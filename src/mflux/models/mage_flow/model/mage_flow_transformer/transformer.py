from collections.abc import Sequence

import mlx.core as mx
from mlx import nn

from mflux.models.mage_flow.model.mage_flow_transformer.normalization import (
    MageFlowLayerNorm,
    MageFlowRMSNorm,
)
from mflux.models.mage_flow.model.mage_flow_transformer.rope_embedder import ImageShape, MageFlowEmbedRope
from mflux.models.mage_flow.model.mage_flow_transformer.timestep_embedder import MageFlowTimeTextEmbed
from mflux.models.mage_flow.model.mage_flow_transformer.transformer_block import MageFlowTransformerBlock


class MageFlowAdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int = 3072, conditioning_embedding_dim: int = 3072, eps: float = 1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=True)
        self.norm = MageFlowLayerNorm(embedding_dim, eps=eps)

    def __call__(self, hidden_states: mx.array, conditioning: mx.array) -> mx.array:
        modulation = self.linear(self.silu(conditioning).astype(hidden_states.dtype))
        scale = modulation[:, : self.embedding_dim]
        shift = modulation[:, self.embedding_dim :]
        return self.norm(hidden_states) * (1 + scale[:, None, :]) + shift[:, None, :]


class MageFlowTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        context_in_dim: int = 2560,
        hidden_size: int = 3072,
        num_attention_heads: int = 24,
        depth: int = 12,
        axes_dim: Sequence[int] = (16, 56, 56),
    ):
        super().__init__()
        attention_head_dim = hidden_size // num_attention_heads
        if sum(axes_dim) != attention_head_dim:
            raise ValueError("RoPE axis dimensions must sum to the attention head dimension")

        self.pos_embed = MageFlowEmbedRope(theta=10000.0, axes_dim=axes_dim)
        self.img_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.txt_norm = MageFlowRMSNorm(context_in_dim, eps=1e-6)
        self.txt_in = nn.Linear(context_in_dim, hidden_size, bias=True)
        self.time_text_embed = MageFlowTimeTextEmbed(embedding_dim=hidden_size)
        self.transformer_blocks = [
            MageFlowTransformerBlock(
                dim=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(depth)
        ]
        self.norm_out = MageFlowAdaLayerNormContinuous(hidden_size, hidden_size)
        self.proj_out = nn.Linear(hidden_size, out_channels, bias=True)

    def __call__(
        self,
        img: mx.array,
        txt: mx.array,
        timesteps: mx.array | float,
        img_shapes: ImageShape | Sequence[ImageShape] | Sequence[Sequence[ImageShape]],
        text_attention_mask: mx.array | None = None,
        image_rotary_emb: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("img and txt must both have shape [batch, sequence, channels]")

        if txt.shape[0] == 2 * img.shape[0]:
            img = mx.concatenate([img, img], axis=0)
        if txt.shape[0] != img.shape[0]:
            raise ValueError("text batch must equal the image batch, or be twice it for CFG")

        if image_rotary_emb is None:
            image_rotary_emb = self.pos_embed(img_shapes)
        if image_rotary_emb[0].shape[0] != img.shape[1]:
            raise ValueError("image RoPE length does not match the image token sequence")

        img = self.img_in(img)
        txt = self.txt_in(self.txt_norm(txt))
        timestep_array = self._prepare_timesteps(timesteps, batch_size=img.shape[0], dtype=img.dtype)
        temb = self.time_text_embed(timestep_array, dtype=img.dtype)
        attention_mask = self._prepare_attention_mask(
            text_attention_mask=text_attention_mask,
            image_length=img.shape[1],
            batch_size=img.shape[0],
            dtype=img.dtype,
        )

        for block in self.transformer_blocks:
            txt, img = block(
                hidden_states=img,
                encoder_hidden_states=txt,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
            )

        return self.proj_out(self.norm_out(img, temb))

    @staticmethod
    def _prepare_timesteps(
        timesteps: mx.array | float,
        batch_size: int,
        dtype: mx.Dtype,
    ) -> mx.array:
        if not isinstance(timesteps, mx.array):
            timesteps = mx.array(timesteps)
        if timesteps.ndim == 0:
            timesteps = mx.broadcast_to(timesteps, (batch_size,))
        elif timesteps.ndim == 1 and timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = mx.broadcast_to(timesteps, (batch_size,))
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must be scalar or have one value per batch element")
        return timesteps.astype(dtype)

    @staticmethod
    def _prepare_attention_mask(
        text_attention_mask: mx.array | None,
        image_length: int,
        batch_size: int,
        dtype: mx.Dtype,
    ) -> mx.array | None:
        if text_attention_mask is None:
            return None
        if text_attention_mask.ndim != 2 or text_attention_mask.shape[0] != batch_size:
            raise ValueError("text_attention_mask must have shape [batch, text_sequence]")

        image_mask = mx.ones((batch_size, image_length), dtype=text_attention_mask.dtype)
        key_is_valid = mx.concatenate([text_attention_mask, image_mask], axis=-1).astype(mx.bool_)
        zero = mx.array(0.0, dtype=dtype)
        masked = mx.array(-1e9, dtype=dtype)
        return mx.where(key_is_valid[:, None, None, :], zero, masked)
