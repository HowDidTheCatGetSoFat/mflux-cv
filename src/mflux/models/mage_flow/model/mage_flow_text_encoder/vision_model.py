from collections.abc import Sequence

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_patch_embed import Qwen3VLVisionPatchEmbed
from mflux.models.mage_flow.model.mage_flow_text_encoder.layers import (
    MageFlowQwen3VLVisionMLP,
    MageFlowQwen3VLVisionPatchMerger,
)


class MageFlowQwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10_000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, sequence_length: int) -> mx.array:
        inv_freq = 1.0 / (self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        return mx.outer(mx.arange(sequence_length, dtype=mx.float32), inv_freq)


class MageFlowQwen3VLVisionAttention(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_heads: int = 16):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("vision hidden size must be divisible by the number of heads")
        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: Sequence[int],
        position_embeddings: tuple[mx.array, mx.array],
    ) -> mx.array:
        sequence_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query_states, key_states, value_states = mx.split(qkv, 3, axis=1)
        query_states = query_states.squeeze(1)
        key_states = key_states.squeeze(1)
        value_states = value_states.squeeze(1)

        # Hugging Face intentionally performs the vision rotary multiply in
        # FP32, then returns q/k to their original checkpoint dtype.
        query_dtype = query_states.dtype
        query_states, key_states = self._apply_rotary_pos_emb(
            query_states.astype(mx.float32),
            key_states.astype(mx.float32),
            position_embeddings[0].astype(mx.float32),
            position_embeddings[1].astype(mx.float32),
        )
        query_states = query_states.astype(query_dtype)
        key_states = key_states.astype(query_dtype)

        outputs = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
            query = query_states[start:end].transpose(1, 0, 2)[None, ...]
            key = key_states[start:end].transpose(1, 0, 2)[None, ...]
            value = value_states[start:end].transpose(1, 0, 2)[None, ...]
            output = scaled_dot_product_attention(
                query,
                key,
                value,
                scale=self.scaling,
            )
            outputs.append(output.squeeze(0).transpose(1, 0, 2))

        attention_output = mx.concatenate(outputs, axis=0).reshape(sequence_length, self.dim)
        return self.proj(attention_output)

    @staticmethod
    def _apply_rotary_pos_emb(
        query_states: mx.array,
        key_states: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> tuple[mx.array, mx.array]:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
        query_embed = query_states * cos + MageFlowQwen3VLVisionAttention._rotate_half(query_states) * sin
        key_embed = key_states * cos + MageFlowQwen3VLVisionAttention._rotate_half(key_states) * sin
        return query_embed, key_embed

    @staticmethod
    def _rotate_half(hidden_states: mx.array) -> mx.array:
        half = hidden_states.shape[-1] // 2
        return mx.concatenate([-hidden_states[..., half:], hidden_states[..., :half]], axis=-1)


class MageFlowQwen3VLVisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        hidden_act: str = "gelu_pytorch_tanh",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = MageFlowQwen3VLVisionAttention(hidden_size=hidden_size, num_heads=num_heads)
        if hidden_act != "gelu_pytorch_tanh":
            raise ValueError("Mage-Flow Qwen3-VL vision blocks require gelu_pytorch_tanh")
        self.mlp = MageFlowQwen3VLVisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: Sequence[int],
        position_embeddings: tuple[mx.array, mx.array],
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MageFlowQwen3VLVisionModel(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1024,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        depth: int = 24,
        spatial_merge_size: int = 2,
        num_position_embeddings: int = 2304,
        out_hidden_size: int = 2560,
        deepstack_visual_indexes: Sequence[int] = (5, 11, 17),
        hidden_act: str = "gelu_pytorch_tanh",
    ):
        super().__init__()
        if int(num_position_embeddings**0.5) ** 2 != num_position_embeddings:
            raise ValueError("num_position_embeddings must be a perfect square")
        if any(index < 0 or index >= depth for index in deepstack_visual_indexes):
            raise ValueError("DeepStack indexes must refer to vision blocks")

        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.spatial_merge_unit = spatial_merge_size**2
        self.patch_embed = Qwen3VLVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings**0.5)
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = MageFlowQwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = [
            MageFlowQwen3VLVisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
            )
            for _ in range(depth)
        ]
        self.merger = MageFlowQwen3VLVisionPatchMerger(
            hidden_size=hidden_size,
            spatial_merge_size=spatial_merge_size,
            out_hidden_size=out_hidden_size,
            use_postshuffle_norm=False,
        )
        self.deepstack_visual_indexes = list(deepstack_visual_indexes)
        self.deepstack_merger_list = [
            MageFlowQwen3VLVisionPatchMerger(
                hidden_size=hidden_size,
                spatial_merge_size=spatial_merge_size,
                out_hidden_size=out_hidden_size,
                use_postshuffle_norm=True,
            )
            for _ in self.deepstack_visual_indexes
        ]

    def __call__(
        self,
        pixel_values: mx.array,
        grid_thw: mx.array,
        return_deepstack: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        grids = np.asarray(grid_thw).astype(np.int64, copy=False)
        if grids.ndim != 2 or grids.shape[1] != 3:
            raise ValueError("grid_thw must have shape [number_of_images, 3]")

        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = self._fast_pos_embed_interpolate(
            self.spatial_merge_size,
            self.pos_embed,
            self.num_grid_per_side,
            grids,
        )
        hidden_states = hidden_states + pos_embeds
        rotary = self._rot_pos_emb(
            self.rotary_pos_emb,
            self.spatial_merge_size,
            grids,
        )
        embeddings = mx.concatenate([rotary, rotary], axis=-1)
        position_embeddings = (mx.cos(embeddings), mx.sin(embeddings))

        cu_seqlens = [0]
        for grid_t, grid_h, grid_w in grids:
            frame_length = int(grid_h * grid_w)
            for _ in range(int(grid_t)):
                cu_seqlens.append(cu_seqlens[-1] + frame_length)

        deepstack_image_embeds = [] if return_deepstack else None
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if return_deepstack and layer_index in self.deepstack_visual_indexes:
                deepstack_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack_image_embeds.append(self.deepstack_merger_list[deepstack_index](hidden_states))

        return self.merger(hidden_states), deepstack_image_embeds

    @staticmethod
    def _fast_pos_embed_interpolate(
        spatial_merge_size: int,
        pos_embed: nn.Embedding,
        num_grid_per_side: int,
        grids: np.ndarray,
    ) -> mx.array:
        indices_per_corner: list[list[np.ndarray]] = [[] for _ in range(4)]
        weights_per_corner: list[list[np.ndarray]] = [[] for _ in range(4)]
        for _, grid_h, grid_w in grids:
            height = int(grid_h)
            width = int(grid_w)
            height_indices = np.linspace(0, num_grid_per_side - 1, height, dtype=np.float32)
            width_indices = np.linspace(0, num_grid_per_side - 1, width, dtype=np.float32)
            height_floor = np.floor(height_indices).astype(np.int32)
            width_floor = np.floor(width_indices).astype(np.int32)
            height_ceil = np.clip(height_floor + 1, 0, num_grid_per_side - 1)
            width_ceil = np.clip(width_floor + 1, 0, num_grid_per_side - 1)
            delta_height = height_indices - height_floor
            delta_width = width_indices - width_floor

            base_height = height_floor * num_grid_per_side
            base_height_ceil = height_ceil * num_grid_per_side
            corner_indices = [
                (base_height[:, None] + width_floor[None, :]).reshape(-1),
                (base_height[:, None] + width_ceil[None, :]).reshape(-1),
                (base_height_ceil[:, None] + width_floor[None, :]).reshape(-1),
                (base_height_ceil[:, None] + width_ceil[None, :]).reshape(-1),
            ]
            corner_weights = [
                ((1 - delta_height)[:, None] * (1 - delta_width)[None, :]).reshape(-1),
                ((1 - delta_height)[:, None] * delta_width[None, :]).reshape(-1),
                (delta_height[:, None] * (1 - delta_width)[None, :]).reshape(-1),
                (delta_height[:, None] * delta_width[None, :]).reshape(-1),
            ]
            for corner in range(4):
                indices_per_corner[corner].append(corner_indices[corner])
                weights_per_corner[corner].append(corner_weights[corner])

        index_array = mx.array(
            np.stack([np.concatenate(values) for values in indices_per_corner]),
            dtype=mx.int32,
        )
        weight_array = mx.array(
            np.stack([np.concatenate(values) for values in weights_per_corner]),
            dtype=pos_embed.weight.dtype,
        )
        corner_embeddings = pos_embed(index_array) * weight_array[..., None]
        interpolated = mx.sum(corner_embeddings, axis=0)

        outputs = []
        start = 0
        for grid_t, grid_h, grid_w in grids:
            temporal = int(grid_t)
            height = int(grid_h)
            width = int(grid_w)
            end = start + height * width
            image_positions = interpolated[start:end]
            start = end
            image_positions = mx.tile(image_positions, (temporal, 1))
            image_positions = image_positions.reshape(
                temporal,
                height // spatial_merge_size,
                spatial_merge_size,
                width // spatial_merge_size,
                spatial_merge_size,
                -1,
            )
            image_positions = image_positions.transpose(0, 1, 3, 2, 4, 5)
            outputs.append(image_positions.reshape(-1, image_positions.shape[-1]))
        return mx.concatenate(outputs)

    @staticmethod
    def _rot_pos_emb(
        rotary_pos_emb: MageFlowQwen3VLVisionRotaryEmbedding,
        spatial_merge_size: int,
        grids: np.ndarray,
    ) -> mx.array:
        position_pairs = []
        for grid_t, grid_h, grid_w in grids:
            temporal = int(grid_t)
            height = int(grid_h)
            width = int(grid_w)
            height_positions = np.broadcast_to(np.arange(height)[:, None], (height, width))
            width_positions = np.broadcast_to(np.arange(width)[None, :], (height, width))
            merged_shape = (
                height // spatial_merge_size,
                spatial_merge_size,
                width // spatial_merge_size,
                spatial_merge_size,
            )
            height_positions = height_positions.reshape(merged_shape).transpose(0, 2, 1, 3).reshape(-1)
            width_positions = width_positions.reshape(merged_shape).transpose(0, 2, 1, 3).reshape(-1)
            pairs = np.stack([height_positions, width_positions], axis=-1)
            position_pairs.append(np.tile(pairs, (temporal, 1)))

        position_ids = np.concatenate(position_pairs)
        rotary = rotary_pos_emb(int(np.max(grids[:, 1:])))
        height_embeddings = rotary[mx.array(position_ids[:, 0], dtype=mx.int32)]
        width_embeddings = rotary[mx.array(position_ids[:, 1], dtype=mx.int32)]
        return mx.stack([height_embeddings, width_embeddings], axis=1).reshape(position_ids.shape[0], -1)
