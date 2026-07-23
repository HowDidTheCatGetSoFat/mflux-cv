from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.mage_flow.model.mage_flow_text_encoder.attention import MageFlowQwen3VLKVCache
from mflux.models.mage_flow.model.mage_flow_text_encoder.decoder_layer import MageFlowQwen3VLDecoderLayer
from mflux.models.mage_flow.model.mage_flow_text_encoder.layers import MageFlowQwen3VLRMSNorm
from mflux.models.mage_flow.model.mage_flow_text_encoder.rope import MageFlowQwen3VLRotaryEmbedding
from mflux.models.mage_flow.model.mage_flow_text_encoder.vision_model import MageFlowQwen3VLVisionModel

if TYPE_CHECKING:
    from mflux.models.mage_flow.model.mage_flow_text_encoder.policy import FilterVerdict


def build_mrope_position_ids(
    input_ids: mx.array,
    image_grid_thw: mx.array | None = None,
    attention_mask: mx.array | None = None,
    *,
    image_token_id: int = 151655,
    vision_start_token_id: int = 151652,
    spatial_merge_size: int = 2,
) -> tuple[mx.array, mx.array]:
    """Build the exact Qwen3-VL temporal/height/width position IDs.

    Returns ``(position_ids, rope_deltas)`` with shapes ``[3, B, L]`` and
    ``[B, 1]``. The latter is retained for parity with Qwen3-VL even though
    Mage-Flow performs a single conditioning pass and does not decode tokens.
    """

    ids = np.asarray(input_ids).astype(np.int64, copy=False)
    if ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    batch_size, sequence_length = ids.shape

    if attention_mask is None:
        mask = np.ones_like(ids, dtype=bool)
    else:
        mask = np.asarray(attention_mask).astype(bool, copy=False)
        if mask.shape != ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

    if image_grid_thw is None:
        if attention_mask is None:
            positions = np.broadcast_to(
                np.arange(sequence_length, dtype=np.int64)[None, None, :],
                (3, batch_size, sequence_length),
            ).copy()
            deltas = np.zeros((batch_size, 1), dtype=np.int64)
        else:
            text_positions = np.cumsum(mask, axis=-1, dtype=np.int64) - 1
            text_positions[~mask] = 1
            positions = np.broadcast_to(text_positions[None, ...], (3, *text_positions.shape)).copy()
            max_positions = text_positions.max(axis=-1, keepdims=True)
            deltas = max_positions + 1 - sequence_length
        return mx.array(positions, dtype=mx.int32), mx.array(deltas, dtype=mx.int32)

    grids = np.asarray(image_grid_thw).astype(np.int64, copy=False)
    if grids.ndim != 2 or grids.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [number_of_images, 3]")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")

    positions = np.ones((3, batch_size, sequence_length), dtype=np.int64)
    deltas: list[int] = []
    image_index = 0

    for batch_index in range(batch_size):
        active_ids = ids[batch_index, mask[batch_index]]
        vision_starts = np.flatnonzero(active_ids == vision_start_token_id)
        if np.any(vision_starts + 1 >= active_ids.size):
            raise ValueError("a vision-start token must be followed by an image token")
        image_count = int(np.sum(active_ids[vision_starts + 1] == image_token_id))

        chunks: list[np.ndarray] = []
        start = 0
        for _ in range(image_count):
            image_locations = np.flatnonzero(active_ids[start:] == image_token_id)
            if image_locations.size == 0:
                raise ValueError("could not find the image-token run announced by a vision-start token")
            image_start = start + int(image_locations[0])
            if image_index >= grids.shape[0]:
                raise ValueError("input_ids reference more images than image_grid_thw provides")

            grid_t, grid_h, grid_w = (int(value) for value in grids[image_index])
            image_index += 1
            if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
                raise ValueError("image grid dimensions must be positive")
            if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
                raise ValueError("image grid height and width must be divisible by spatial_merge_size")

            llm_grid_h = grid_h // spatial_merge_size
            llm_grid_w = grid_w // spatial_merge_size
            visual_length = grid_t * llm_grid_h * llm_grid_w
            image_end = image_start + visual_length
            if image_end > active_ids.size or np.any(active_ids[image_start:image_end] != image_token_id):
                raise ValueError("the image-token run length does not match image_grid_thw")

            text_length = image_start - start
            position_start = int(chunks[-1].max()) + 1 if chunks else 0
            text_positions = np.broadcast_to(
                np.arange(text_length, dtype=np.int64)[None, :],
                (3, text_length),
            )
            chunks.append(text_positions + position_start)

            temporal = np.repeat(np.arange(grid_t, dtype=np.int64), llm_grid_h * llm_grid_w)
            height = np.tile(
                np.repeat(np.arange(llm_grid_h, dtype=np.int64), llm_grid_w),
                grid_t,
            )
            width = np.tile(np.arange(llm_grid_w, dtype=np.int64), grid_t * llm_grid_h)
            visual_positions = np.stack([temporal, height, width])
            chunks.append(visual_positions + text_length + position_start)
            start = image_end

        if start < active_ids.size:
            position_start = int(chunks[-1].max()) + 1 if chunks else 0
            text_length = active_ids.size - start
            trailing_positions = np.broadcast_to(
                np.arange(text_length, dtype=np.int64)[None, :],
                (3, text_length),
            )
            chunks.append(trailing_positions + position_start)

        sample_positions = np.concatenate(chunks, axis=1) if chunks else np.empty((3, 0), dtype=np.int64)
        if sample_positions.shape[1] != active_ids.size:
            raise ValueError("constructed multimodal positions do not match the active token count")
        positions[:, batch_index, mask[batch_index]] = sample_positions
        max_position = int(sample_positions.max()) if sample_positions.size else 0
        deltas.append(max_position + 1 - sequence_length)

    if image_index != grids.shape[0]:
        raise ValueError("image_grid_thw provides more images than input_ids reference")
    return mx.array(positions, dtype=mx.int32), mx.array(deltas, dtype=mx.int32)[:, None]


class MageFlowQwen3VLLanguageModel(nn.Module):
    """Qwen3-VL language backbone returning normalized hidden states only."""

    DEEPSTACK_INJECTION_LAYERS = (0, 1, 2)

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9728,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
        attention_bias: bool = False,
        mrope_section: Sequence[int] = (24, 20, 20),
        attention_scaling: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            MageFlowQwen3VLDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                attention_bias=attention_bias,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = MageFlowQwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = MageFlowQwen3VLRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=attention_scaling,
            mrope_section=mrope_section,
        )

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        visual_positions: mx.array | None = None,
        deepstack_visual_embeds: Sequence[mx.array] | None = None,
        use_cache: bool = False,
        past_key_values: Sequence[MageFlowQwen3VLKVCache] | None = None,
        max_cache_length: int | None = None,
    ) -> mx.array | tuple[mx.array, list[MageFlowQwen3VLKVCache]]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one cache per Qwen3-VL layer")

        batch_size, sequence_length, _ = inputs_embeds.shape
        past_length = self._past_length(past_key_values)
        key_length = past_length + sequence_length
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, key_length), dtype=mx.int32)
        elif attention_mask.shape == (batch_size, sequence_length) and past_length:
            prefix_mask = mx.ones((batch_size, past_length), dtype=attention_mask.dtype)
            attention_mask = mx.concatenate([prefix_mask, attention_mask], axis=-1)
        elif attention_mask.shape != (batch_size, key_length):
            raise ValueError("attention_mask must cover all cached and current input tokens")

        if position_ids is None:
            sequential = mx.arange(past_length, key_length, dtype=mx.int32)
            position_ids = mx.broadcast_to(sequential[None, :], (batch_size, sequence_length))
        elif position_ids.ndim == 3 and position_ids.shape == (batch_size, sequence_length, 3):
            position_ids = position_ids.transpose(2, 0, 1)

        attention_mask_4d = self._causal_attention_mask(
            attention_mask,
            query_length=sequence_length,
            query_start=past_length,
            dtype=inputs_embeds.dtype,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds

        if deepstack_visual_embeds is not None:
            if visual_positions is None:
                raise ValueError("visual_positions are required with DeepStack visual embeddings")
            if len(deepstack_visual_embeds) > len(self.DEEPSTACK_INJECTION_LAYERS):
                raise ValueError("Qwen3-VL supports at most three DeepStack feature sets")

        present_key_values = [] if use_cache else None
        for layer_index, layer in enumerate(self.layers):
            if use_cache:
                layer_output = layer(
                    hidden_states,
                    attention_mask_4d,
                    position_embeddings,
                    use_cache=True,
                    past_key_value=(past_key_values[layer_index] if past_key_values is not None else None),
                    max_cache_length=max_cache_length,
                )
                hidden_states, present_key_value = layer_output
                present_key_values.append(present_key_value)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask_4d,
                    position_embeddings,
                )
            if deepstack_visual_embeds is not None and layer_index in self.DEEPSTACK_INJECTION_LAYERS:
                deepstack_index = self.DEEPSTACK_INJECTION_LAYERS.index(layer_index)
                if deepstack_index >= len(deepstack_visual_embeds):
                    continue
                hidden_states = self._scatter_add(
                    hidden_states,
                    visual_positions,
                    deepstack_visual_embeds[deepstack_index],
                )

        hidden_states = self.norm(hidden_states)
        if present_key_values is not None:
            return hidden_states, present_key_values
        return hidden_states

    @staticmethod
    def _causal_attention_mask(
        attention_mask: mx.array,
        *,
        query_length: int,
        query_start: int,
        dtype: mx.Dtype,
    ) -> mx.array:
        batch_size, key_length = attention_mask.shape
        query_indices = mx.arange(query_start, query_start + query_length, dtype=mx.int32)
        key_indices = mx.arange(key_length, dtype=mx.int32)
        is_future = key_indices[None, :] > query_indices[:, None]
        zero = mx.array(0.0, dtype=dtype)
        negative_infinity = mx.array(-float("inf"), dtype=dtype)
        causal = mx.where(is_future, negative_infinity, zero)
        causal = mx.broadcast_to(
            causal[None, None, :, :],
            (batch_size, 1, query_length, key_length),
        )
        padding = mx.where(attention_mask[:, None, None, :].astype(mx.bool_), zero, negative_infinity)
        return causal + padding

    @staticmethod
    def _past_length(past_key_values: Sequence[MageFlowQwen3VLKVCache] | None) -> int:
        if not past_key_values:
            return 0
        past_length = past_key_values[0][2]
        if any(cache[2] != past_length for cache in past_key_values):
            raise ValueError("all Qwen3-VL layer caches must have the same length")
        return past_length

    @staticmethod
    def _scatter_add(hidden_states: mx.array, flat_positions: mx.array, values: mx.array) -> mx.array:
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if values.shape != (flat_positions.shape[0], hidden_states.shape[-1]):
            raise ValueError("visual feature count and width must match the visual token positions")
        updated = flat_hidden_states.at[flat_positions].add(values.astype(hidden_states.dtype))
        return updated.reshape(hidden_states.shape)


class MageFlowTextEncoder(nn.Module):
    """Native MLX Qwen3-VL conditioner used by Mage-Flow T2I and editing."""

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9728,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
        attention_bias: bool = False,
        mrope_section: Sequence[int] = (24, 20, 20),
        attention_scaling: float = 1.0,
        image_token_id: int = 151655,
        vision_start_token_id: int = 151652,
        vision_config: Mapping[str, Any] | None = None,
        visual: nn.Module | None = None,
    ):
        super().__init__()
        self.language_model = MageFlowQwen3VLLanguageModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            head_dim=head_dim,
            attention_bias=attention_bias,
            mrope_section=mrope_section,
            attention_scaling=attention_scaling,
        )
        if visual is None:
            visual_kwargs: dict[str, Any] = {"out_hidden_size": hidden_size}
            visual_kwargs.update(vision_config or {})
            visual = MageFlowQwen3VLVisionModel(**visual_kwargs)
        self.visual = visual
        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        hidden_states, _, _ = self._forward(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
        )
        return hidden_states

    def forward_with_cache(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        position_ids: mx.array | None = None,
        rope_deltas: mx.array | None = None,
        past_key_values: Sequence[MageFlowQwen3VLKVCache] | None = None,
        max_cache_length: int | None = None,
    ) -> tuple[mx.array, list[MageFlowQwen3VLKVCache], mx.array]:
        hidden_states, present_key_values, rope_deltas = self._forward(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            use_cache=True,
            past_key_values=past_key_values,
            max_cache_length=max_cache_length,
        )
        if present_key_values is None:
            raise RuntimeError("Qwen3-VL cache generation did not return layer caches")
        return hidden_states, present_key_values, rope_deltas

    def generate_greedy(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        max_new_tokens: int,
        eos_token_id: int,
    ) -> mx.array:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("policy generation requires one tokenized prompt")
        if input_ids.shape[1] == 0:
            raise ValueError("policy generation requires a non-empty prompt")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return mx.zeros((1, 0), dtype=mx.int32)

        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        max_cache_length = input_ids.shape[1] + max_new_tokens
        hidden_states, past_key_values, rope_deltas = self.forward_with_cache(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_cache_length=max_cache_length,
        )

        generated_tokens = []
        for token_index in range(max_new_tokens):
            next_token = self._greedy_next_token(hidden_states)
            generated_tokens.append(next_token[:, None])
            mx.eval(next_token, *self._cache_arrays(past_key_values))
            if int(next_token[0].item()) == eos_token_id:
                break
            if token_index + 1 == max_new_tokens:
                break

            attention_mask = mx.concatenate(
                [attention_mask, mx.ones((1, 1), dtype=attention_mask.dtype)],
                axis=-1,
            )
            cache_length = past_key_values[0][2]
            next_position = cache_length + rope_deltas[:, 0]
            position_ids = mx.broadcast_to(
                next_position[None, :, None],
                (3, input_ids.shape[0], 1),
            )
            hidden_states, past_key_values, rope_deltas = self.forward_with_cache(
                next_token[:, None],
                attention_mask=attention_mask,
                position_ids=position_ids,
                rope_deltas=rope_deltas,
                past_key_values=past_key_values,
                max_cache_length=max_cache_length,
            )

        return mx.concatenate(generated_tokens, axis=1)

    def screen_text(
        self,
        prompt: str,
        tokenizer: Any,
        max_new_tokens: int = 160,
    ) -> FilterVerdict:
        from mflux.models.mage_flow.model.mage_flow_text_encoder.policy import MageFlowContentPolicy

        return MageFlowContentPolicy.screen_text(
            text_encoder=self,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    def screen_edit(
        self,
        prompt: str,
        ref_images: Any,
        tokenizer: Any,
        max_new_tokens: int = 192,
    ) -> FilterVerdict:
        from mflux.models.mage_flow.model.mage_flow_text_encoder.policy import MageFlowContentPolicy

        return MageFlowContentPolicy.screen_edit(
            text_encoder=self,
            tokenizer=tokenizer,
            prompt=prompt,
            ref_images=ref_images,
            max_new_tokens=max_new_tokens,
        )

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        return build_mrope_position_ids(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            image_token_id=self.image_token_id,
            vision_start_token_id=self.vision_start_token_id,
            spatial_merge_size=self.visual.spatial_merge_size,
        )

    def _forward(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        position_ids: mx.array | None = None,
        rope_deltas: mx.array | None = None,
        use_cache: bool = False,
        past_key_values: Sequence[MageFlowQwen3VLKVCache] | None = None,
        max_cache_length: int | None = None,
    ) -> tuple[mx.array, list[MageFlowQwen3VLKVCache] | None, mx.array]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be provided together")

        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        if position_ids is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        elif rope_deltas is None:
            rope_deltas = mx.zeros((input_ids.shape[0], 1), dtype=mx.int32)

        inputs_embeds = self.language_model.embed_tokens(input_ids)
        visual_positions = None
        deepstack_visual_embeds = None
        if pixel_values is not None:
            image_embeds, deepstack_visual_embeds = self.visual(
                pixel_values,
                image_grid_thw,
                return_deepstack=True,
            )
            if deepstack_visual_embeds is None:
                raise RuntimeError("the Qwen3-VL vision model did not return DeepStack features")

            ids = np.asarray(input_ids)
            mask = np.asarray(attention_mask).astype(bool, copy=False)
            flat_visual_positions = np.flatnonzero(((ids == self.image_token_id) & mask).reshape(-1))
            if flat_visual_positions.size != image_embeds.shape[0]:
                raise ValueError("the number of image placeholder tokens does not match the merged vision features")
            visual_positions = mx.array(flat_visual_positions, dtype=mx.int32)
            inputs_embeds = self._replace_visual_embeddings(
                inputs_embeds,
                visual_positions,
                image_embeds,
            )

        language_output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_positions=visual_positions,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            max_cache_length=max_cache_length,
        )

        if use_cache:
            hidden_states, present_key_values = language_output
        else:
            hidden_states = language_output
            present_key_values = None
        return hidden_states, present_key_values, rope_deltas

    def _greedy_next_token(self, hidden_states: mx.array) -> mx.array:
        logits = self.language_model.embed_tokens.as_linear(hidden_states[:, -1])
        return mx.argmax(logits, axis=-1).astype(mx.int32)

    @staticmethod
    def _cache_arrays(past_key_values: Sequence[MageFlowQwen3VLKVCache]) -> list[mx.array]:
        return [array for key_states, value_states, _ in past_key_values for array in (key_states, value_states)]

    @staticmethod
    def _replace_visual_embeddings(
        inputs_embeds: mx.array,
        flat_positions: mx.array,
        image_embeds: mx.array,
    ) -> mx.array:
        flat_inputs = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        if image_embeds.shape != (flat_positions.shape[0], inputs_embeds.shape[-1]):
            raise ValueError("vision features must match the image-token count and language hidden size")
        replacements = image_embeds.astype(inputs_embeds.dtype)
        updated = flat_inputs.at[flat_positions].add(replacements - flat_inputs[flat_positions])
        return updated.reshape(inputs_embeds.shape)
