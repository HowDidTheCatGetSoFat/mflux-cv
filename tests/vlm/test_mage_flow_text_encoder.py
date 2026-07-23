import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx import nn
from mlx.utils import tree_flatten
from PIL import Image

from mflux.models.mage_flow.model.mage_flow_text_encoder import (
    MageFlowPromptProcessor,
    MageFlowQwen3VLProcessor,
    MageFlowQwen3VLRotaryEmbedding,
    MageFlowQwen3VLVisionModel,
    MageFlowTextEncoder,
    build_mrope_position_ids,
)
from mflux.models.mage_flow.model.mage_flow_text_encoder.layers import MageFlowQwen3VLRMSNorm
from mflux.models.mage_flow.variants.conditioning import MageFlowConditioning


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.last_max_length = None
        self.last_truncation = False

    def convert_tokens_to_ids(self, token: str) -> int:
        return {
            "<|image_pad|>": 3,
            "<|video_pad|>": 4,
        }[token]

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        return_tensors: str,
        max_length: int | None = None,
        truncation: bool = False,
    ) -> dict[str, np.ndarray]:
        del padding, return_tensors
        self.last_max_length = max_length
        self.last_truncation = truncation
        sequences = []
        for text in texts:
            image_token_count = text.count("<|image_pad|>")
            sequences.append([1, *([3] * image_token_count), 2])
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = np.array(
            [sequence + [self.pad_token_id] * (max_length - len(sequence)) for sequence in sequences],
            dtype=np.int32,
        )
        attention_mask = input_ids != self.pad_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask.astype(np.int32),
        }


class _FakeVision(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.spatial_merge_size = 2
        self.deepstack_visual_indexes = [5, 11, 17]

    def __call__(
        self,
        pixel_values: mx.array,
        image_grid_thw: mx.array,
        return_deepstack: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        del pixel_values, image_grid_thw
        primary = mx.full((4, self.hidden_size), 10.0)
        deepstack = [mx.full((4, self.hidden_size), value) for value in (1.0, 2.0, 3.0)]
        return primary, deepstack if return_deepstack else None


class _IdentityDecoderLayer(nn.Module):
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
    ) -> mx.array:
        del attention_mask, position_embeddings
        return hidden_states


class _IdentityNorm(nn.Module):
    def __call__(self, hidden_states: mx.array) -> mx.array:
        return hidden_states


class _ControlledPolicyGenerator:
    generate_greedy = MageFlowTextEncoder.generate_greedy
    _greedy_next_token = MageFlowTextEncoder._greedy_next_token
    _cache_arrays = staticmethod(MageFlowTextEncoder._cache_arrays)

    def __init__(self):
        self.language_model = type("_LanguageModel", (), {})()
        self.language_model.embed_tokens = nn.Embedding(8, 4)
        self.language_model.embed_tokens.weight = mx.zeros((8, 4))
        self.language_model.embed_tokens.weight[5, 0] = 2.0
        self.language_model.embed_tokens.weight[2, 1] = 2.0
        self.forward_calls = []

    def forward_with_cache(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array,
        max_cache_length: int,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        position_ids: mx.array | None = None,
        rope_deltas: mx.array | None = None,
        past_key_values=None,
    ):
        del pixel_values, image_grid_thw
        cache_length = (past_key_values[0][2] if past_key_values else 0) + input_ids.shape[1]
        cache = (
            mx.zeros((1, 1, max_cache_length, 4)),
            mx.zeros((1, 1, max_cache_length, 4)),
            cache_length,
        )
        output_axis = 0 if not past_key_values else 1
        hidden_states = mx.zeros((*input_ids.shape, 4))
        hidden_states[:, -1, output_axis] = 1.0
        rope_deltas = rope_deltas if rope_deltas is not None else mx.zeros((1, 1), dtype=mx.int32)
        self.forward_calls.append(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        return hidden_states, [cache], rope_deltas


def _tiny_encoder(*, num_hidden_layers: int = 1, visual: nn.Module | None = None) -> MageFlowTextEncoder:
    return MageFlowTextEncoder(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=16,
        max_position_embeddings=128,
        rope_theta=10_000.0,
        head_dim=8,
        mrope_section=(2, 1, 1),
        image_token_id=3,
        vision_start_token_id=2,
        visual=visual or _FakeVision(hidden_size=8),
    )


def test_mage_flow_prompt_processor_preserves_training_templates_and_drop_counts() -> None:
    text_prompt = MageFlowPromptProcessor.format_text_to_image("a red fox")
    assert text_prompt == (
        "<|im_start|>system\n"
        "Describe the image by detailing the color, shape, size, texture, quantity, "
        "text, spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\na red fox<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    edit_prompt = MageFlowPromptProcessor.format_edit("make both blue", num_images=2)
    assert (
        "Image 1: <|vision_start|><|image_pad|><|vision_end|>"
        "Image 2: <|vision_start|><|image_pad|><|vision_end|>make both blue"
    ) in edit_prompt
    assert MageFlowPromptProcessor.TEXT_TO_IMAGE_DROP_TOKENS == 34
    assert MageFlowPromptProcessor.EDIT_DROP_TOKENS == 64


def test_mage_flow_prompt_processor_drops_per_sample_then_right_pads() -> None:
    hidden_states = mx.broadcast_to(
        mx.arange(66, dtype=mx.float32)[None, :, None],
        (2, 66, 2),
    )
    attention_mask = mx.array(
        [
            [1] * 66,
            [1] * 65 + [0],
        ],
        dtype=mx.int32,
    )

    prompt_embeds, prompt_mask = MageFlowPromptProcessor.process_edit_hidden_states(
        hidden_states,
        attention_mask,
    )
    np.testing.assert_array_equal(np.asarray(prompt_mask), np.array([[1, 1], [1, 0]]))
    np.testing.assert_array_equal(np.asarray(prompt_embeds[0, :, 0]), np.array([64.0, 65.0]))
    np.testing.assert_array_equal(np.asarray(prompt_embeds[1, :, 0]), np.array([64.0, 0.0]))

    long_hidden_states = mx.zeros((1, 34 + 2050, 2))
    capped, capped_mask = MageFlowPromptProcessor.process_text_to_image_hidden_states(
        long_hidden_states,
        mx.ones(long_hidden_states.shape[:2], dtype=mx.int32),
    )
    assert capped.shape == (1, 2048, 2)
    assert capped_mask.shape == (1, 2048)


def test_mage_flow_edit_conditioning_passes_sequential_positions_for_padded_cfg_batch() -> None:
    sequence_length = MageFlowPromptProcessor.EDIT_DROP_TOKENS + 3

    class _Processor:
        def __call__(self, **kwargs):
            assert len(kwargs["text"]) == 2
            assert len(kwargs["images"]) == 2
            return {
                "input_ids": mx.zeros((2, sequence_length), dtype=mx.int32),
                "attention_mask": mx.array(
                    [
                        [1] * sequence_length,
                        [1] * (sequence_length - 1) + [0],
                    ],
                    dtype=mx.int32,
                ),
                "pixel_values": mx.zeros((2, 1)),
                "image_grid_thw": mx.ones((2, 3), dtype=mx.int32),
            }

    class _Tokenizer:
        processor = _Processor()

    class _TextEncoder:
        def __init__(self):
            self.position_ids = None

        def __call__(self, **kwargs):
            self.position_ids = kwargs["position_ids"]
            return mx.zeros((2, sequence_length, 8))

    text_encoder = _TextEncoder()
    image = Image.new("RGB", (1, 1))
    prompt_embeds, prompt_mask = MageFlowConditioning.encode_edit(
        prompts=["make it blue", ""],
        images_per_prompt=[[image], [image]],
        tokenizer=_Tokenizer(),
        text_encoder=text_encoder,
    )

    expected = np.broadcast_to(np.arange(sequence_length, dtype=np.int32), (2, sequence_length))
    np.testing.assert_array_equal(np.asarray(text_encoder.position_ids), expected)
    assert prompt_embeds.shape == (2, 3, 8)
    np.testing.assert_array_equal(np.asarray(prompt_mask), np.array([[1, 1, 1], [1, 1, 0]]))


def test_mage_flow_qwen3_vl_processor_uses_checkpoint_image_config_and_expands_placeholder() -> None:
    tokenizer = _FakeTokenizer()
    processor = MageFlowQwen3VLProcessor(tokenizer)
    image = Image.new("RGB", (1024, 512), color=(255, 255, 255))

    inputs = processor(
        text=[f"Image 1: {MageFlowPromptProcessor.IMAGE_PLACEHOLDER}edit"],
        images=[image],
        max_length=2112,
        truncation=True,
    )

    assert processor.image_processor.min_pixels == 65_536
    assert processor.image_processor.max_pixels == 16_777_216
    assert processor.image_processor.patch_size == 16
    assert processor.image_processor.temporal_patch_size == 2
    assert processor.image_processor.merge_size == 2
    np.testing.assert_array_equal(inputs["image_grid_thw"], np.array([[1, 12, 24]]))
    assert int(mx.sum(inputs["input_ids"] == processor.image_token_id).item()) == 72
    assert inputs["pixel_values"].shape == (288, 1536)
    np.testing.assert_allclose(inputs["pixel_values"], np.ones((288, 1536)), rtol=0, atol=0)
    assert tokenizer.last_max_length == 2112
    assert tokenizer.last_truncation is True

    with pytest.raises(ValueError, match="image placeholders"):
        processor(text=["missing placeholder"], images=[image])


def test_qwen3_vl_text_positions_respect_padding() -> None:
    input_ids = mx.array([[4, 5, 6, 0], [7, 8, 0, 0]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=mx.int32)

    position_ids, deltas = build_mrope_position_ids(
        input_ids,
        attention_mask=attention_mask,
    )

    expected = np.array(
        [
            [0, 1, 2, 1],
            [0, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    for axis in range(3):
        np.testing.assert_array_equal(np.asarray(position_ids[axis]), expected)
    np.testing.assert_array_equal(np.asarray(deltas), np.array([[-1], [-2]], dtype=np.int32))


def test_qwen3_vl_multimodal_positions_use_merged_image_grid() -> None:
    # Two text tokens, a 2x2 merged image grid, then two trailing text tokens.
    input_ids = mx.array([[7, 2, 3, 3, 3, 3, 8, 9]], dtype=mx.int32)
    grid_thw = mx.array([[1, 4, 4]], dtype=mx.int32)

    position_ids, deltas = build_mrope_position_ids(
        input_ids,
        image_grid_thw=grid_thw,
        image_token_id=3,
        vision_start_token_id=2,
        spatial_merge_size=2,
    )

    expected = np.array(
        [
            [[0, 1, 2, 2, 2, 2, 4, 5]],
            [[0, 1, 2, 2, 3, 3, 4, 5]],
            [[0, 1, 2, 3, 2, 3, 4, 5]],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(np.asarray(position_ids), expected)
    np.testing.assert_array_equal(np.asarray(deltas), np.array([[-2]], dtype=np.int32))


def test_qwen3_vl_rope_interleaves_temporal_height_and_width_frequencies() -> None:
    rope = MageFlowQwen3VLRotaryEmbedding(
        dim=8,
        base=1.0,
        mrope_section=(2, 1, 1),
    )
    hidden_states = mx.zeros((1, 1, 8), dtype=mx.float32)
    position_ids = mx.array([[[1]], [[2]], [[3]]], dtype=mx.int32)

    cos, sin = rope(hidden_states, position_ids)
    expected_frequencies = np.array([1.0, 2.0, 3.0, 1.0] * 2)
    np.testing.assert_allclose(np.asarray(cos[0, 0]), np.cos(expected_frequencies), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sin[0, 0]), np.sin(expected_frequencies), rtol=1e-6, atol=1e-6)


def test_qwen3_vl_fused_rms_norm_matches_hf_bfloat16_ordering() -> None:
    rng = np.random.default_rng(7)
    values = rng.standard_normal((8, 16), dtype=np.float32)
    weights = rng.standard_normal((16,), dtype=np.float32)
    norm = MageFlowQwen3VLRMSNorm(16)
    norm.weight = mx.array(weights).astype(mx.bfloat16)
    mlx_values = mx.array(values).astype(mx.bfloat16)

    output = norm(mlx_values)
    torch_values = torch.from_numpy(values).to(torch.bfloat16)
    torch_weights = torch.from_numpy(weights).to(torch.bfloat16)
    normalized = torch_values.float() * torch.rsqrt(torch_values.float().square().mean(-1, keepdim=True) + 1e-6)
    reference = torch_weights * normalized.to(torch.bfloat16)

    np.testing.assert_array_equal(
        np.asarray(output.astype(mx.float32)),
        reference.float().numpy(),
    )


def test_qwen3_vl_real_vision_tower_keeps_bfloat16_and_captures_deepstack() -> None:
    vision = MageFlowQwen3VLVisionModel(
        patch_size=2,
        temporal_patch_size=2,
        hidden_size=8,
        num_heads=2,
        intermediate_size=16,
        depth=3,
        spatial_merge_size=2,
        num_position_embeddings=16,
        out_hidden_size=8,
        deepstack_visual_indexes=(0, 1, 2),
    )
    vision.set_dtype(mx.bfloat16)
    primary, deepstack = vision(
        mx.ones((16, 24), dtype=mx.bfloat16),
        mx.array([[1, 4, 4]], dtype=mx.int32),
        return_deepstack=True,
    )
    mx.eval(primary, *deepstack)

    assert primary.shape == (4, 8)
    assert primary.dtype == mx.bfloat16
    assert len(deepstack) == 3
    assert all(features.shape == (4, 8) for features in deepstack)
    assert all(features.dtype == mx.bfloat16 for features in deepstack)


def test_mage_flow_text_encoder_preserves_qwen_checkpoint_paths_without_lm_head() -> None:
    model = MageFlowTextEncoder(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=16,
        head_dim=8,
        mrope_section=(2, 1, 1),
        vision_config={
            "patch_size": 2,
            "temporal_patch_size": 2,
            "hidden_size": 8,
            "num_heads": 2,
            "intermediate_size": 16,
            "depth": 18,
            "num_position_embeddings": 16,
        },
    )
    parameter_paths = {name for name, _ in tree_flatten(model.parameters())}

    assert "language_model.embed_tokens.weight" in parameter_paths
    assert "language_model.layers.0.self_attn.q_proj.weight" in parameter_paths
    assert "language_model.norm.weight" in parameter_paths
    assert "visual.patch_embed.proj.weight" in parameter_paths
    assert "visual.deepstack_merger_list.2.linear_fc2.weight" in parameter_paths
    assert model.visual.deepstack_visual_indexes == [5, 11, 17]
    assert model.language_model.DEEPSTACK_INJECTION_LAYERS == (0, 1, 2)
    assert "visual.rotary_pos_emb.inv_freq" not in parameter_paths
    assert not any("lm_head" in name for name in parameter_paths)
    assert not any(name.startswith("model.") for name in parameter_paths)


def test_mage_flow_edit_replaces_image_tokens_and_injects_all_deepstack_features() -> None:
    model = _tiny_encoder(num_hidden_layers=3)
    model.language_model.embed_tokens.weight = mx.zeros_like(model.language_model.embed_tokens.weight)
    model.language_model.layers = [_IdentityDecoderLayer() for _ in range(3)]
    model.language_model.norm = _IdentityNorm()

    input_ids = mx.array([[7, 2, 3, 3, 3, 3, 8]], dtype=mx.int32)
    hidden_states = model(
        input_ids,
        pixel_values=mx.zeros((1, 1)),
        image_grid_thw=mx.array([[1, 4, 4]], dtype=mx.int32),
    )
    mx.eval(hidden_states)

    expected = np.zeros((1, 7, 8), dtype=np.float32)
    expected[:, 2:6] = 16.0  # primary 10 + DeepStack features 1 + 2 + 3
    np.testing.assert_array_equal(np.asarray(hidden_states), expected)


def test_mage_flow_text_only_forward_returns_final_normalized_hidden_states() -> None:
    mx.random.seed(7)
    model = _tiny_encoder()
    hidden_states = model(
        mx.array([[4, 5, 6, 7], [8, 9, 10, 0]], dtype=mx.int32),
        attention_mask=mx.array([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=mx.int32),
    )
    mx.eval(hidden_states)

    assert hidden_states.shape == (2, 4, 8)
    assert bool(mx.all(mx.isfinite(hidden_states)).item())
    mean_square = mx.mean(hidden_states.astype(mx.float32) ** 2, axis=-1)
    np.testing.assert_allclose(np.asarray(mean_square), np.ones((2, 4)), rtol=2e-4, atol=2e-4)


def test_mage_flow_text_cache_matches_full_forward_and_is_preallocated_bfloat16() -> None:
    mx.random.seed(17)
    model = _tiny_encoder()
    model.set_dtype(mx.bfloat16)
    input_ids = mx.array([[4, 5, 6, 7]], dtype=mx.int32)

    full_hidden_states = model(input_ids)
    _, caches, rope_deltas = model.forward_with_cache(
        input_ids[:, :3],
        max_cache_length=4,
    )
    cached_hidden_states, caches, _ = model.forward_with_cache(
        input_ids[:, 3:],
        attention_mask=mx.ones((1, 4), dtype=mx.int32),
        position_ids=mx.full((3, 1, 1), 3, dtype=mx.int32),
        rope_deltas=rope_deltas,
        past_key_values=caches,
        max_cache_length=4,
    )
    mx.eval(full_hidden_states, cached_hidden_states, *model._cache_arrays(caches))

    np.testing.assert_allclose(
        np.asarray(cached_hidden_states[:, -1].astype(mx.float32)),
        np.asarray(full_hidden_states[:, -1].astype(mx.float32)),
        rtol=0,
        atol=2e-2,
    )
    assert len(caches) == 1
    key_states, value_states, cache_length = caches[0]
    assert key_states.shape == (1, 1, 4, 8)
    assert value_states.shape == (1, 1, 4, 8)
    assert key_states.dtype == mx.bfloat16
    assert value_states.dtype == mx.bfloat16
    assert cache_length == 4


def test_mage_flow_multimodal_cache_continues_from_mrope_delta() -> None:
    mx.random.seed(23)
    model = _tiny_encoder()
    full_ids = mx.array([[7, 2, 3, 3, 3, 3, 8, 9]], dtype=mx.int32)
    prefix_ids = full_ids[:, :-1]
    image_grid_thw = mx.array([[1, 4, 4]], dtype=mx.int32)

    full_hidden_states = model(
        full_ids,
        pixel_values=mx.zeros((1, 1)),
        image_grid_thw=image_grid_thw,
    )
    _, caches, rope_deltas = model.forward_with_cache(
        prefix_ids,
        pixel_values=mx.zeros((1, 1)),
        image_grid_thw=image_grid_thw,
        max_cache_length=full_ids.shape[1],
    )
    next_position = prefix_ids.shape[1] + rope_deltas[:, 0]
    position_ids = mx.broadcast_to(next_position[None, :, None], (3, 1, 1))
    cached_hidden_states, caches, _ = model.forward_with_cache(
        full_ids[:, -1:],
        attention_mask=mx.ones(full_ids.shape, dtype=mx.int32),
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        past_key_values=caches,
        max_cache_length=full_ids.shape[1],
    )
    mx.eval(full_hidden_states, cached_hidden_states, *model._cache_arrays(caches))

    np.testing.assert_array_equal(np.asarray(rope_deltas), np.array([[-2]], dtype=np.int32))
    np.testing.assert_allclose(
        np.asarray(cached_hidden_states[:, -1]),
        np.asarray(full_hidden_states[:, -1]),
        rtol=3e-3,
        atol=2e-3,
    )


def test_mage_flow_greedy_generation_uses_tied_embeddings_and_stops_on_eos() -> None:
    model = _ControlledPolicyGenerator()

    generated_ids = model.generate_greedy(
        mx.array([[6, 7]], dtype=mx.int32),
        max_new_tokens=4,
        eos_token_id=2,
    )
    mx.eval(generated_ids)

    np.testing.assert_array_equal(np.asarray(generated_ids), np.array([[5, 2]], dtype=np.int32))
    assert len(model.forward_calls) == 2
    np.testing.assert_array_equal(
        np.asarray(model.forward_calls[1]["position_ids"]),
        np.full((3, 1, 1), 2, dtype=np.int32),
    )
