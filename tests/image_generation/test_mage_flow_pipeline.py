from types import SimpleNamespace
from weakref import ref

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from PIL import Image

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.mage_flow.mage_flow_initializer import MageFlowInitializer
from mflux.models.mage_flow.variants.edit.mage_flow_edit import MageFlowEdit
from mflux.models.mage_flow.variants.pipeline_helpers import (
    default_guidance,
    default_inference_steps,
    make_velocity_predictor,
    normalize_image_dimension,
    resolve_generation_parameters,
)
from mflux.models.mage_flow.variants.txt2img.mage_flow import MageFlow


class _RawTokenizer:
    def __call__(self, prompts, **kwargs):
        sequence_length = 35
        ids = np.zeros((len(prompts), sequence_length), dtype=np.int32)
        return {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
        }


class _EditProcessor:
    def __call__(self, *, text, images, **kwargs):
        sequence_length = 65
        input_ids = mx.zeros((len(text), sequence_length), dtype=mx.int32)
        return {
            "input_ids": input_ids,
            "attention_mask": mx.ones_like(input_ids),
            "pixel_values": mx.zeros((len(images), 1536), dtype=mx.float32),
            "image_grid_thw": mx.array([[1, 2, 2]] * len(images), dtype=mx.int32),
        }


class _TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = False

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        position_ids=None,
    ):
        return mx.zeros((*input_ids.shape, 8), dtype=mx.bfloat16)

    def screen_text(self, prompt, tokenizer):
        return _Verdict(self.block)

    def screen_edit(self, prompt, references, tokenizer):
        return _Verdict(self.block)


class _Verdict:
    def __init__(self, violates):
        self.violates = violates
        self.categories = ["policy"] if violates else []
        self.reason = "blocked for test" if violates else "allowed"
        self.raw = ""

    def banner(self):
        return "blocked for test" if self.violates else ""


class _PositionEmbedder:
    def __call__(self, image_shapes):
        sequence_length = sum(frames * height * width for frames, height, width in image_shapes)
        return (
            mx.ones((sequence_length, 64), dtype=mx.float32),
            mx.zeros((sequence_length, 64), dtype=mx.float32),
        )


class _ZeroTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = _PositionEmbedder()

    def __call__(
        self,
        img,
        txt,
        timesteps,
        img_shapes,
        text_attention_mask=None,
        image_rotary_emb=None,
    ):
        return mx.zeros((txt.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)


class _ConditionedTransformer(_ZeroTransformer):
    def __call__(
        self,
        img,
        txt,
        timesteps,
        img_shapes,
        text_attention_mask=None,
        image_rotary_emb=None,
    ):
        offsets = txt[:, :1, :1]
        return mx.broadcast_to(img, (txt.shape[0], *img.shape[1:])) + offsets


class _VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_posterior = True
        self.encoder = SimpleNamespace(patch_cond_embed=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.bfloat16)))

    def encode(self, image, *, key=None):
        return mx.zeros(
            (image.shape[0], 128, image.shape[2] // 16, image.shape[3] // 16),
            dtype=mx.bfloat16,
        )

    def decode(self, latents):
        return mx.zeros(
            (latents.shape[0], 3, latents.shape[2] * 16, latents.shape[3] * 16),
            dtype=mx.bfloat16,
        )


def _bare_pipeline(cls, model_config, tokenizer):
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.model_config = model_config
    model.callbacks = CallbackRegistry()
    model.prompt_cache = {}
    model.policy_cache = {}
    model.tokenizers = {"mage": tokenizer}
    model.text_encoder = _TextEncoder()
    model.transformer = _ZeroTransformer()
    model.vae = _VAE()
    model.bits = None
    model.lora_paths = None
    model.lora_scales = None
    model.tiling_config = None
    return model


def test_mage_flow_defaults_match_all_six_released_checkpoints():
    cases = [
        (ModelConfig.mage_flow_base(), 30, 5.0),
        (ModelConfig.mage_flow(), 20, 5.0),
        (ModelConfig.mage_flow_turbo(), 4, 1.0),
        (ModelConfig.mage_flow_edit_base(), 30, 5.0),
        (ModelConfig.mage_flow_edit(), 30, 5.0),
        (ModelConfig.mage_flow_edit_turbo(), 4, 1.0),
    ]
    for config, steps, guidance in cases:
        assert default_inference_steps(config) == steps
        assert default_guidance(config) == guidance


def test_mage_flow_turbo_python_api_rejects_cfg():
    with pytest.raises(ValueError, match="require guidance=1.0"):
        resolve_generation_parameters(
            model_config=ModelConfig.mage_flow_turbo(),
            num_inference_steps=4,
            guidance=5.0,
        )


@pytest.mark.parametrize("size", [1, 15, 0, -1, -32])
def test_mage_flow_dimensions_match_official_minimum_floor(size):
    assert normalize_image_dimension(size) == 16


def test_mage_flow_initializer_rejects_hf_shape_drift_before_applying_weights():
    model = SimpleNamespace(
        vae=nn.Linear(2, 2),
        transformer=nn.Linear(2, 2),
        text_encoder=nn.Linear(2, 2),
    )
    components = {
        name: {
            "weight": mx.zeros((2, 2)),
            "bias": mx.zeros((2,)),
        }
        for name in ("vae", "transformer", "text_encoder")
    }
    weights = LoadedWeights(components=components, meta_data=MetaData())
    MageFlowInitializer._validate_hf_model_coverage(model, weights)

    components["transformer"]["weight"] = mx.zeros((3, 2))
    with pytest.raises(ValueError, match="shape_mismatches=\\['weight'\\]"):
        MageFlowInitializer._validate_hf_model_coverage(model, weights)


def test_mage_flow_cfg_predictor_batches_unconditional_then_conditional_and_slices_target():
    transformer = _ConditionedTransformer()
    text = mx.array([[[0.0]], [[2.0]]])
    mask = mx.ones((2, 1), dtype=mx.int32)
    predictor = make_velocity_predictor(
        transformer=transformer,
        text_embeddings=text,
        text_attention_mask=mask,
        image_shapes=[(1, 1, 3)],
        guidance=5.0,
        target_length=2,
        compile_model=False,
    )

    velocity = predictor(mx.ones((1, 3, 1)), mx.array(1.0))
    mx.eval(velocity)

    np.testing.assert_array_equal(np.asarray(velocity), np.full((1, 2, 1), 11.0))


def test_mage_flow_text_to_image_pipeline_runs_one_native_mlx_step(monkeypatch):
    import mflux.models.mage_flow.variants.txt2img.mage_flow as pipeline_module

    original_predictor = make_velocity_predictor
    monkeypatch.setattr(
        pipeline_module,
        "make_velocity_predictor",
        lambda **kwargs: original_predictor(**kwargs, compile_model=False),
    )
    tokenizer = SimpleNamespace(tokenizer=_RawTokenizer())
    model = _bare_pipeline(MageFlow, ModelConfig.mage_flow_turbo(), tokenizer)

    generated = model.generate_image(
        seed=7,
        prompt="a tiny test",
        num_inference_steps=1,
        height=16,
        width=16,
        guidance=1.0,
    )

    assert generated.image.size == (16, 16)
    assert generated.model_config is ModelConfig.mage_flow_turbo()
    assert generated.steps == 1


@pytest.mark.parametrize("size", [1, 15, -1])
def test_mage_flow_text_to_image_clamps_small_dimensions_to_16(monkeypatch, size):
    import mflux.models.mage_flow.variants.txt2img.mage_flow as pipeline_module

    original_predictor = make_velocity_predictor
    monkeypatch.setattr(
        pipeline_module,
        "make_velocity_predictor",
        lambda **kwargs: original_predictor(**kwargs, compile_model=False),
    )
    tokenizer = SimpleNamespace(tokenizer=_RawTokenizer())
    model = _bare_pipeline(MageFlow, ModelConfig.mage_flow_turbo(), tokenizer)

    generated = model.generate_image(
        seed=7,
        prompt="a tiny test",
        num_inference_steps=1,
        height=size,
        width=size,
        guidance=1.0,
    )

    assert generated.image.size == (16, 16)


def test_mage_flow_text_to_image_resolves_random_seed_sentinel_once(monkeypatch):
    import mflux.models.mage_flow.variants.txt2img.mage_flow as pipeline_module

    original_predictor = make_velocity_predictor
    monkeypatch.setattr(
        pipeline_module,
        "make_velocity_predictor",
        lambda **kwargs: original_predictor(**kwargs, compile_model=False),
    )
    monkeypatch.setattr(
        "mflux.models.mage_flow.variants.pipeline_helpers.random.randint",
        lambda low, high: 123456789,
    )
    tokenizer = SimpleNamespace(tokenizer=_RawTokenizer())
    model = _bare_pipeline(MageFlow, ModelConfig.mage_flow_turbo(), tokenizer)

    generated = model.generate_image(
        seed=-1,
        prompt="a tiny test",
        num_inference_steps=1,
        height=16,
        width=16,
        guidance=1.0,
    )

    assert generated.seed == 123456789


def test_mage_flow_releases_predictor_before_low_ram_after_loop(monkeypatch):
    import mflux.models.mage_flow.variants.txt2img.mage_flow as pipeline_module

    original_predictor = make_velocity_predictor
    monkeypatch.setattr(
        pipeline_module,
        "make_velocity_predictor",
        lambda **kwargs: original_predictor(**kwargs, compile_model=False),
    )
    tokenizer = SimpleNamespace(tokenizer=_RawTokenizer())
    model = _bare_pipeline(MageFlow, ModelConfig.mage_flow_turbo(), tokenizer)
    transformer_ref = ref(model.transformer)

    class TrackingMemorySaver(MemorySaver):
        transformer_was_released = False

        def call_after_loop(self, *args, **kwargs):
            super().call_after_loop(*args, **kwargs)
            self.transformer_was_released = transformer_ref() is None

    memory_saver = TrackingMemorySaver(model=model, keep_transformer=False, cache_limit_bytes=None)
    model.callbacks.register(memory_saver)
    model.generate_image(
        seed=7,
        prompt="a tiny test",
        num_inference_steps=1,
        height=16,
        width=16,
        guidance=1.0,
    )

    assert memory_saver.transformer_was_released is True


def test_mage_flow_content_policy_returns_white_refusal_before_denoising():
    tokenizer = SimpleNamespace(tokenizer=_RawTokenizer())
    model = _bare_pipeline(MageFlow, ModelConfig.mage_flow_turbo(), tokenizer)
    model.text_encoder.block = True

    generated = model.generate_image(
        seed=7,
        prompt="blocked",
        num_inference_steps=1,
        height=16,
        width=16,
        guidance=1.0,
    )

    assert generated.image.getpixel((0, 0)) == (255, 255, 255)
    assert generated.generation_time == 0.0


def test_mage_flow_edit_pipeline_keeps_reference_clean_and_steps_target_only(monkeypatch):
    import mflux.models.mage_flow.variants.edit.mage_flow_edit as pipeline_module

    original_predictor = make_velocity_predictor
    monkeypatch.setattr(
        pipeline_module,
        "make_velocity_predictor",
        lambda **kwargs: original_predictor(**kwargs, compile_model=False),
    )
    tokenizer = SimpleNamespace(processor=_EditProcessor())
    model = _bare_pipeline(MageFlowEdit, ModelConfig.mage_flow_edit_turbo(), tokenizer)
    reference = Image.new("RGB", (16, 16), "white")

    generated = model.generate_image(
        seed=11,
        prompt="make it black",
        image_paths=reference,
        num_inference_steps=1,
        guidance=1.0,
    )

    assert generated.image.size == (16, 16)
    assert generated.model_config is ModelConfig.mage_flow_edit_turbo()
    assert generated.image_paths is None


def test_mage_flow_edit_resolves_random_seed_sentinel_before_vae_sampling(monkeypatch):
    import mflux.models.mage_flow.variants.edit.mage_flow_edit as pipeline_module

    original_predictor = make_velocity_predictor
    monkeypatch.setattr(
        pipeline_module,
        "make_velocity_predictor",
        lambda **kwargs: original_predictor(**kwargs, compile_model=False),
    )
    monkeypatch.setattr(
        "mflux.models.mage_flow.variants.pipeline_helpers.random.randint",
        lambda low, high: 987654321,
    )
    tokenizer = SimpleNamespace(processor=_EditProcessor())
    model = _bare_pipeline(MageFlowEdit, ModelConfig.mage_flow_edit_turbo(), tokenizer)

    generated = model.generate_image(
        seed=-1,
        prompt="make it black",
        image_paths=Image.new("RGB", (16, 16), "white"),
        num_inference_steps=1,
        guidance=1.0,
    )

    assert generated.seed == 987654321
