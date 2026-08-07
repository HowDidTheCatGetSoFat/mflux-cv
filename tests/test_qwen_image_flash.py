import math
import sys
import warnings

import numpy as np
import pytest

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.qwen.cli import qwen_image_generate


# --- config resolution ----------------------------------------------------------------


@pytest.mark.fast
def test_both_aliases_resolve_to_flash():
    for alias in ("qwen-image-flash", "qwen-flash"):
        config = ModelConfig.from_name(alias)
        assert config.model_name == "nvidia/Qwen-Image-Flash"


@pytest.mark.fast
def test_hf_name_resolves_to_flash():
    config = ModelConfig.from_name("nvidia/Qwen-Image-Flash")
    assert "qwen-image-flash" in config.aliases
    assert config.supports_guidance is False


@pytest.mark.fast
def test_flash_declares_static_shift_three_and_no_cfg():
    config = ModelConfig.qwen_image_flash()
    assert config.supports_guidance is False  # DMD2 distillation: CFG is internalized
    assert config.requires_sigma_shift is True
    # A STATIC shift S is expressed as sigma_base_shift == sigma_max_shift == ln(S):
    # the dynamic-mu line collapses to a resolution-independent mu.
    assert config.sigma_base_shift == config.sigma_max_shift
    assert config.sigma_base_shift == pytest.approx(math.log(3.0))
    assert config.sigma_shift_terminal is None  # shift_terminal is null upstream


# --- scheduler: the static-shift case -------------------------------------------------


@pytest.mark.fast
def test_linear_scheduler_applies_static_shift_three_resolution_independently():
    # FlowMatch static shift: sigma' = S*s / (1 + (S-1)*s). For S=3 and the 4-step
    # base ramp [1.0, 0.75, 0.5, 0.25] that is exactly [1.0, 0.9, 0.75, 0.5].
    expected = np.array([1.0, 0.9, 0.75, 0.5, 0.0], dtype=np.float32)
    for width, height in ((1024, 1024), (512, 768)):
        config = Config(
            model_config=ModelConfig.qwen_image_flash(),
            num_inference_steps=4,
            width=width,
            height=height,
            scheduler="linear",
        )
        np.testing.assert_allclose(np.asarray(config.scheduler.sigmas), expected, rtol=1e-6, atol=1e-6)


# --- CLI defaults ---------------------------------------------------------------------


@pytest.mark.fast
def test_flash_step_defaults_and_model_choices():
    for alias in ("qwen-image-flash", "qwen-flash"):
        assert alias in ui_defaults.MODEL_CHOICES  # keeps the parser from treating the alias as a local path
        assert ui_defaults.MODEL_INFERENCE_STEPS[alias] == 4
    assert ui_defaults.MODEL_INFERENCE_STEPS["nvidia/Qwen-Image-Flash"] == 4


@pytest.mark.fast
def test_qwen_parser_defaults_steps_to_four_for_flash(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--model", "qwen-image-flash", "--prompt", "x"])
    args = qwen_image_generate.build_parser().parse_args()
    assert args.steps == 4
    assert args.model_path is None


@pytest.mark.fast
def test_flash_effective_guidance_is_pinned_to_one():
    flash = ModelConfig.qwen_image_flash()
    assert qwen_image_generate.resolve_effective_guidance(None, flash) == 1.0
    assert qwen_image_generate.resolve_effective_guidance(3.0, flash) == 1.0


@pytest.mark.fast
def test_cfg_models_keep_the_historical_guidance_default():
    qwen = ModelConfig.qwen_image()
    assert qwen_image_generate.resolve_effective_guidance(None, qwen) == ui_defaults.GUIDANCE_SCALE
    assert qwen_image_generate.resolve_effective_guidance(2.0, qwen) == 2.0


# --- CLI warning wiring (model constructor stubbed; parsing and warnings are real) ----


class _ModelStubbed(Exception):
    """Raised by the stubbed model constructor: everything before it (parsing, model
    resolution and the ignored-option warnings) ran through the CLI's real main()."""


def _run_qwen_main(monkeypatch, argv):
    captured = {}

    def boom(*args, **kwargs):
        captured.update(kwargs)
        raise _ModelStubbed

    monkeypatch.setattr(qwen_image_generate, "QwenImage", boom)
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_ModelStubbed):
            qwen_image_generate.main()
    return [str(w.message) for w in caught], captured


@pytest.mark.fast
def test_qwen_cli_flash_warns_on_guidance_and_negative_prompt(monkeypatch):
    messages, captured = _run_qwen_main(
        monkeypatch,
        ["--model", "qwen-image-flash", "--prompt", "x", "--guidance", "3.0", "--negative-prompt", "y"],
    )
    assert any("--guidance is ignored" in m for m in messages)
    assert any("--negative-prompt is ignored" in m for m in messages)
    assert captured["model_config"].model_name == "nvidia/Qwen-Image-Flash"


@pytest.mark.fast
def test_qwen_cli_default_model_stays_silent_and_resolves_2512(monkeypatch):
    messages, captured = _run_qwen_main(
        monkeypatch,
        ["--prompt", "x", "--guidance", "3.0", "--negative-prompt", "y"],
    )
    assert messages == []
    assert captured["model_config"].model_name == "Qwen/Qwen-Image-2512"
