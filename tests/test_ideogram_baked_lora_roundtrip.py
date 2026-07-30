import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4


@pytest.mark.fast
def test_save_records_baked_lora_in_config(tmp_path):
    model = SimpleNamespace(
        lora_paths=["/some/where/fxavatar.safetensors"],
        lora_scales=[0.8],
    )
    ModelSaver._record_baked_lora(model=model, base_path=str(tmp_path))
    config = json.loads((tmp_path / ConfigResolution.SAVED_CONFIG_FILENAME).read_text())
    assert config["baked_lora"] == {"paths": ["fxavatar.safetensors"], "scales": [0.8]}


@pytest.mark.fast
def test_save_without_lora_records_nothing(tmp_path):
    model = SimpleNamespace(lora_paths=None, lora_scales=None)
    ModelSaver._record_baked_lora(model=model, base_path=str(tmp_path))
    assert not (tmp_path / ConfigResolution.SAVED_CONFIG_FILENAME).exists()


@pytest.mark.fast
def test_cfg_negative_routing_survives_reload():
    # A reloaded native save has no lora_paths and no live LoRA layers; only the
    # baked_lora marker keeps the CFG negative routed through the conditional
    # transformer. Without it the clean unconditional negative amplifies the baked
    # LoRA delta at full guidance.
    reloaded = SimpleNamespace(
        lora_paths=None,
        baked_lora={"paths": ["fxavatar.safetensors"], "scales": [1.0]},
        conditional_transformer=object(),
    )
    assert Ideogram4._lora_is_active(reloaded)

    plain = SimpleNamespace(
        lora_paths=None,
        baked_lora=None,
        conditional_transformer=object(),
    )
    assert not Ideogram4._lora_is_active(plain)


@pytest.mark.fast
def test_validator_skips_deferred_fp8_placeholders():
    # Fp8Linear leaves large weights as empty arrays until update fills them; the
    # native-checkpoint validator must not read that placeholder shape as a mismatch.
    class Holder:
        def parameters(self):
            return {"big": {"weight": mx.array([], dtype=mx.uint8)}}

    weights = LoadedWeights(components={}, meta_data=SimpleNamespace(mflux_version="0.18.31"))
    checkpoint = {"big": {"weight": mx.zeros((512, 4608), dtype=mx.uint8)}}
    WeightApplier._validate_native_component(weights, "transformer", Holder(), checkpoint)


@pytest.mark.fast
def test_validator_still_rejects_real_shape_mismatches():
    class Holder:
        def parameters(self):
            return {"small": {"weight": mx.zeros((8, 16))}}

    weights = LoadedWeights(components={}, meta_data=SimpleNamespace(mflux_version="0.18.31"))
    checkpoint = {"small": {"weight": mx.zeros((8, 32))}}
    with pytest.raises(ValueError, match="shape_mismatches"):
        WeightApplier._validate_native_component(weights, "transformer", Holder(), checkpoint)
