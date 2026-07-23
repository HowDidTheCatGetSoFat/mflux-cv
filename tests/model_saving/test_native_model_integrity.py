from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.saving.model_saver import ModelSaver


class _TinyWeightDefinition:
    @staticmethod
    def get_components():
        return [ComponentDefinition(name="transformer", hf_subdir="transformer")]

    @staticmethod
    def get_tokenizers():
        return []


@pytest.mark.fast
def test_native_mflux_update_rejects_missing_tensor():
    model = nn.Linear(2, 2)
    weights = LoadedWeights(
        components={"transformer": {"weight": mx.zeros_like(model.weight)}},
        meta_data=MetaData(mflux_version="test"),
    )

    with pytest.raises(ValueError, match=r"missing=\['bias'\]"):
        WeightApplier.apply_and_quantize(
            weights=weights,
            models={"transformer": model},
            quantize_arg=None,
            weight_definition=_TinyWeightDefinition,
        )


@pytest.mark.fast
def test_native_mflux_update_rejects_shape_mismatch():
    model = nn.Linear(2, 2)
    weights = LoadedWeights(
        components={
            "transformer": {
                "weight": mx.zeros((3, 2)),
                "bias": mx.zeros_like(model.bias),
            }
        },
        meta_data=MetaData(mflux_version="test"),
    )

    with pytest.raises(ValueError, match=r"shape_mismatches=\['weight'\]"):
        WeightApplier.apply_and_quantize(
            weights=weights,
            models={"transformer": model},
            quantize_arg=None,
            weight_definition=_TinyWeightDefinition,
        )


@pytest.mark.fast
def test_model_saver_rejects_evicted_required_component_before_writing(tmp_path):
    output = tmp_path / "incomplete"
    model = SimpleNamespace(transformer=None, tokenizers={})

    with pytest.raises(ValueError, match="required components are unloaded or missing"):
        ModelSaver.save_model(
            model=model,
            bits=None,
            base_path=str(output),
            weight_definition=_TinyWeightDefinition,
        )

    assert not output.exists()


@pytest.mark.fast
def test_saved_model_directory_resolves_its_persisted_base_config(tmp_path):
    output = tmp_path / "opaque-directory-name"
    model = SimpleNamespace(
        transformer=nn.Linear(2, 2),
        tokenizers={},
        model_config=ModelConfig.mage_flow_turbo(),
    )

    ModelSaver.save_model(
        model=model,
        bits=None,
        base_path=str(output),
        weight_definition=_TinyWeightDefinition,
    )
    resolved = ModelConfig.from_name(str(output))

    assert (output / ConfigResolution.SAVED_CONFIG_FILENAME).is_file()
    assert resolved.model_name == str(output)
    assert resolved.base_model == "microsoft/Mage-Flow-Turbo"
    assert resolved.supports_guidance is False
