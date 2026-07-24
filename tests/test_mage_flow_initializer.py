from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.mage_flow.mage_flow_initializer import MageFlowInitializer
from mflux.models.mage_flow.weights import MageFlowWeightDefinition


@pytest.mark.fast
def test_mage_flow_initializer_shares_resolved_snapshot_for_weights_and_tokenizers(tmp_path: Path) -> None:
    shared_root = tmp_path / "snapshots" / "rev-a"
    shared_root.mkdir(parents=True)
    seen: dict[str, str] = {}

    def fake_resolve(*, path, patterns, required_pattern_groups):
        assert path == "microsoft/Mage-Flow"
        assert patterns == MageFlowWeightDefinition.get_download_patterns()
        assert required_pattern_groups == MageFlowWeightDefinition.get_required_download_pattern_groups()
        return shared_root

    def fake_load_weights(*, weight_definition, model_path):
        seen["weights"] = model_path
        return LoadedWeights(components={}, meta_data=MetaData(mflux_version="test"))

    def fake_load_tokenizers(*, definitions, model_path):
        seen["tokenizers"] = model_path
        return {"mage": object()}

    model = SimpleNamespace()
    model_config = SimpleNamespace(model_name="microsoft/Mage-Flow")

    with (
        patch("mflux.models.mage_flow.mage_flow_initializer.PathResolution.resolve", side_effect=fake_resolve),
        patch("mflux.models.mage_flow.mage_flow_initializer.WeightLoader.load", side_effect=fake_load_weights),
        patch(
            "mflux.models.mage_flow.mage_flow_initializer.TokenizerLoader.load_all",
            side_effect=fake_load_tokenizers,
        ),
        patch.object(MageFlowWeightDefinition, "validate_loaded_weights", return_value=None),
        patch.object(MageFlowInitializer, "_init_models", return_value=None),
        patch.object(MageFlowInitializer, "_validate_hf_model_coverage", return_value=None),
        patch.object(MageFlowInitializer, "_apply_weights", return_value=None),
        patch.object(mx, "eval", return_value=None),
        patch.object(mx, "clear_cache", return_value=None),
    ):
        MageFlowInitializer.init(model, model_config, quantize=None)

    assert seen["weights"] == str(shared_root)
    assert seen["tokenizers"] == str(shared_root)
    assert seen["weights"] == seen["tokenizers"]
    assert "mage" in model.tokenizers
