import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.utils.version_util import VersionUtil


class _SingleComponentWeightDefinition:
    @staticmethod
    def get_download_patterns() -> list[str]:
        return ["component/*.safetensors"]

    @staticmethod
    def get_components() -> list[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="component",
                hf_subdir="component",
                mapping_getter=None,
            )
        ]


class _TwoComponentWeightDefinition:
    @staticmethod
    def get_download_patterns() -> list[str]:
        return ["first/*.safetensors", "second/*.safetensors"]

    @staticmethod
    def get_components() -> list[ComponentDefinition]:
        return [
            ComponentDefinition(name="first", hf_subdir="first", mapping_getter=None),
            ComponentDefinition(name="second", hf_subdir="second", mapping_getter=None),
        ]


def _metadata(quantization_level: int | None, version: str) -> dict[str, str]:
    return {
        "quantization_level": str(quantization_level),
        "mflux_version": version,
    }


def _write_index(
    component_path: Path,
    weight_map: dict[str, str],
    metadata: dict[str, str],
) -> None:
    component_path.mkdir(parents=True, exist_ok=True)
    index = {
        "metadata": metadata,
        "weight_map": weight_map,
    }
    (component_path / "model.safetensors.index.json").write_text(json.dumps(index))


def _write_shard(
    component_path: Path,
    filename: str,
    weights: dict[str, mx.array],
    metadata: dict[str, str],
) -> None:
    component_path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(component_path / filename), weights, metadata)


@pytest.mark.parametrize("quantization_level", [None, 4])
def test_weight_loader_propagates_native_save_metadata_independently(
    tmp_path: Path,
    quantization_level: int | None,
) -> None:
    ModelSaver._save_weights(
        base_path=str(tmp_path),
        bits=quantization_level,
        model=nn.Linear(2, 2),
        subdir="component",
    )

    loaded = WeightLoader.load(_SingleComponentWeightDefinition, model_path=str(tmp_path))

    assert loaded.meta_data.quantization_level == quantization_level
    assert loaded.meta_data.mflux_version == VersionUtil.get_mflux_version()
    assert {key for key, _ in tree_flatten(loaded.component)} == {"bias", "weight"}


def test_mflux_loader_rejects_index_referencing_a_missing_shard(tmp_path: Path) -> None:
    component_path = tmp_path / "component"
    metadata = _metadata(4, "1.2.3")
    _write_shard(component_path, "0.safetensors", {"present.weight": mx.ones((1,))}, metadata)
    _write_index(
        component_path,
        {
            "present.weight": "0.safetensors",
            "missing.weight": "1.safetensors",
        },
        metadata,
    )

    with pytest.raises(FileNotFoundError, match=r"1\.safetensors"):
        WeightLoader._try_load_mflux_format(component_path)


def test_mflux_loader_rejects_partial_shard_tensor_keys(tmp_path: Path) -> None:
    component_path = tmp_path / "component"
    metadata = _metadata(None, "1.2.3")
    _write_shard(component_path, "0.safetensors", {"present.weight": mx.ones((1,))}, metadata)
    _write_index(
        component_path,
        {
            "present.weight": "0.safetensors",
            "missing.weight": "0.safetensors",
        },
        metadata,
    )

    with pytest.raises(ValueError, match=r"missing tensor keys.*missing\.weight"):
        WeightLoader._try_load_mflux_format(component_path)


def test_mflux_loader_rejects_unsafe_index_shard_filename(tmp_path: Path) -> None:
    component_path = tmp_path / "component"
    metadata = _metadata(4, "1.2.3")
    _write_index(component_path, {"weight": "../outside.safetensors"}, metadata)

    with pytest.raises(ValueError, match="Invalid MFLUX shard filename"):
        WeightLoader._try_load_mflux_format(component_path)


def test_mflux_loader_rejects_native_metadata_without_version(tmp_path: Path) -> None:
    component_path = tmp_path / "component"
    metadata = {"quantization_level": "None"}
    _write_shard(component_path, "0.safetensors", {"weight": mx.ones((1,))}, metadata)
    _write_index(component_path, {"weight": "0.safetensors"}, metadata)

    with pytest.raises(ValueError, match="Missing or invalid MFLUX version metadata"):
        WeightLoader._try_load_mflux_format(component_path)


def test_weight_loader_rejects_inconsistent_metadata_across_components(tmp_path: Path) -> None:
    for component_name, quantization_level in (("first", 4), ("second", 8)):
        component_path = tmp_path / component_name
        metadata = _metadata(quantization_level, "1.2.3")
        _write_shard(component_path, "0.safetensors", {"weight": mx.ones((1,))}, metadata)
        _write_index(component_path, {"weight": "0.safetensors"}, metadata)

    with pytest.raises(ValueError, match="Inconsistent MFLUX metadata across components"):
        WeightLoader.load(_TwoComponentWeightDefinition, model_path=str(tmp_path))


@pytest.mark.parametrize(
    ("second_quantization_level", "second_version"),
    [
        (8, "1.2.3"),
        (4, "2.0.0"),
    ],
)
def test_mflux_loader_rejects_inconsistent_metadata_across_shards(
    tmp_path: Path,
    second_quantization_level: int,
    second_version: str,
) -> None:
    component_path = tmp_path / "component"
    metadata = _metadata(4, "1.2.3")
    _write_shard(component_path, "0.safetensors", {"first.weight": mx.ones((1,))}, metadata)
    _write_shard(
        component_path,
        "1.safetensors",
        {"second.weight": mx.ones((1,))},
        _metadata(second_quantization_level, second_version),
    )
    _write_index(
        component_path,
        {
            "first.weight": "0.safetensors",
            "second.weight": "1.safetensors",
        },
        metadata,
    )

    with pytest.raises(ValueError, match="Inconsistent MFLUX metadata"):
        WeightLoader._try_load_mflux_format(component_path)
