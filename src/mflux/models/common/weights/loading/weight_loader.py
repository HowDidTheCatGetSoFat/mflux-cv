import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten
from safetensors.torch import load_file as torch_load_file

from mflux.cli.defaults.defaults import MFLUX_CACHE_DIR
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.safetensors_reader import SafetensorsReader
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import WeightDefinitionType

logger = logging.getLogger(__name__)


class WeightLoader:
    @staticmethod
    def load_single(
        component: ComponentDefinition,
        repo_id: str,
        file_pattern: str = "*.safetensors",
    ) -> LoadedWeights:
        # repo_id may be a HuggingFace repo or a local directory (--controlnet-path accepts both);
        # PathResolution handles each, where a bare snapshot_download would treat a local path as a
        # repo id. HF repos keep the same allow_patterns, so their download behavior is unchanged.
        root_path = PathResolution.resolve(path=repo_id, patterns=[file_pattern, "config.json"])
        weights, q_level, version = WeightLoader._load_component(root_path, component)
        return LoadedWeights(
            components={component.name: weights},
            meta_data=MetaData(quantization_level=q_level, mflux_version=version),
        )

    @staticmethod
    def load(
        weight_definition: "WeightDefinitionType",
        model_path: str | None = None,
        download_patterns: list[str] | None = None,
        path_overrides: dict[str, str] | None = None,
    ) -> LoadedWeights:
        # download_patterns lets a caller supply variant-aware HF allow_patterns (e.g. Krea 2 Turbo
        # vs Raw need different transformer layouts); otherwise fall back to the definition's default.
        root_path = PathResolution.resolve(
            path=model_path,
            patterns=download_patterns if download_patterns is not None else weight_definition.get_download_patterns(),
            required_pattern_groups=(
                weight_definition.get_required_download_pattern_groups()
                if hasattr(weight_definition, "get_required_download_pattern_groups")
                else None
            ),
        )

        components = {}
        quantization_level = None
        mflux_version = None
        component_quantization_levels: dict[str, int | None] = {}
        native_metadata: tuple[int | None, str] | None = None
        raw_weights_cache: dict[tuple, dict] = {}  # Cache by (path, loading_mode, weight_files)

        for component in weight_definition.get_components():
            if path_overrides and component.name in path_overrides:
                override_root = Path(path_overrides[component.name])
                if not override_root.exists():
                    raise FileNotFoundError(
                        f"--model-{component.name.replace('_', '-')}: path does not exist: {override_root}"
                    )
                component_path = override_root / component.hf_subdir
                if not component_path.exists():
                    raise FileNotFoundError(
                        f"--model-{component.name.replace('_', '-')}: expected '{component.hf_subdir}/' subdirectory "
                        f"not found in {override_root}"
                    )
                weights, q_level, version = WeightLoader._load_component(override_root, component)
                component_quantization_levels[component.name] = q_level
            else:
                weights, q_level, version = WeightLoader._load_component(root_path, component, raw_weights_cache)

            components[component.name] = weights

            if version is not None:
                component_metadata = (q_level, version)
                if native_metadata is not None and component_metadata != native_metadata:
                    raise ValueError(
                        f"Inconsistent MFLUX metadata across components: "
                        f"expected quantization_level={native_metadata[0]}, "
                        f"mflux_version={native_metadata[1]!r}; {component.name} has "
                        f"quantization_level={q_level}, mflux_version={version!r}"
                    )
                native_metadata = component_metadata
                quantization_level, mflux_version = component_metadata

        return LoadedWeights(
            components=components,
            meta_data=MetaData(
                quantization_level=quantization_level,
                mflux_version=mflux_version,
                component_quantization_levels=component_quantization_levels,
            ),
        )

    @staticmethod
    def _load_component(
        root_path: Path | None,
        component: ComponentDefinition,
        raw_weights_cache: dict[tuple, dict] | None = None,
    ) -> tuple[dict, int | None, str | None]:
        # Some components are distributed in more than one on-disk layout (e.g. a native
        # single-file checkpoint vs a diffusers sharded directory with different keys).
        # Let the component pick the concrete definition based on what is present on disk.
        if component.variant_selector is not None and root_path is not None:
            component = component.variant_selector(root_path)

        # Handle direct URL downloads (e.g., Apple CDN for DepthPro)
        if component.download_url is not None:
            file_path = WeightLoader._download_from_url(component.download_url, component.name)
            raw_weights = WeightLoader._load_weights_file(file_path, component.loading_mode)
        else:
            if root_path is None:
                raise ValueError(f"No root_path and no download_url for component: {component.name}")
            component_path = root_path / component.hf_subdir

            # Try mflux saved format first (including FP8 components reloaded after mflux-save).
            weights, q_level, version = WeightLoader._try_load_mflux_format(component_path)
            if weights is not None:
                return weights, q_level, version

            # Check cache for shared loading (e.g., FIBO VLM decoder + visual from same source)
            cache_key = (str(component_path), component.loading_mode, tuple(component.weight_files or []))
            if raw_weights_cache is not None and cache_key in raw_weights_cache:
                raw_weights = raw_weights_cache[cache_key]
            else:
                # Fall back to HuggingFace format with mapping
                raw_weights = WeightLoader._load_safetensors(
                    component_path, component.loading_mode, component.weight_files
                )
                # Cache for potential reuse by other components
                if raw_weights_cache is not None:
                    raw_weights_cache[cache_key] = raw_weights

        # Apply prefix filtering if specified (e.g., filter "model.language_model" vs "model.visual")
        if component.weight_prefix_filters is not None:
            raw_weights = {
                k: v
                for k, v in raw_weights.items()
                if any(k.startswith(prefix) for prefix in component.weight_prefix_filters)
            }

        if component.key_transform is not None:
            transformed_weights = {}
            for key, value in raw_weights.items():
                transformed_key = component.key_transform(key)
                if transformed_key is not None:
                    transformed_weights[transformed_key] = value
            raw_weights = transformed_weights

        if component.weight_transform is not None:
            raw_weights = {k: component.weight_transform(k, v) for k, v in raw_weights.items()}

        # Apply precision conversion if specified
        if component.precision is not None:
            raw_weights = WeightLoader._convert_precision(raw_weights, component.precision)

        # Passthrough mode: apply bulk transform and unflatten (no key mapping)
        if component.mapping_getter is None:
            if component.bulk_transform is not None:
                raw_weights = {k: component.bulk_transform(v) for k, v in raw_weights.items()}
            return tree_unflatten(list(raw_weights.items())), None, None

        # Standard mode: apply declarative weight mapping
        mapped_weights = WeightMapper.apply_mapping(
            hf_weights=raw_weights,
            mapping=component.mapping_getter(),
            num_blocks=component.num_blocks,
            num_layers=component.num_layers,
        )
        return mapped_weights, None, None

    @staticmethod
    def _try_load_mflux_format(path: Path) -> tuple[dict | None, int | None, str | None]:
        if not path.exists():
            return None, None, None

        index_path = path / "model.safetensors.index.json"
        index: dict | None = None
        index_metadata: tuple[int | None, str | None] | None = None
        if index_path.exists():
            with index_path.open(encoding="utf-8") as index_file:
                loaded_index = json.load(index_file)
            if not isinstance(loaded_index, dict):
                raise ValueError(f"Invalid MFLUX weight index in {index_path}: expected a JSON object")
            index = loaded_index
            index_metadata = WeightLoader._parse_mflux_metadata(index.get("metadata"), index_path)

        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        first_shard_metadata = None
        if index_metadata is None and shard_files:
            first_shard = mx.load(str(shard_files[0]), return_metadata=True)
            first_shard_metadata = WeightLoader._parse_mflux_metadata(first_shard[1], shard_files[0])

        # Hugging Face checkpoints do not carry MFLUX metadata and use their normal loader.
        if index_metadata is None and first_shard_metadata is None:
            return None, None, None

        if index is None:
            raise FileNotFoundError(f"MFLUX weight index not found: {index_path}")
        if index_metadata is None:
            raise ValueError(f"MFLUX metadata is missing from weight index: {index_path}")

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid MFLUX weight index in {index_path}: weight_map must be a non-empty object")

        keys_by_shard: dict[str, list[str]] = {}
        for tensor_key, shard_filename in weight_map.items():
            if not isinstance(tensor_key, str) or not tensor_key:
                raise ValueError(f"Invalid MFLUX tensor key in {index_path}: {tensor_key!r}")
            WeightLoader._validate_mflux_shard_filename(shard_filename, index_path)
            keys_by_shard.setdefault(shard_filename, []).append(tensor_key)

        quantization_level, mflux_version = index_metadata
        all_weights: dict[str, mx.array] = {}
        for shard_filename, expected_keys in keys_by_shard.items():
            shard_path = path / shard_filename
            if not shard_path.is_file():
                raise FileNotFoundError(f"MFLUX weight shard referenced by {index_path} is missing: {shard_path}")

            shard_weights, raw_shard_metadata = mx.load(str(shard_path), return_metadata=True)
            shard_metadata = WeightLoader._parse_mflux_metadata(raw_shard_metadata, shard_path)
            if shard_metadata != index_metadata:
                raise ValueError(
                    f"Inconsistent MFLUX metadata in {shard_path}: "
                    f"expected quantization_level={quantization_level}, mflux_version={mflux_version!r}; "
                    f"got {shard_metadata!r}"
                )

            expected_key_set = set(expected_keys)
            actual_key_set = set(shard_weights)
            missing_keys = sorted(expected_key_set - actual_key_set)
            unexpected_keys = sorted(actual_key_set - expected_key_set)
            if missing_keys or unexpected_keys:
                problems = []
                if missing_keys:
                    problems.append(f"missing tensor keys {WeightLoader._summarize_keys(missing_keys)}")
                if unexpected_keys:
                    problems.append(f"unexpected tensor keys {WeightLoader._summarize_keys(unexpected_keys)}")
                raise ValueError(f"MFLUX shard {shard_path} does not match {index_path}: {'; '.join(problems)}")

            all_weights.update((key, shard_weights[key]) for key in expected_keys)

        unflattened = tree_unflatten(list(all_weights.items()))
        return unflattened, quantization_level, mflux_version

    @staticmethod
    def _parse_mflux_metadata(metadata: object, source: Path) -> tuple[int | None, str | None] | None:
        if not isinstance(metadata, dict):
            return None
        if "quantization_level" not in metadata and "mflux_version" not in metadata:
            return None

        raw_quantization_level = metadata.get("quantization_level")
        if raw_quantization_level in (None, "None", "null", ""):
            quantization_level = None
        elif isinstance(raw_quantization_level, bool):
            raise ValueError(f"Invalid MFLUX quantization metadata in {source}: {raw_quantization_level!r}")
        else:
            try:
                quantization_level = int(raw_quantization_level)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid MFLUX quantization metadata in {source}: {raw_quantization_level!r}"
                ) from error

        mflux_version = metadata.get("mflux_version")
        if not isinstance(mflux_version, str) or not mflux_version.strip():
            raise ValueError(f"Missing or invalid MFLUX version metadata in {source}: {mflux_version!r}")
        return quantization_level, mflux_version

    @staticmethod
    def _validate_mflux_shard_filename(shard_filename: object, index_path: Path) -> None:
        if not isinstance(shard_filename, str) or not shard_filename:
            raise ValueError(f"Invalid MFLUX shard filename in {index_path}: {shard_filename!r}")

        shard_path = Path(shard_filename)
        if (
            shard_path.is_absolute()
            or shard_path.name != shard_filename
            or shard_path.suffix != ".safetensors"
            or shard_filename.startswith("._")
        ):
            raise ValueError(f"Invalid MFLUX shard filename in {index_path}: {shard_filename!r}")

    @staticmethod
    def _summarize_keys(keys: list[str], limit: int = 5) -> str:
        summary = ", ".join(repr(key) for key in keys[:limit])
        if len(keys) > limit:
            summary += f", ... ({len(keys)} total)"
        return f"[{summary}]"

    @staticmethod
    def _download_from_url(url: str, component_name: str) -> Path:
        cache_dir = MFLUX_CACHE_DIR / component_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename from URL
        filename = url.split("/")[-1]
        file_path = cache_dir / filename

        if not file_path.exists():
            logger.info(f"Downloading {component_name} weights from {url}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                logger.info(f"Downloaded to {file_path}")
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.error(f"Failed to download: {e}")
                logger.info(f"Please manually download from: {url}")
                raise FileNotFoundError(f"Model file not found at {file_path}") from e

        return file_path

    @staticmethod
    def _load_weights_file(file_path: Path, loading_mode: str) -> dict[str, mx.array]:
        if loading_mode == "torch_checkpoint":
            return WeightLoader._load_torch_checkpoint(file_path)
        elif loading_mode in ("mlx_native", "single"):
            data = mx.load(str(file_path), return_metadata=True)
            return dict(data[0].items())
        else:
            raise ValueError(f"Unsupported loading mode for single file: {loading_mode}")

    @staticmethod
    def _load_torch_checkpoint(file_path: Path) -> dict[str, mx.array]:
        pt_weights = torch.load(file_path, map_location="cpu", weights_only=False)
        return {k: mx.array(v.numpy()) for k, v in pt_weights.items() if isinstance(v, torch.Tensor)}

    @staticmethod
    def _load_safetensors(path: Path, loading_mode: str, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if loading_mode == "mlx_native":
            return WeightLoader._load_mlx_native(path, weight_files)
        elif loading_mode == "torch_convert":
            return WeightLoader._load_torch_convert(path, weight_files)
        elif loading_mode == "multi_json":
            return WeightLoader._load_multi_json(path)
        elif loading_mode == "torch_bfloat16":
            return WeightLoader._load_torch_bfloat16(path)
        elif loading_mode == "single":
            return WeightLoader._load_single(path)
        elif loading_mode == "multi_glob":
            return WeightLoader._load_multi_glob(path)
        elif loading_mode == "fp8_safetensors":
            return WeightLoader._load_fp8_safetensors(path)
        else:
            raise ValueError(f"Unknown loading mode: {loading_mode}")

    @staticmethod
    def _load_mlx_native(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if weight_files:
            # Load only specified files
            missing = [f for f in weight_files if not (path / f).exists()]
            if missing:
                raise FileNotFoundError(f"Missing specified weight files in {path}: {missing}")
            shard_files = [path / f for f in weight_files]
        else:
            # Fall back to loading all safetensors files
            shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
            if not shard_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            weights = mx.load(str(shard))
            all_weights.update(weights)

        return all_weights

    @staticmethod
    def _load_torch_convert(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if weight_files:
            # Load only specified files
            missing = [f for f in weight_files if not (path / f).exists()]
            if missing:
                raise FileNotFoundError(f"Missing specified weight files in {path}: {missing}")
            shard_files = [path / f for f in weight_files]
        else:
            # Fall back to loading all safetensors files
            shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
            if not shard_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            torch_weights = torch_load_file(str(shard))
            for key, tensor in torch_weights.items():
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                all_weights[key] = mx.array(tensor.numpy())

        return all_weights

    @staticmethod
    def _load_multi_json(path: Path) -> dict[str, mx.array]:
        index_path = path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        # Group weights by file
        files_to_load: dict[str, list[str]] = {}
        for param_name, file_name in index["weight_map"].items():
            if file_name not in files_to_load:
                files_to_load[file_name] = []
            files_to_load[file_name].append(param_name)

        all_weights: dict[str, mx.array] = {}
        for file_name, param_names in files_to_load.items():
            file_path = path / file_name

            # Use mx.load which handles bfloat16 natively
            file_weights = mx.load(str(file_path))

            for param_name in param_names:
                if param_name in file_weights:
                    all_weights[param_name] = file_weights[param_name]

        return all_weights

    @staticmethod
    def _load_torch_bfloat16(path: Path) -> dict[str, mx.array]:
        index_path = path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        weight_files = sorted(set(index["weight_map"].values()))

        all_weights: dict[str, mx.array] = {}
        for wf in weight_files:
            file_path = path / wf
            data = torch_load_file(str(file_path))
            for k, v in data.items():
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float16)
                np_arr = v.detach().cpu().numpy()
                all_weights[k] = mx.array(np_arr)

        return all_weights

    @staticmethod
    def _load_single(path: Path) -> dict[str, mx.array]:
        safetensors_files = [f for f in path.glob("*.safetensors") if not f.name.startswith("._")]
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        weights_file = safetensors_files[0]
        data = mx.load(str(weights_file), return_metadata=True)
        return dict(data[0].items())

    @staticmethod
    def _load_multi_glob(path: Path) -> dict[str, mx.array]:
        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            data, _ = mx.load(str(shard), return_metadata=True)
            all_weights.update(dict(data.items()))

        return all_weights

    @staticmethod
    def _load_fp8_safetensors(path: Path) -> dict[str, mx.array]:
        return SafetensorsReader.read_directory(path)

    @staticmethod
    def _convert_precision(weights: dict[str, mx.array], precision: mx.Dtype) -> dict[str, mx.array]:
        return {k: v if v.dtype == precision else v.astype(precision) for k, v in weights.items()}
