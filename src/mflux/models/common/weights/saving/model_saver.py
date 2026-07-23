import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from mflux.models.common.lora.mapping.lora_saver import LoRASaver
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.utils.version_util import VersionUtil

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import WeightDefinitionType


class ModelSaver:
    @staticmethod
    def save_model(
        model: Any,
        bits: int | None,
        base_path: str,
        weight_definition: "WeightDefinitionType",
    ) -> None:
        component_defs = weight_definition.get_components()
        missing_components = [
            component.model_attr or component.name
            for component in component_defs
            if getattr(model, component.model_attr or component.name, None) is None
        ]
        if missing_components:
            raise ValueError(
                "Cannot save an incomplete model; required components are unloaded or missing: "
                f"{missing_components}. Reload the model before saving."
            )

        ModelSaver._save_model_config(model=model, base_path=base_path)

        # Save tokenizers from model.tokenizers dict
        tokenizer_defs = weight_definition.get_tokenizers()
        for t in tokenizer_defs:
            if hasattr(model, "tokenizers") and t.name in model.tokenizers:
                tokenizer_wrapper = model.tokenizers[t.name]
                if hasattr(tokenizer_wrapper, "tokenizer"):
                    ModelSaver._save_tokenizer(base_path, tokenizer_wrapper.tokenizer, t.hf_subdir)

        # Save model components with progress bar
        components = [(c.model_attr or c.name, c.hf_subdir) for c in component_defs]
        for attr_name, subdir in tqdm(components, desc="Saving components", unit="component"):
            component = getattr(model, attr_name, None)
            if component is not None:
                # Bake and strip any LoRA wrappers to avoid duplicating shared weights. strict=True:
                # fail the save loudly if any layer can't bake rather than write a checkpoint that
                # silently drops that layer's adaptation.
                LoRASaver.bake_and_strip_lora(component, strict=True)
                ModelSaver._save_weights(base_path, bits, component, subdir)

    @staticmethod
    def _save_model_config(model: Any, base_path: str) -> None:
        model_config = getattr(model, "model_config", None)
        if model_config is None:
            return

        base_model = model_config.base_model or model_config.model_name
        path = Path(base_path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / ConfigResolution.SAVED_CONFIG_FILENAME).open("w", encoding="utf-8") as config_file:
            json.dump(
                {
                    "model_name": model_config.model_name,
                    "base_model": base_model,
                },
                config_file,
                indent=2,
            )

    @staticmethod
    def _save_tokenizer(base_path: str, tokenizer: PreTrainedTokenizer, subdir: str) -> None:
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)

    @staticmethod
    def _save_weights(base_path: str, bits: int | None, model: nn.Module, subdir: str) -> None:
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(model.parameters()))
        shards = ModelSaver._split_weights(weights)

        # Build weight_map for index.json (maps each weight key to its shard file)
        weight_map = {}
        shard_iter = tqdm(enumerate(shards), total=len(shards), desc=f"  {subdir}", unit="shard", leave=False)
        for i, shard in shard_iter:
            shard_filename = f"{i}.safetensors"
            mx.save_safetensors(
                str(path / shard_filename),
                shard,
                {
                    "quantization_level": str(bits),
                    "mflux_version": VersionUtil.get_mflux_version(),
                },
            )
            # Record which file each weight belongs to
            for key in shard.keys():
                weight_map[key] = shard_filename

        # Write model.safetensors.index.json for HuggingFace compatibility
        # This ensures the saved model works even if custom metadata is stripped
        index_data = {
            "metadata": {
                "quantization_level": str(bits),
                "mflux_version": VersionUtil.get_mflux_version(),
            },
            "weight_map": weight_map,
        }
        with open(path / "model.safetensors.index.json", "w") as f:
            json.dump(index_data, f, indent=2)

    @staticmethod
    def _split_weights(weights: dict, max_file_size_gb: int = 2) -> list[dict]:
        max_file_size_bytes = max_file_size_gb << 30
        shards: list[dict] = []
        shard: dict = {}
        shard_size = 0
        for k, v in weights.items():
            if shard_size + v.nbytes > max_file_size_bytes:
                shards.append(shard)
                shard, shard_size = {}, 0
            shard[k] = v
            shard_size += v.nbytes
        if shard:  # Don't append empty shard
            shards.append(shard)
        return shards
