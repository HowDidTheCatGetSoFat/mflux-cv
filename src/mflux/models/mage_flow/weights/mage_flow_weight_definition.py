from dataclasses import dataclass
from typing import List

import mlx.nn as nn
from mlx.utils import tree_flatten

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.mage_flow.weights.mage_flow_weight_mapping import MageFlowWeightMapping


@dataclass
class MageFlowComponentDefinition(ComponentDefinition):
    expected_hf_weight_count: int = 0
    folded_weight_count: int | None = None
    post_load_hook: str | None = None


class MageFlowWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            MageFlowComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="mlx_native",
                precision=ModelConfig.precision,
                key_transform=MageFlowWeightMapping.transform_vae_key,
                weight_transform=MageFlowWeightMapping.transform_vae_weight,
                expected_hf_weight_count=MageFlowWeightMapping.EXPECTED_HF_WEIGHT_COUNTS["vae"],
                folded_weight_count=MageFlowWeightMapping.EXPECTED_FOLDED_VAE_WEIGHT_COUNT,
                post_load_hook="freeze_adaln_cache",
            ),
            MageFlowComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                loading_mode="mlx_native",
                precision=ModelConfig.precision,
                key_transform=MageFlowWeightMapping.transform_transformer_key,
                expected_hf_weight_count=MageFlowWeightMapping.EXPECTED_HF_WEIGHT_COUNTS["transformer"],
            ),
            MageFlowComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="mlx_native",
                precision=ModelConfig.precision,
                skip_quantization=True,
                key_transform=MageFlowWeightMapping.transform_text_encoder_key,
                weight_transform=MageFlowWeightMapping.transform_text_encoder_weight,
                expected_hf_weight_count=MageFlowWeightMapping.EXPECTED_HF_WEIGHT_COUNTS["text_encoder"],
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        from mflux.models.common.tokenizer import VisionLanguageTokenizer
        from mflux.models.mage_flow.model.mage_flow_text_encoder import MageFlowQwen3VLProcessor

        return [
            TokenizerDefinition(
                name="mage",
                hf_subdir="text_encoder",
                tokenizer_class="AutoTokenizer",
                encoder_class=VisionLanguageTokenizer,
                processor_class=MageFlowQwen3VLProcessor,
                max_length=2112,
                padding="longest",
                template=None,
                download_patterns=["text_encoder/**"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "mflux_model_config.json",
            "model_index.json",
            "scheduler/scheduler_config.json",
            "vae/*.safetensors",
            "vae/config.json",
            "vae/model.safetensors.index.json",
            "transformer/*.safetensors",
            "transformer/config.json",
            "transformer/model.safetensors.index.json",
            "text_encoder/*.safetensors",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/config.json",
            "text_encoder/preprocessor_config.json",
            "text_encoder/tokenizer.json",
            "text_encoder/tokenizer_config.json",
            "text_encoder/vocab.json",
            "text_encoder/merges.txt",
            "text_encoder/**",
        ]

    @staticmethod
    def get_required_download_pattern_groups() -> List[List[str]]:
        """Alternative complete cache manifests for official and native saves."""

        official = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "vae/*.safetensors",
            "vae/config.json",
            "transformer/*.safetensors",
            "transformer/config.json",
            "text_encoder/*.safetensors",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/config.json",
            "text_encoder/preprocessor_config.json",
            "text_encoder/tokenizer.json",
            "text_encoder/tokenizer_config.json",
            "text_encoder/vocab.json",
            "text_encoder/merges.txt",
        ]
        native_mflux = [
            "vae/*.safetensors",
            "vae/model.safetensors.index.json",
            "transformer/*.safetensors",
            "transformer/model.safetensors.index.json",
            "text_encoder/*.safetensors",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/tokenizer.json",
            "text_encoder/tokenizer_config.json",
            "text_encoder/chat_template.jinja",
        ]
        return [official, native_mflux]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        if "t_embedder" in path or "adaLN_modulation" in path:
            return False
        if not hasattr(module, "to_quantized"):
            return False
        weight = getattr(module, "weight", None)
        return weight is None or weight.shape[-1] % 64 == 0

    @staticmethod
    def validate_loaded_weights(weights: LoadedWeights) -> None:
        if weights.meta_data.mflux_version is not None:
            return
        for component_name, expected_count in MageFlowWeightMapping.EXPECTED_HF_WEIGHT_COUNTS.items():
            component_weights = weights.components.get(component_name)
            if component_weights is None:
                raise ValueError(f"Missing Mage Flow weight component: {component_name}")
            actual_count = len(tree_flatten(component_weights))
            if actual_count != expected_count:
                raise ValueError(
                    f"Mage Flow {component_name} expected {expected_count} weights, got {actual_count}",
                )

    @staticmethod
    def is_folded_vae_weights(component_weights: dict) -> bool:
        keys = [key for key, _ in tree_flatten(component_weights)]
        dico_keys = [key for key in keys if key.startswith(("encoder.blocks.", "decoder_model.blocks."))]
        folded = [key for key in dico_keys if key.endswith(".adaLN_modulation.modulation")]
        unfolded = [key for key in dico_keys if ".adaLN_modulation.1." in key]
        if folded and unfolded:
            raise ValueError("Mage Flow VAE weights mix folded and unfolded AdaLN tensors")
        return bool(folded)

    @staticmethod
    def prepare_vae_for_loading(vae: nn.Module, component_weights: dict) -> bool:
        if not MageFlowWeightDefinition.is_folded_vae_weights(component_weights):
            return False

        folded_keys = [
            key
            for key, _ in tree_flatten(component_weights)
            if key.startswith(("encoder.blocks.", "decoder_model.blocks."))
            and key.endswith(".adaLN_modulation.modulation")
        ]
        expected_blocks = len(vae.encoder.blocks) + len(vae.decoder_model.blocks)
        if len(folded_keys) != expected_blocks:
            raise ValueError(
                f"Folded Mage Flow VAE expected {expected_blocks} modulation tensors, got {len(folded_keys)}",
            )

        replaced = vae.freeze_adaln_cache()
        if replaced not in (0, expected_blocks):
            raise ValueError(
                f"Mage Flow VAE folded {replaced} AdaLN blocks, expected {expected_blocks}",
            )
        return True

    @staticmethod
    def finalize_vae_after_loading(vae: nn.Module, component_weights: dict) -> int:
        if MageFlowWeightDefinition.is_folded_vae_weights(component_weights):
            return 0
        return vae.freeze_adaln_cache()
