from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.mage_flow.model.mage_flow_text_encoder import MageFlowTextEncoder
from mflux.models.mage_flow.model.mage_flow_transformer import MageFlowTransformer
from mflux.models.mage_flow.model.mage_flow_vae import MageVAE
from mflux.models.mage_flow.weights import MageFlowWeightDefinition


class MageFlowInitializer:
    """Load a released Mage-Flow repository into native MLX modules."""

    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
    ) -> None:
        path = model_path or model_config.model_name
        root_path = MageFlowInitializer._resolve_model_path(path)
        MageFlowInitializer._init_config(model, model_config)
        weights = MageFlowInitializer._load_weights(root_path)
        MageFlowWeightDefinition.validate_loaded_weights(weights)
        MageFlowInitializer._init_tokenizers(model, root_path)
        MageFlowInitializer._init_models(model, model_config)
        MageFlowInitializer._validate_hf_model_coverage(model, weights)
        MageFlowInitializer._apply_weights(model, weights, quantize)

        del weights
        mx.eval(model)
        mx.clear_cache()

    @staticmethod
    def _resolve_model_path(path: str) -> Path:
        # Resolve once so weights and tokenizer share the same HF snapshot revision.
        root_path = PathResolution.resolve(
            path=path,
            patterns=MageFlowWeightDefinition.get_download_patterns(),
            required_pattern_groups=MageFlowWeightDefinition.get_required_download_pattern_groups(),
        )
        if root_path is None:
            raise ValueError(f"No model path resolved for {path!r}")
        return root_path

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.policy_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths = None
        model.lora_scales = None

    @staticmethod
    def _load_weights(model_path: Path) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=MageFlowWeightDefinition,
            model_path=str(model_path),
        )

    @staticmethod
    def _init_tokenizers(model, model_path: Path) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=MageFlowWeightDefinition.get_tokenizers(),
            model_path=str(model_path),
        )

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        model.vae = MageVAE(sample_posterior=True)
        model.transformer = MageFlowTransformer(**model_config.transformer_overrides)
        model.text_encoder = MageFlowTextEncoder(**model_config.text_encoder_overrides)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        vae_weights = weights.components["vae"]
        MageFlowWeightDefinition.prepare_vae_for_loading(model.vae, vae_weights)
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=MageFlowWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )
        MageFlowWeightDefinition.finalize_vae_after_loading(model.vae, vae_weights)

    @staticmethod
    def _validate_hf_model_coverage(model, weights: LoadedWeights) -> None:
        """Fail before allocation if a released checkpoint no longer matches the port."""

        if weights.meta_data.mflux_version is not None:
            return

        for component_name in ("vae", "transformer", "text_encoder"):
            source = dict(tree_flatten(weights.components[component_name]))
            target_model = getattr(model, component_name)
            target = dict(tree_flatten(target_model.parameters()))
            missing = sorted(target.keys() - source.keys())
            unexpected = sorted(source.keys() - target.keys())
            shape_mismatches = sorted(
                key for key in source.keys() & target.keys() if source[key].shape != target[key].shape
            )
            if missing or unexpected or shape_mismatches:
                raise ValueError(
                    f"Mage Flow {component_name} checkpoint does not match the MLX module: "
                    f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
                    f"shape_mismatches={shape_mismatches[:5]}"
                )
