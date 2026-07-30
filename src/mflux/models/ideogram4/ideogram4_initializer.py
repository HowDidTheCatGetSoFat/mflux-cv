import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path, tree_unflatten

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.ideogram4.model.ideogram4_text_encoder import Qwen3TextEncoder
from mflux.models.ideogram4.model.ideogram4_transformer import Ideogram4Config, Ideogram4Transformer
from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear
from mflux.models.ideogram4.weights import Ideogram4LoRAMapping, Ideogram4WeightDefinition


class Ideogram4Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        root_path = Ideogram4Initializer._resolve_model_path(path)
        Ideogram4Initializer._init_config(model, model_config, root_path)
        Ideogram4Initializer._init_tokenizers(model, root_path)
        if Ideogram4Initializer._is_mlx_forge(root_path):
            Ideogram4Initializer._init_models_mlx_forge(model, root_path)
            bits = Ideogram4Initializer._mlx_forge_bits(root_path)
            Ideogram4Initializer._apply_mlx_forge_weights(model, root_path, bits)
            model.bits = bits
        else:
            weights = Ideogram4Initializer._load_weights(root_path)
            Ideogram4Initializer._init_models(model, root_path)
            Ideogram4Initializer._apply_weights(model, weights, quantize)
            del weights
        mx.eval(model)
        mx.clear_cache()
        Ideogram4Initializer._apply_lora(model, lora_paths, lora_scales)

    @staticmethod
    def _resolve_model_path(path: str) -> Path:
        # Fast-path: local mlx-forge directory (detected by split_model.json).
        local = Path(path).expanduser()
        if local.exists() and (local / "split_model.json").exists():
            return local
        root_path = PathResolution.resolve(
            path=path,
            patterns=Ideogram4WeightDefinition.get_download_patterns(),
        )
        if root_path is None:
            raise ValueError(f"No model path resolved for {path!r}")
        # mlx-forge repos (resolved from an HF id or a local path) carry split_model.json;
        # load them directly instead of demanding the FP8 checkpoint layout.
        if (root_path / "split_model.json").exists():
            return root_path
        return Ideogram4WeightDefinition.validate_fp8_checkpoint(root_path)

    @staticmethod
    def _is_mlx_forge(root_path: Path) -> bool:
        return (root_path / "split_model.json").exists()

    @staticmethod
    def _mlx_forge_bits(root_path: Path) -> int | None:
        with open(root_path / "split_model.json") as f:
            info = json.load(f)
        if not info.get("quantized", False):
            return None
        return info.get("quantization_bits")

    @staticmethod
    def _init_config(model, model_config: ModelConfig, model_path: Path) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.model_path = model_path
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths = None
        model.lora_scales = None
        # A native save bakes any active LoRA into the conditional transformer and strips the
        # wrappers, so nothing in the weights marks the model as a LoRA model; the saved config
        # records it. CFG-negative routing (through the conditional transformer when a LoRA is
        # active) must survive the save/load round-trip, or the clean unconditional negative
        # amplifies the baked LoRA delta at full guidance.
        saved_config = Ideogram4Initializer._load_json(model_path / ConfigResolution.SAVED_CONFIG_FILENAME)
        model.baked_lora = saved_config.get("baked_lora") or None

    @staticmethod
    def _load_weights(model_path: Path) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=Ideogram4WeightDefinition,
            model_path=str(model_path),
        )

    @staticmethod
    def _init_tokenizers(model, model_path: Path) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Ideogram4WeightDefinition.get_tokenizers(),
            model_path=str(model_path),
        )

    @staticmethod
    def _init_models(model, model_path: Path) -> None:
        model.vae = Flux2VAE()
        model.conditional_transformer = Ideogram4Transformer(
            Ideogram4Initializer._transformer_config(model_path / "transformer" / "config.json")
        )
        model.unconditional_transformer = Ideogram4Transformer(
            Ideogram4Initializer._transformer_config(model_path / "unconditional_transformer" / "config.json")
        )
        model.text_encoder = Qwen3TextEncoder(
            **Ideogram4Initializer._text_encoder_kwargs(model_path / "text_encoder" / "config.json")
        )

    @staticmethod
    def _init_models_mlx_forge(model, root_path: Path) -> None:
        model.vae = Flux2VAE()
        model.conditional_transformer = Ideogram4Transformer(
            Ideogram4Initializer._transformer_config(root_path / "conditional_transformer_config.json")
        )
        model.unconditional_transformer = Ideogram4Transformer(
            Ideogram4Initializer._transformer_config(root_path / "unconditional_transformer_config.json")
        )
        model.text_encoder = Qwen3TextEncoder(
            **Ideogram4Initializer._text_encoder_kwargs(root_path / "text_encoder_config.json")
        )

    @staticmethod
    def _apply_mlx_forge_weights(model, root_path: Path, bits: int | None) -> None:
        components = {
            "conditional_transformer": model.conditional_transformer,
            "unconditional_transformer": model.unconditional_transformer,
            "text_encoder": model.text_encoder,
            "vae": model.vae,
        }
        for comp_name, comp_model in components.items():
            raw = mx.load(str(root_path / f"{comp_name}.safetensors"))
            prefix = f"{comp_name}."
            stripped = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
            if bits is not None and comp_name != "vae":
                Ideogram4Initializer._replace_fp8_with_quantized(comp_model, bits)
            comp_model.update(tree_unflatten(list(stripped.items())), strict=False)
            del raw, stripped

    @staticmethod
    def _replace_fp8_with_quantized(model: nn.Module, bits: int, group_size: int = 64) -> None:
        """Swap every Fp8Linear submodule for an empty nn.QuantizedLinear.

        Mirrors mlx.nn.quantize's leaf_modules → tree_map_with_path →
        update_modules idiom. The QuantizedLinear modules are zero-initialized
        here; their packed weight/scales/biases are populated by the subsequent
        model.update() against the mlx-forge int8 safetensors.
        """

        def maybe_replace(_path: str, module: nn.Module) -> nn.Module:
            if isinstance(module, Fp8Linear):
                return nn.QuantizedLinear(
                    input_dims=module.in_features,
                    output_dims=module.out_features,
                    bias=module.bias is not None,
                    bits=bits,
                    group_size=group_size,
                )
            return module

        leaves = model.leaf_modules()
        leaves = tree_map_with_path(maybe_replace, leaves, is_leaf=nn.Module.is_module)
        model.update_modules(leaves)

    @staticmethod
    def _rebuild_q8_folded_layers(module, tree) -> None:
        """A native save can hold layers folded to MLX q8: baking a LoRA over an fp8 base
        (LoRASaver.bake_and_strip_lora) dequantizes and requantizes the merged weight to
        q8, so the checkpoint stores a packed 'weight' plus 'scales'/'biases'. Fp8Linear
        cannot hold those tensors and the native-checkpoint validator rejects the load
        (missing 'weight_scale', unexpected 'scales'/'biases'). Rebuild any such layer as
        QuantizedLinear before validation, so mixed fp8/q8 checkpoints load. Original fp8
        checkpoints carry 'weight_scale' instead of 'scales'/'biases' and are left
        untouched."""
        if isinstance(tree, list):
            children = list(module) if hasattr(module, "__iter__") else []
            for idx, sub in enumerate(tree):
                if idx < len(children):
                    Ideogram4Initializer._rebuild_q8_folded_layers(children[idx], sub)
            return
        if not isinstance(tree, dict):
            return
        for key, sub in tree.items():
            if not isinstance(sub, (dict, list)):
                continue
            child = getattr(module, key, None)
            if child is None and isinstance(module, dict):
                child = module.get(key)
            if child is None:
                continue
            if (
                isinstance(sub, dict)
                and "scales" in sub
                and "biases" in sub
                and "weight" in sub
                and not isinstance(child, nn.QuantizedLinear)
            ):
                scales = sub["scales"]
                output_dims = scales.shape[0]
                input_dims = scales.shape[1] * 64
                replacement = nn.QuantizedLinear(
                    input_dims, output_dims, bias="bias" in sub, group_size=64, bits=8
                )
                if isinstance(module, dict):
                    module[key] = replacement
                else:
                    setattr(module, key, replacement)
            else:
                Ideogram4Initializer._rebuild_q8_folded_layers(child, sub)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        for name in ("conditional_transformer", "unconditional_transformer"):
            tree = weights.components.get(name)
            if tree:
                Ideogram4Initializer._rebuild_q8_folded_layers(getattr(model, name), tree)
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Ideogram4WeightDefinition,
            models={
                "vae": model.vae,
                "conditional_transformer": model.conditional_transformer,
                "unconditional_transformer": model.unconditional_transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        # Apply ONLY to the conditional transformer. The unconditional transformer is a
        # DIFFERENT (distilled) model: a LoRA trained against the conditional's weights pushes
        # it off-distribution, and CFG then amplifies that corrupted negative by the guidance
        # factor (the patch-mosaic bug). With a LoRA active, generation instead routes the CFG
        # negative through the (LoRA'd) conditional transformer.
        lora_mapping = Ideogram4LoRAMapping.get_mapping()
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=lora_mapping,
            transformer=model.conditional_transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @staticmethod
    def _text_encoder_kwargs(config_file: Path) -> dict[str, Any]:
        config = Ideogram4Initializer._load_json(config_file)
        text_config = config.get("text_config") if isinstance(config, dict) else None
        if not isinstance(text_config, dict):
            text_config = {}
        rope_parameters = text_config.get("rope_parameters")
        if not isinstance(rope_parameters, dict):
            rope_parameters = {}
        return {
            "vocab_size": int(text_config.get("vocab_size", 151936)),
            "hidden_size": int(text_config.get("hidden_size", 4096)),
            "num_hidden_layers": int(text_config.get("num_hidden_layers", 36)),
            "num_attention_heads": int(text_config.get("num_attention_heads", 32)),
            "num_key_value_heads": int(text_config.get("num_key_value_heads", 8)),
            "intermediate_size": int(text_config.get("intermediate_size", 12288)),
            "max_position_embeddings": int(text_config.get("max_position_embeddings", 262144)),
            "rope_theta": float(rope_parameters.get("rope_theta", text_config.get("rope_theta", 5_000_000.0))),
            "rms_norm_eps": float(text_config.get("rms_norm_eps", 1e-6)),
            "head_dim": int(text_config.get("head_dim", 128)),
        }

    @staticmethod
    def _transformer_config(config_file: Path) -> Ideogram4Config:
        config = Ideogram4Initializer._load_json(config_file)
        num_heads = int(config.get("num_attention_heads", 18))
        head_dim = int(config.get("attention_head_dim", 256))
        return Ideogram4Config(
            emb_dim=num_heads * head_dim,
            num_layers=int(config.get("num_layers", 34)),
            num_heads=num_heads,
            intermediate_size=int(config.get("intermediate_size", 12288)),
            adanln_dim=int(config.get("adaln_dim", 512)),
            in_channels=int(config.get("in_channels", 128)),
            llm_features_dim=int(config.get("llm_features_dim", 53248)),
            rope_theta=int(config.get("rope_theta", 5_000_000)),
            mrope_section=tuple(config.get("mrope_section", (24, 20, 20))),
            norm_eps=float(config.get("norm_eps", 1e-5)),
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
