import mlx.core as mx
from mlx import nn

import mflux.models.krea2.model.krea2_scheduler  # noqa: F401 — register er_sde/euler schedulers
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE


class Krea2Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        uncensor: float = 1.0,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        Krea2Initializer._init_config(model, model_config)
        weights = Krea2Initializer._load_weights(path, model_config)
        Krea2Initializer._init_tokenizers(model, path)
        Krea2Initializer._init_models(model, model_config)
        Krea2Initializer._apply_weights(model, weights, quantize)
        Krea2Initializer._apply_lora(model, lora_paths, lora_scales)
        Krea2Initializer._apply_uncensor(model, uncensor)
        del weights
        mx.eval(model)
        mx.clear_cache()

    # Order of the widened input projection's channel targets: the reference DiT concatenates the
    # attention/MLP LoRA over blocks 0..27, each with these five attention + three MLP sub-layers.
    _CONTROL_LORA_TARGETS = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate", "mlp.gate", "mlp.up", "mlp.down")

    @staticmethod
    def init_depth(
        model,
        model_config: ModelConfig,
        controlnet_path: str,
        quantize: int | None,
        controlnet_strength: float = 1.0,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        uncensor: float = 1.0,
    ) -> None:
        # The depth checkpoint stores a full 128-wide `first` weight plus attention/MLP deltas as raw
        # A/B matrices (not diffusers/kohya LoRA naming), so it is merged directly into the base weights
        # rather than through the standard LoRA loader. The base is therefore loaded unquantized; when
        # quantization is requested the deltas are baked in first and the model is quantized afterwards,
        # so the packed weights already include the control.
        path = model_path if model_path else model_config.model_name
        Krea2Initializer._init_config(model, model_config)
        weights = Krea2Initializer._load_weights(path, model_config)
        Krea2Initializer._init_tokenizers(model, path)
        Krea2Initializer._init_models(model, model_config)
        Krea2Initializer._apply_weights(model, weights, quantize=None)
        Krea2Initializer._apply_control_checkpoint(model, controlnet_path, controlnet_strength)
        Krea2Initializer._apply_lora(model, lora_paths, lora_scales)
        Krea2Initializer._apply_uncensor(model, uncensor)
        Krea2Initializer._quantize_after_control(model, quantize)
        del weights
        mx.eval(model)
        mx.clear_cache()

    @staticmethod
    def _quantize_after_control(model, quantize: int | None) -> None:
        # Quantize once the control deltas are merged in, mirroring the base path's post-load quantize
        # (same predicate + group size + skip_quantization components, e.g. the text encoder).
        if quantize is None:
            model.bits = None
            return
        components = {c.name: c for c in Krea2WeightDefinition.get_components()}
        WeightApplier._quantize(
            models={"vae": model.vae, "transformer": model.transformer, "text_encoder": model.text_encoder},
            bits=quantize,
            components=components,
            weight_definition=Krea2WeightDefinition,
        )
        model.bits = quantize

    @staticmethod
    def _apply_control_checkpoint(model, controlnet_path: str, controlnet_strength: float) -> None:
        control = mx.load(str(controlnet_path))
        transformer = model.transformer

        # 1. Replace the input projection with the widened (2*c*p*p -> features) version and load its
        #    full trained weight/bias from the checkpoint.
        first_w = control.get("first.weight")
        first_b = control.get("first.bias")
        if first_w is None or first_b is None:
            raise ValueError(f"Control checkpoint {controlnet_path} is missing 'first.weight'/'first.bias'.")

        # Validate every tensor shape up front so a malformed checkpoint fails cleanly here instead of
        # installing a bad projection or half-updating the blocks and only breaking during inference.
        expected_out = transformer.first.weight.shape[0]
        expected_in = 2 * transformer.channels * transformer.patch**2
        if tuple(first_w.shape) != (expected_out, expected_in):
            raise ValueError(f"Control 'first.weight' is {tuple(first_w.shape)}, expected {(expected_out, expected_in)}.")
        if tuple(first_b.shape) != (expected_out,):
            raise ValueError(f"Control 'first.bias' is {tuple(first_b.shape)}, expected {(expected_out,)}.")
        deltas: dict[tuple[int, str], mx.array] = {}
        for i, block in enumerate(transformer.blocks):
            for target in Krea2Initializer._CONTROL_LORA_TARGETS:
                a = control.get(f"blocks.{i}.{target}.A")
                b = control.get(f"blocks.{i}.{target}.B")
                if a is None or b is None:
                    continue
                sub, attr = target.split(".")  # e.g. "attn", "wq"
                weight_shape = tuple(getattr(getattr(block, sub), attr).weight.shape)
                if a.shape[0] != b.shape[1] or (b.shape[0], a.shape[1]) != weight_shape:
                    raise ValueError(f"Control delta blocks.{i}.{target} (B{tuple(b.shape)} @ A{tuple(a.shape)}) does not match weight {weight_shape}.")  # noqa: E501
                deltas[(i, target)] = b @ a

        # 1. Replace the input projection with the widened (2*c*p*p -> features) version.
        widened = nn.Linear(expected_in, expected_out, bias=True)
        widened.weight = first_w.astype(transformer.first.weight.dtype)
        widened.bias = first_b.astype(transformer.first.weight.dtype)
        transformer.first = widened

        # 2. Merge each attention/MLP delta (B @ A, scale = strength) into the base weight.
        for (i, target), delta in deltas.items():
            sub, attr = target.split(".")
            layer = getattr(getattr(transformer.blocks[i], sub), attr)
            layer.weight = layer.weight + controlnet_strength * delta.astype(layer.weight.dtype)
        mx.eval(transformer)

    @staticmethod
    def _apply_uncensor(model, uncensor: float) -> None:
        # Krea 2's content refusal is carried by the text-fusion projector's weighting of the
        # tapped Qwen3-VL layers 9/10/11 (indices 8, 9, 10; default weights ~ -0.51/-0.89/-0.61).
        # Scaling those (the same direction the community "FilterBypass" LoRA and skc3vo amplify)
        # neutralises the refusal without touching anything else. uncensor=1.0 leaves the model
        # untouched; ~6.0 flips refused explicit prompts to coherent output (empirically validated).
        if uncensor == 1.0:
            return
        proj = getattr(getattr(model.transformer, "txtfusion", None), "projector", None)
        if proj is None or tuple(proj.weight.shape) != (1, 12):
            return  # projector missing or reshaped; skip rather than corrupt conditioning
        w = proj.weight
        new = mx.array(w)
        for i in (8, 9, 10):
            new[0, i] = w[0, i] * uncensor
        proj.weight = new
        mx.eval(proj.weight)

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Krea2WeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths = None
        model.lora_scales = None

    @staticmethod
    def _load_weights(model_path: str, model_config: ModelConfig) -> LoadedWeights:
        # Variant-aware HF download: Raw needs the diffusers transformer/ dir, Turbo the single file.
        return WeightLoader.load(
            weight_definition=Krea2WeightDefinition,
            model_path=model_path,
            download_patterns=Krea2WeightDefinition.get_download_patterns(model_config.model_name),
        )

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        model.vae = QwenVAE()
        model.transformer = Krea2Transformer(**(model_config.transformer_overrides or {}))
        model.text_encoder = Krea2TextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Krea2WeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
