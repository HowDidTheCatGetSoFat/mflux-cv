import random
from collections.abc import Sequence

import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.mage_flow.model.mage_flow_transformer import MageFlowTransformer
from mflux.utils.apple_silicon import AppleSiliconUtil

ImageShape = tuple[int, int, int]

_DEFAULT_STEPS = {
    "microsoft/Mage-Flow-Base": 30,
    "microsoft/Mage-Flow": 20,
    "microsoft/Mage-Flow-Turbo": 4,
    "microsoft/Mage-Flow-Edit-Base": 30,
    "microsoft/Mage-Flow-Edit": 30,
    "microsoft/Mage-Flow-Edit-Turbo": 4,
}


def default_inference_steps(model_config: ModelConfig) -> int:
    for checkpoint in (model_config.model_name, model_config.base_model):
        if checkpoint in _DEFAULT_STEPS:
            return _DEFAULT_STEPS[checkpoint]
    raise ValueError(f"Unknown Mage Flow checkpoint: {model_config.model_name}")


def default_guidance(model_config: ModelConfig) -> float:
    return 5.0 if model_config.supports_guidance else 1.0


def resolve_generation_parameters(
    *,
    model_config: ModelConfig,
    num_inference_steps: int | None,
    guidance: float | None,
) -> tuple[int, float]:
    steps = default_inference_steps(model_config) if num_inference_steps is None else num_inference_steps
    resolved_guidance = default_guidance(model_config) if guidance is None else float(guidance)
    if steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if resolved_guidance < 0:
        raise ValueError("guidance must be non-negative")
    if model_config.supports_guidance is False and resolved_guidance != 1.0:
        raise ValueError("Mage Flow Turbo checkpoints require guidance=1.0")
    return steps, resolved_guidance


def normalize_image_dimension(size: int) -> int:
    """Match the released Mage Flow pipeline's minimum and /16 floor."""

    return max(16, 16 * (size // 16))


def resolve_seed(seed: int) -> int:
    """Resolve Mage Flow's sentinel seed exactly once per sample."""

    return random.randint(0, 2**32 - 1) if seed == -1 else seed


def make_velocity_predictor(
    *,
    transformer: MageFlowTransformer,
    text_embeddings: mx.array,
    text_attention_mask: mx.array,
    image_shapes: Sequence[ImageShape],
    guidance: float,
    target_length: int | None = None,
    renormalization: bool = False,
    compile_model: bool = True,
):
    image_rotary_emb = transformer.pos_embed(image_shapes)
    mx.eval(*image_rotary_emb)

    def predict(image_tokens: mx.array, sigma: mx.array) -> mx.array:
        velocity = transformer(
            img=image_tokens,
            txt=text_embeddings,
            timesteps=sigma,
            img_shapes=image_shapes,
            text_attention_mask=text_attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        if target_length is not None:
            velocity = velocity[:, :target_length]
        if text_embeddings.shape[0] == 1:
            return velocity
        if text_embeddings.shape[0] != 2:
            raise ValueError("single-image Mage Flow inference supports one prompt or one CFG pair")

        unconditional, conditional = velocity[:1], velocity[1:2]
        guided = unconditional + guidance * (conditional - unconditional)
        if not renormalization:
            return guided

        conditional_norm = mx.sqrt(mx.sum(mx.square(conditional.astype(mx.float32)), axis=-1, keepdims=True))
        guided_norm = mx.sqrt(mx.sum(mx.square(guided.astype(mx.float32)), axis=-1, keepdims=True))
        return (guided.astype(mx.float32) * conditional_norm / (guided_norm + 1e-6)).astype(guided.dtype)

    if compile_model and not AppleSiliconUtil.is_m1_or_m2():
        return mx.compile(predict)
    return predict
