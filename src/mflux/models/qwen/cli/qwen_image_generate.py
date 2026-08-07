from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


# Single source of truth for CFG-dependent options: main() warns from these and the
# mflux-capabilities dump reads them. Both flags depend on what --model resolves to
# (the default Qwen-Image-2512 does true CFG; Qwen-Image-Flash is DMD2
# guidance-distilled), so they are conditional, not statically ignored.
CONDITIONAL_OPTIONS = {
    "--guidance": {
        "condition": "the resolved model supports classifier-free guidance",
        "reason": "Qwen-Image-Flash internalizes CFG (DMD2 distillation); guidance is forced to 1.0.",
    },
    "--negative-prompt": {
        "condition": "the resolved model supports classifier-free guidance",
        "reason": "the negative pass is skipped on guidance-distilled Qwen models, so the negative prompt has no effect.",
    },
}


def resolve_effective_guidance(guidance: float | None, model_config: ModelConfig) -> float:
    # Guidance-distilled checkpoints run the conditional pass only; applying guidance
    # again degrades output, so the effective value is pinned to 1.0. CFG models keep
    # the historical default when --guidance is omitted.
    if model_config.supports_guidance is False:
        return 1.0
    return ui_defaults.GUIDANCE_SCALE if guidance is None else guidance


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Qwen Image model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_pid_decode_arguments()
    parser.add_output_arguments()
    return parser


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    # 0. Resolve the model configuration (mirrors the Edit CLI: --model may be an alias,
    # a HuggingFace name, or a local path whose config falls back to the default).
    model_config = ModelConfig.qwen_image()
    if args.model is not None:
        try:
            model_config = ModelConfig.from_name(args.model, base_model=args.base_model)
        except ModelConfigError:
            if args.model_path is None:
                raise

    # 0. Warn on the EFFECTIVE behavior and set the default guidance: a model that
    # resolved to a guidance-distilled config cannot honour --guidance/--negative-prompt.
    if model_config.supports_guidance is False:
        CommandLineParser.warn_ignored_options(
            {
                "--guidance": CONDITIONAL_OPTIONS["--guidance"]["reason"],
                "--negative-prompt": CONDITIONAL_OPTIONS["--negative-prompt"]["reason"],
            }
        )
    args.guidance = resolve_effective_guidance(args.guidance, model_config)

    # 1. Load the model
    qwen = QwenImage(
        quantize=args.quantize,
        model_config=model_config,
        model_path=args.model_path,
        **lora_init_kwargs_from_args(args),
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=qwen,
        latent_creator=QwenLatentCreator,
    )

    try:
        # Resolve dimensions (supports ScaleFactor like "2x" when --image-path is provided)
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )

        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = qwen.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=PromptUtil.read_negative_prompt(args),
                width=width,
                height=height,
                guidance=args.guidance,
                scheduler=args.scheduler,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
                pid_decode=args.pid_decode,
                pid_degrade_sigma=args.pid_degrade_sigma,
            )
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
