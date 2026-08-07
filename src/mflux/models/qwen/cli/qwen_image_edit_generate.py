from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Qwen Image Edit with image conditioning.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_argument("--image-paths", type=Path, nargs="+", required=True, help="Local paths to one or more init images. For single image editing, provide one path. For multiple image editing, provide multiple paths.")  # fmt: off
    parser.add_output_arguments()
    return parser


# Single source of truth for options a guidance-distilled model cannot honour here:
# the runtime warning and the mflux-capabilities dump both read it.
CONDITIONAL_OPTIONS = {
    "--guidance": {
        "condition": "the resolved model supports guidance (distilled checkpoints internalize CFG)",
        "reason": "guidance-distilled checkpoints internalize CFG; applying guidance again degrades the output.",
    },
    "--negative-prompt": {
        "condition": "the resolved model supports guidance",
        "reason": "CFG is disabled on guidance-distilled checkpoints, so the negative prompt is never applied.",
    },
}


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    # 1. Load the model
    model_config = ModelConfig.qwen_image_edit()
    if args.model is not None:
        try:
            model_config = ModelConfig.from_name(args.model, base_model=args.base_model)
        except ModelConfigError:
            if args.model_path is None:
                raise

    # Warn on the EFFECTIVE behavior and set the guidance: a model that resolved to a
    # guidance-distilled config (Qwen-Image-Flash through this CLI) must run at 1.0,
    # which also skips the per-step negative pass in the variant.
    if model_config.supports_guidance is False:
        CommandLineParser.warn_ignored_options(
            {
                "--guidance": CONDITIONAL_OPTIONS["--guidance"]["reason"],
                "--negative-prompt": CONDITIONAL_OPTIONS["--negative-prompt"]["reason"],
            }
        )
        args.guidance = 1.0
    elif args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE_KONTEXT

    qwen = QwenImageEdit(
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
        # 3. Prepare image paths and resolve dimensions against the source image by default.
        image_paths = [str(p) for p in args.image_paths]
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=image_paths[0],
        )

        for seed in args.seed:
            # 4. Generate an image for each seed value
            image = qwen.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=PromptUtil.read_negative_prompt(args),
                width=width,
                height=height,
                guidance=args.guidance,
                image_path=image_paths[0],  # Use first image for metadata
                image_paths=image_paths,
                num_inference_steps=args.steps,
                scheduler=args.scheduler,
            )

            # 5. Save the image
            output_path = Path(args.output.format(seed=seed))
            image.save(path=output_path, export_json_metadata=args.metadata)

    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
