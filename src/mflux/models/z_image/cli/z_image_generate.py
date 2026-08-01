import sys
import warnings

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config import ModelConfig
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil
from mflux.utils.saveinfo_util import build_saveinfo_filename


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Z-Image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments()
    parser.add_pid_decode_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    if "--scheduler" not in sys.argv:
        args.scheduler = "flow_match_euler_discrete"

    model_name = args.model or "z-image"
    model_config = ModelConfig.from_name(model_name=model_name)

    # Warn on the EFFECTIVE behavior, not the flag value: --model may resolve to a
    # guidance-distilled variant (guidance forced to 0.0 regardless of the flag), and on
    # CFG-capable models the negative prompt is only encoded at guidance > 1.0, where an
    # omitted --guidance defaults to 0.0.
    if not model_config.supports_guidance:
        CommandLineParser.warn_ignored_options(
            {
                "--guidance": f"{model_name} is guidance-distilled; guidance is forced to 0.0.",
                "--negative-prompt": f"CFG is disabled on {model_name}, so the negative prompt is never encoded.",
            }
        )
    elif CommandLineParser._option_was_provided("--negative-prompt") and (args.guidance is None or args.guidance <= 1.0):
        warnings.warn(
            "--negative-prompt has no effect: CFG is disabled at guidance <= 1.0, and the default guidance is 0.0."
            " Pass --guidance above 1.0 to enable it.",
            stacklevel=1,
        )

    # 1. Load the model
    model = ZImage(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        **lora_init_kwargs_from_args(args),
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ZImageLatentCreator,
    )

    try:
        # Resolve dimensions
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )

        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
                guidance=args.guidance,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
                scheduler=args.scheduler,
                negative_prompt=args.negative_prompt,
                shift=args.shift,
                mcf_max_change=args.mcf_max_change,
                sigma_schedule=args.sigma_schedule,
                pid_decode=args.pid_decode,
                pid_degrade_sigma=args.pid_degrade_sigma,
            )
            # 4. Save the image
            output_path = build_saveinfo_filename(args, seed) if args.saveinfo else args.output.format(seed=seed)
            image.save(path=output_path, export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
