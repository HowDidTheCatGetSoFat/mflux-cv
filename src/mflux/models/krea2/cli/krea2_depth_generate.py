from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.krea2.latent_creator import Krea2LatentCreator
from mflux.models.krea2.variants.controlnet.krea2_depth import Krea2Depth
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# Krea-2 turbo defaults (8 steps, CFG 1.0, er_sde). Raw base wants ~28-52 steps, CFG 3.5.
DEFAULT_STEPS = 8
DEFAULT_GUIDANCE = 1.0


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Krea-2 with a depth ControlNet.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False, supports_dimension_scale_factor=True)
    parser.add_depth_arguments()
    parser.add_output_arguments()
    parser.add_argument(
        "--controlnet-path",
        type=str,
        required=True,
        help="Local path to the Krea 2 depth-control checkpoint (widened 'first' + attention/MLP deltas).",
    )
    parser.add_argument(
        "--controlnet-strength",
        type=float,
        default=1.0,
        help="Scale applied to the control deltas when merging them into the base weights (default 1.0).",
    )
    parser.add_argument(
        "--krea2-uncensor",
        type=float,
        default=1.0,
        help="Scale the text-fusion projector's refusal layers (tapped Qwen3-VL 9/10/11). "
        "1.0 = default; ~6.0 neutralises Krea 2's content refusal. Applied once at load.",
    )
    args = parser.parse_args()

    # 1. Load the model
    model = Krea2Depth(
        model_config=ModelConfig.krea2(),
        controlnet_path=args.controlnet_path,
        controlnet_strength=args.controlnet_strength,
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
        uncensor=args.krea2_uncensor,
    )

    # 2. Register callbacks (stepwise image output, memory stats, battery saver)
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Krea2LatentCreator,
        enable_depth_saver=True,
    )

    try:
        steps = args.steps if args.steps is not None else DEFAULT_STEPS
        guidance = args.guidance if args.guidance is not None else DEFAULT_GUIDANCE
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path or args.depth_image_path,
        )
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                scheduler=args.scheduler,
                negative_prompt=args.negative_prompt,
                image_path=args.image_path,
                depth_image_path=args.depth_image_path,
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
