from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.mage_flow.cli.util import MageFlowCLIUtil
from mflux.models.mage_flow.latent_creator import MageFlowLatentCreator
from mflux.models.mage_flow.variants.txt2img.mage_flow import MageFlow
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Microsoft Mage Flow.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.set_defaults(model="mage-flow")
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_mage_flow_arguments()
    parser.add_output_arguments()
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model_config = MageFlowCLIUtil.resolve_model_config(parser, args, edit=False)
    MageFlowCLIUtil.apply_model_defaults(parser, args, model_config)

    model = MageFlow(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
    )
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=MageFlowLatentCreator,
    )

    try:
        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                guidance=args.guidance,
                negative_prompt=PromptUtil.read_negative_prompt(args) or None,
                renormalization=args.renormalization,
                gaussian_shading_key=args.gaussian_shading_key,
                scheduler=args.scheduler,
            )
            image.save(path=args.output.format(seed=image.seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError, ValueError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
