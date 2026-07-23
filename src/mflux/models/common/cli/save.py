from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.boogu.variants.txt2img.boogu_image import BooguImage
from mflux.models.common.config import ModelConfig
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4
from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.models.mage_flow import MageFlow, MageFlowEdit
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.qwen_layered.variants.i2l.qwen_image_layered import QwenImageLayered
from mflux.models.z_image import ZImage, ZImageTurbo


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Save a quantized version of a model to disk.")  # fmt: off
    parser.add_model_arguments(path_type="save", require_model_arg=True)
    parser.add_lora_arguments()
    args = parser.parse_args()

    # 1. Resolve the family before dispatch. Saved MFLUX directories can have
    # opaque names and carry their canonical base in mflux_model_config.json.
    model_config = ModelConfig.from_name(args.model, base_model=args.base_model)
    family = " ".join(
        identifier
        for identifier in (
            model_config.base_model or model_config.model_name,
            *model_config.aliases,
        )
        if identifier
    ).lower()
    is_mage_flow = "mage-flow" in family or "mageflow" in family
    if "boogu" in family:
        model_class = BooguImage
    elif "ernie" in family:
        model_class = ErnieImage
    elif is_mage_flow and "edit" in family:
        model_class = MageFlowEdit
    elif is_mage_flow:
        model_class = MageFlow
    elif "qwen" in family and "layered" in family:
        model_class = QwenImageLayered
    elif "qwen" in family and "edit" in family:
        model_class = QwenImageEdit
    elif "qwen" in family:
        model_class = QwenImage
    elif "fibo" in family:
        model_class = FIBO
    elif "z-image-turbo" in family or "zimage-turbo" in family:
        model_class = ZImageTurbo
    elif "z-image" in family or "zimage" in family:
        model_class = ZImage
    elif "flux2" in family or "flux.2" in family:
        model_class = Flux2Klein
    elif "ideogram" in family:
        model_class = Ideogram4
    elif "krea-2" in family or "krea2" in family:
        # "krea-2"/"krea2" only — must not match Flux.1 Krea ("krea-dev"/"dev-krea").
        model_class = Krea2
    else:
        model_class = Flux1

    # 2. Load, quantize and save the model
    if model_class in (MageFlow, MageFlowEdit):
        if args.lora_paths:
            parser.error("LoRA adapters are not currently supported when saving Mage Flow models.")
        model = model_class(
            quantize=args.quantize,
            model_path=args.model_path,
            model_config=model_config,
        )
    elif model_class is Ideogram4:
        model_config = Ideogram4WeightDefinition.resolve_inference_config(
            args.model,
            args.base_model,
            args.model_path,
        )
        model_path = None if Ideogram4WeightDefinition.is_builtin_name(args.model) else args.model_path
        model = model_class(
            quantize=args.quantize,
            lora_paths=args.lora_paths,
            lora_scales=args.lora_scales,
            model_path=model_path,
            model_config=model_config,
        )
    else:
        model = model_class(
            quantize=args.quantize,
            lora_paths=args.lora_paths,
            lora_scales=args.lora_scales,
            model_path=args.model_path,
            model_config=model_config,
        )

    model.save_model(args.path)


if __name__ == "__main__":
    main()
