from argparse import Namespace

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.mage_flow.variants.pipeline_helpers import default_inference_steps
from mflux.utils.exceptions import ModelConfigError


class MageFlowCLIUtil:
    TEXT_TO_IMAGE_ALIASES = {
        "mage-flow-base",
        "mageflow-base",
        "mage-flow",
        "mageflow",
        "mage-flow-turbo",
        "mageflow-turbo",
    }
    EDIT_ALIASES = {
        "mage-flow-edit-base",
        "mageflow-edit-base",
        "mage-flow-edit",
        "mageflow-edit",
        "mage-flow-edit-turbo",
        "mageflow-edit-turbo",
    }

    @staticmethod
    def resolve_model_config(
        parser: CommandLineParser,
        args: Namespace,
        *,
        edit: bool,
    ) -> ModelConfig:
        try:
            model_config = ModelConfig.from_name(model_name=args.model, base_model=args.base_model)
        except ModelConfigError as exc:
            # parser.error() exits the process; re-raise as a safeguard so model_config is
            # always bound on the lines below.
            parser.error(str(exc))
            raise

        compatible_aliases = MageFlowCLIUtil.EDIT_ALIASES if edit else MageFlowCLIUtil.TEXT_TO_IMAGE_ALIASES
        if not compatible_aliases.intersection(model_config.aliases):
            command = "mflux-generate-mage-flow-edit" if edit else "mflux-generate-mage-flow"
            variant = "Edit" if edit else "text-to-image"
            parser.error(f"{command} requires a Mage Flow {variant} model.")
        return model_config

    @staticmethod
    def apply_model_defaults(
        parser: CommandLineParser,
        args: Namespace,
        model_config: ModelConfig,
    ) -> None:
        if args.steps is None:
            args.steps = default_inference_steps(model_config)
        if args.guidance is None:
            args.guidance = 5.0 if model_config.supports_guidance else 1.0
        if not model_config.supports_guidance and args.guidance != 1.0:
            parser.error("Mage Flow Turbo models require --guidance 1.0.")
