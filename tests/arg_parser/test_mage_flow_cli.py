import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.mage_flow.cli.util import MageFlowCLIUtil


def _text_to_image_parser() -> CommandLineParser:
    parser = CommandLineParser()
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.set_defaults(model="mage-flow")
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_mage_flow_arguments()
    parser.add_output_arguments()
    return parser


def _edit_parser() -> CommandLineParser:
    parser = CommandLineParser()
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.set_defaults(model="mage-flow-edit")
    parser.add_image_generator_arguments(
        supports_metadata_config=True,
        dimensions_default_to_none=True,
    )
    parser.add_mage_flow_edit_arguments()
    parser.add_output_arguments()
    return parser


@pytest.mark.fast
@pytest.mark.parametrize(
    ("model_name", "expected_steps"),
    [
        ("mage-flow-base", 30),
        ("mage-flow", 20),
        ("mage-flow-turbo", 4),
        ("mage-flow-edit-base", 30),
        ("mage-flow-edit", 30),
        ("mage-flow-edit-turbo", 4),
        ("mageflow-base", 30),
        ("mageflow", 20),
        ("mageflow-turbo", 4),
        ("mageflow-edit-base", 30),
        ("mageflow-edit", 30),
        ("mageflow-edit-turbo", 4),
    ],
)
def test_mage_flow_model_defaults(model_name: str, expected_steps: int) -> None:
    assert model_name in ui_defaults.MODEL_CHOICES
    assert ui_defaults.MODEL_INFERENCE_STEPS[model_name] == expected_steps


@pytest.mark.fast
@pytest.mark.parametrize(
    ("model_name", "expected_steps"),
    [
        ("microsoft/Mage-Flow-Base", 30),
        ("microsoft/Mage-Flow", 20),
        ("microsoft/Mage-Flow-Turbo", 4),
        ("microsoft/Mage-Flow-Edit-Base", 30),
        ("microsoft/Mage-Flow-Edit", 30),
        ("microsoft/Mage-Flow-Edit-Turbo", 4),
    ],
)
def test_mage_flow_checkpoint_step_defaults(model_name: str, expected_steps: int) -> None:
    assert ui_defaults.MODEL_INFERENCE_STEPS[model_name] == expected_steps


@pytest.mark.fast
def test_mage_flow_text_to_image_defaults() -> None:
    parser = _text_to_image_parser()
    with patch("sys.argv", ["mflux-generate-mage-flow", "--prompt", "a silver fox"]):
        args = parser.parse_args()

    model_config = MageFlowCLIUtil.resolve_model_config(parser, args, edit=False)
    MageFlowCLIUtil.apply_model_defaults(parser, args, model_config)

    assert args.model == "mage-flow"
    assert args.steps == 20
    assert args.guidance == pytest.approx(5.0)
    assert args.scheduler == "mage_flow"
    assert args.renormalization is False
    assert args.gaussian_shading_key is None
    assert args.width == 1024
    assert args.height == 1024


@pytest.mark.fast
def test_mage_flow_turbo_defaults_and_flags() -> None:
    parser = _text_to_image_parser()
    with patch(
        "sys.argv",
        [
            "mflux-generate-mage-flow",
            "--model",
            "mageflow-turbo",
            "--prompt",
            "a silver fox",
            "--renormalization",
            "--gaussian-shading-key",
            "private-key",
        ],
    ):
        args = parser.parse_args()

    model_config = MageFlowCLIUtil.resolve_model_config(parser, args, edit=False)
    MageFlowCLIUtil.apply_model_defaults(parser, args, model_config)

    assert args.steps == 4
    assert args.guidance == pytest.approx(1.0)
    assert args.renormalization is True
    assert args.gaussian_shading_key == "private-key"


@pytest.mark.fast
@pytest.mark.parametrize(
    ("model_args", "expected_steps"),
    [
        (["--model", "acme/custom", "--base-model", "mage-flow-turbo"], 4),
        (["--model", "mage-flow-turbo-4bit"], 4),
        (["--model", "acme/custom", "--base-model", "mage-flow-base"], 30),
    ],
)
def test_mage_flow_custom_models_inherit_base_step_default(model_args: list[str], expected_steps: int) -> None:
    parser = _text_to_image_parser()
    with patch(
        "sys.argv",
        ["mflux-generate-mage-flow", *model_args, "--prompt", "a silver fox"],
    ):
        args = parser.parse_args()

    assert args.steps is None
    model_config = MageFlowCLIUtil.resolve_model_config(parser, args, edit=False)
    MageFlowCLIUtil.apply_model_defaults(parser, args, model_config)

    assert args.steps == expected_steps


@pytest.mark.fast
def test_mage_flow_custom_model_preserves_explicit_steps() -> None:
    parser = _text_to_image_parser()
    with patch(
        "sys.argv",
        [
            "mflux-generate-mage-flow",
            "--model",
            "acme/custom",
            "--base-model",
            "mage-flow-turbo",
            "--prompt",
            "a silver fox",
            "--steps",
            "9",
        ],
    ):
        args = parser.parse_args()

    model_config = MageFlowCLIUtil.resolve_model_config(parser, args, edit=False)
    MageFlowCLIUtil.apply_model_defaults(parser, args, model_config)

    assert args.steps == 9


@pytest.mark.fast
def test_mage_flow_custom_model_preserves_metadata_steps(tmp_path: Path) -> None:
    metadata_path = tmp_path / "generation.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": "acme/custom",
                "base_model": "mage-flow-turbo",
                "prompt": "a silver fox",
                "steps": 7,
            }
        )
    )
    parser = _text_to_image_parser()
    with patch(
        "sys.argv",
        [
            "mflux-generate-mage-flow",
            "--config-from-metadata",
            str(metadata_path),
        ],
    ):
        args = parser.parse_args()

    model_config = MageFlowCLIUtil.resolve_model_config(parser, args, edit=False)
    MageFlowCLIUtil.apply_model_defaults(parser, args, model_config)

    assert args.steps == 7


@pytest.mark.fast
def test_mage_flow_edit_preserves_automatic_dimensions() -> None:
    parser = _edit_parser()
    with patch(
        "sys.argv",
        [
            "mflux-generate-mage-flow-edit",
            "--prompt",
            "blend these images",
            "--image-paths",
            "scene.png",
            "object.png",
            "--max-size",
            "1024",
        ],
    ):
        args = parser.parse_args()

    model_config = MageFlowCLIUtil.resolve_model_config(parser, args, edit=True)
    MageFlowCLIUtil.apply_model_defaults(parser, args, model_config)

    assert args.model == "mage-flow-edit"
    assert args.steps == 30
    assert args.guidance == pytest.approx(5.0)
    assert args.scheduler == "mage_flow"
    assert args.width is None
    assert args.height is None
    assert args.max_size == 1024
    assert args.image_paths == [Path("scene.png"), Path("object.png")]


@pytest.mark.fast
def test_mage_flow_edit_accepts_one_explicit_dimension() -> None:
    parser = _edit_parser()
    with patch(
        "sys.argv",
        [
            "mflux-generate-mage-flow-edit",
            "--prompt",
            "make it nocturnal",
            "--image-paths",
            "scene.png",
            "--width",
            "768",
        ],
    ):
        args = parser.parse_args()

    assert args.width == 768
    assert args.height is None


@pytest.mark.fast
def test_mage_flow_commands_reject_incompatible_variants() -> None:
    text_parser = _text_to_image_parser()
    with patch(
        "sys.argv",
        [
            "mflux-generate-mage-flow",
            "--model",
            "mage-flow-edit",
            "--prompt",
            "a silver fox",
        ],
    ):
        text_args = text_parser.parse_args()
    with pytest.raises(SystemExit):
        MageFlowCLIUtil.resolve_model_config(text_parser, text_args, edit=False)

    edit_parser = _edit_parser()
    with patch(
        "sys.argv",
        [
            "mflux-generate-mage-flow-edit",
            "--model",
            "mage-flow",
            "--prompt",
            "make it nocturnal",
            "--image-paths",
            "scene.png",
        ],
    ):
        edit_args = edit_parser.parse_args()
    with pytest.raises(SystemExit):
        MageFlowCLIUtil.resolve_model_config(edit_parser, edit_args, edit=True)


@pytest.mark.fast
def test_mage_flow_turbo_rejects_cfg() -> None:
    parser = _text_to_image_parser()
    with patch(
        "sys.argv",
        [
            "mflux-generate-mage-flow",
            "--model",
            "mage-flow-turbo",
            "--prompt",
            "a silver fox",
            "--guidance",
            "5",
        ],
    ):
        args = parser.parse_args()
    model_config = MageFlowCLIUtil.resolve_model_config(parser, args, edit=False)

    with pytest.raises(SystemExit):
        MageFlowCLIUtil.apply_model_defaults(parser, args, model_config)
