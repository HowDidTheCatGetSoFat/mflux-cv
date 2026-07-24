import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.mage_flow.cli import mage_flow_edit_generate, mage_flow_generate


@pytest.mark.fast
def test_mage_flow_generate_command_forwards_recommended_defaults() -> None:
    with (
        patch(
            "sys.argv",
            [
                "mflux-generate-mage-flow",
                "--prompt",
                "a silver fox",
                "--seed",
                "42",
                "--renormalization",
                "--gaussian-shading-key",
                "secret",
            ],
        ),
        patch.object(mage_flow_generate, "MageFlow", autospec=True) as model_class,
        patch.object(mage_flow_generate.CallbackManager, "register_callbacks", return_value=None),
    ):
        model = model_class.return_value
        image = model.generate_image.return_value
        mage_flow_generate.main()

    model_config = model_class.call_args.kwargs["model_config"]
    assert model_config.model_name == "microsoft/Mage-Flow"
    model.generate_image.assert_called_once_with(
        seed=42,
        prompt="a silver fox",
        num_inference_steps=20,
        height=1024,
        width=1024,
        guidance=5.0,
        negative_prompt=None,
        renormalization=True,
        gaussian_shading_key="secret",
        scheduler="mage_flow",
    )
    image.save.assert_called_once_with(path="image.png", export_json_metadata=False)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("model_name", "class_name", "expected_repo"),
    [
        ("mage-flow-turbo", "MageFlow", "microsoft/Mage-Flow-Turbo"),
        ("mage-flow-edit-turbo", "MageFlowEdit", "microsoft/Mage-Flow-Edit-Turbo"),
    ],
)
def test_mflux_save_dispatches_mage_flow_models(
    model_name: str,
    class_name: str,
    expected_repo: str,
) -> None:
    from mflux.models.common.cli import save

    with (
        patch(
            "sys.argv",
            [
                "mflux-save",
                "--model",
                model_name,
                "--path",
                "saved-model",
                "--quantize",
                "4",
            ],
        ),
        patch.object(save, class_name, autospec=True) as model_class,
    ):
        save.main()

    model_config = model_class.call_args.kwargs["model_config"]
    assert model_config.model_name == expected_repo
    assert model_class.call_args.kwargs["quantize"] == 4
    assert model_class.call_args.kwargs["model_path"] is None
    model_class.return_value.save_model.assert_called_once_with("saved-model")


@pytest.mark.fast
@pytest.mark.parametrize(
    ("base_model", "class_name", "directory_name"),
    [
        ("microsoft/Mage-Flow-Turbo", "MageFlow", "ernie-edited-model"),
        ("microsoft/Mage-Flow-Edit-Turbo", "MageFlowEdit", "opaque-source"),
    ],
)
def test_mflux_resave_dispatches_opaque_native_mage_directory(
    tmp_path: Path,
    base_model: str,
    class_name: str,
    directory_name: str,
) -> None:
    from mflux.models.common.cli import save

    source = tmp_path / directory_name
    source.mkdir()
    (source / ConfigResolution.SAVED_CONFIG_FILENAME).write_text(
        json.dumps({"model_name": base_model, "base_model": base_model})
    )

    with (
        patch(
            "sys.argv",
            [
                "mflux-save",
                "--model",
                str(source),
                "--path",
                str(tmp_path / "resaved"),
                "--quantize",
                "4",
            ],
        ),
        patch.object(save, class_name, autospec=True) as model_class,
        patch.object(save, "Flux1", autospec=True) as flux_class,
    ):
        save.main()

    flux_class.assert_not_called()
    assert model_class.call_args.kwargs["model_config"].base_model == base_model
    assert model_class.call_args.kwargs["model_path"] == str(source)
    model_class.return_value.save_model.assert_called_once_with(str(tmp_path / "resaved"))


@pytest.mark.fast
def test_mage_flow_edit_command_forwards_reference_sizing() -> None:
    with (
        patch(
            "sys.argv",
            [
                "mflux-generate-mage-flow-edit",
                "--model",
                "mage-flow-edit-turbo",
                "--prompt",
                "blend these",
                "--image-paths",
                "scene.png",
                "object.png",
                "--max-size",
                "768",
                "--seed",
                "7",
            ],
        ),
        patch.object(mage_flow_edit_generate, "MageFlowEdit", autospec=True) as model_class,
        patch.object(mage_flow_edit_generate.CallbackManager, "register_callbacks", return_value=None),
    ):
        model = model_class.return_value
        image = model.generate_image.return_value
        mage_flow_edit_generate.main()

    model_config = model_class.call_args.kwargs["model_config"]
    assert model_config.model_name == "microsoft/Mage-Flow-Edit-Turbo"
    model.generate_image.assert_called_once_with(
        seed=7,
        prompt="blend these",
        image_paths=[Path("scene.png"), Path("object.png")],
        num_inference_steps=4,
        height=None,
        width=None,
        max_size=768,
        guidance=1.0,
        negative_prompt=None,
        renormalization=False,
        gaussian_shading_key=None,
        scheduler="mage_flow",
    )
    image.save.assert_called_once_with(path="image.png", export_json_metadata=False)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("module", "model_class_name", "command", "extra_args"),
    [
        (mage_flow_generate, "MageFlow", "mflux-generate-mage-flow", []),
        (
            mage_flow_edit_generate,
            "MageFlowEdit",
            "mflux-generate-mage-flow-edit",
            ["--image-paths", "scene.png"],
        ),
    ],
)
def test_mage_flow_commands_format_output_with_resolved_random_seed(
    module,
    model_class_name: str,
    command: str,
    extra_args: list[str],
) -> None:
    first_image = MagicMock(seed=111)
    second_image = MagicMock(seed=222)
    with (
        patch(
            "sys.argv",
            [
                command,
                "--prompt",
                "a silver fox",
                "--seed",
                "-1",
                "-1",
                "--output",
                "result.png",
                *extra_args,
            ],
        ),
        patch.object(module, model_class_name, autospec=True) as model_class,
        patch.object(module.CallbackManager, "register_callbacks", return_value=None),
    ):
        model_class.return_value.generate_image.side_effect = [first_image, second_image]
        module.main()

    first_image.save.assert_called_once_with(path="result_seed_111.png", export_json_metadata=False)
    second_image.save.assert_called_once_with(path="result_seed_222.png", export_json_metadata=False)
