"""Multi-ControlNet stacking for FLUX.1.

The residual path is additive, so several controlnets stack by summing their per-block residuals.
Nets can have different block counts, so each net's residuals are first spread over the
transformer's blocks with the same rule the transformer itself applies, then summed. These tests
pin that the expansion matches the transformer exactly (single-net behavior is unchanged) and that
heterogeneous nets sum correctly. No weights are loaded.
"""
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet


def _samples(values):
    return [mx.full((1, 1), float(v)) for v in values]


# --------------------------------------------------------------------------- #
# residual expansion + summing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("num_samples", "num_blocks"), [(5, 19), (2, 19), (1, 19), (19, 19), (2, 38), (4, 38)])
def test_broadcast_matches_the_transformer_rule(num_samples, num_blocks):
    # Expanding a net's residuals must reproduce, index for index, what the transformer would have
    # selected on its own. This is what makes the single-controlnet path numerically unchanged.
    samples = _samples(range(num_samples))
    expanded = Flux1Controlnet._broadcast_samples(samples, num_blocks)

    assert len(expanded) == num_blocks
    blocks = list(range(num_blocks))
    for idx in range(num_blocks):
        expected = Transformer._get_controlnet_sample(idx, blocks, samples)
        assert expanded[idx].item() == expected.item()


def test_broadcast_of_no_samples_is_none():
    assert Flux1Controlnet._broadcast_samples([], 19) is None


def test_single_net_combine_is_identical_to_the_old_direct_path():
    # combine() of one net, indexed 1:1 by the transformer, equals the transformer's own broadcast
    # of that net's raw samples. Stacking must not change a single-controlnet render.
    samples = _samples([1, 2, 3, 4, 5])
    combined = Flux1Controlnet._combine_samples([samples], 19)

    blocks = list(range(19))
    for idx in range(19):
        direct = Transformer._get_controlnet_sample(idx, blocks, samples)
        # the transformer now receives a full-length list, so it indexes it 1:1
        stacked = Transformer._get_controlnet_sample(idx, blocks, combined)
        assert stacked.item() == direct.item()


def test_two_nets_with_equal_block_counts_sum():
    a = _samples([1, 1, 1, 1, 1])
    b = _samples([2, 2, 2, 2, 2])
    combined = Flux1Controlnet._combine_samples([a, b], 19)

    assert len(combined) == 19
    assert all(s.item() == 3.0 for s in combined)


def test_two_nets_with_different_block_counts_sum_per_block():
    # a: 5 residuals over 19 blocks (interval 4), b: 2 residuals over 19 blocks (interval 10)
    a = _samples([0, 1, 2, 3, 4])
    b = _samples([10, 20])
    combined = Flux1Controlnet._combine_samples([a, b], 19)

    assert len(combined) == 19
    for idx in range(19):
        expected = a[idx // 4].item() + b[idx // 10].item()
        assert combined[idx].item() == expected


def test_combine_skips_nets_that_contributed_nothing():
    # a net with no single-transformer blocks contributes an empty list and must not break the sum
    a = _samples([1, 1])
    combined = Flux1Controlnet._combine_samples([a, []], 19)

    assert len(combined) == 19
    assert all(s.item() == 1.0 for s in combined)


def test_combine_of_all_empty_is_empty_so_the_transformer_adds_nothing():
    combined = Flux1Controlnet._combine_samples([[], []], 38)
    assert combined == []
    # an empty list makes the transformer contribute no residual at all
    assert Transformer._get_controlnet_sample(0, list(range(38)), combined) is None


# --------------------------------------------------------------------------- #
# CLI: repeatable controlnet flags
# --------------------------------------------------------------------------- #
def _parser() -> CommandLineParser:
    parser = CommandLineParser(description="controlnet")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_controlnet_arguments(require_image=True)
    parser.add_output_arguments()
    return parser


def _parse(argv):
    with patch("sys.argv", ["prog", *argv]):
        return _parser().parse_args()


def test_single_controlnet_keeps_the_scalar_shape():
    args = _parse(["--prompt", "x", "--controlnet-image-path", "a.png", "--controlnet-strength", "0.8"])
    assert args.controlnet_image_path == "a.png"
    assert args.controlnet_strength == pytest.approx(0.8)
    assert args.controlnet_path is None


def test_strength_defaults_to_the_scalar_default():
    args = _parse(["--prompt", "x", "--controlnet-image-path", "a.png"])
    assert isinstance(args.controlnet_strength, float)


def test_repeating_the_flags_stacks_controlnets():
    args = _parse([
        "--prompt", "x",
        "--controlnet-image-path", "depth.png", "--controlnet-path", "org/depth",
        "--controlnet-image-path", "canny.png", "--controlnet-path", "org/canny",
        "--controlnet-strength", "0.7", "--controlnet-strength", "0.4",
    ])
    assert args.controlnet_image_path == ["depth.png", "canny.png"]
    assert args.controlnet_path == ["org/depth", "org/canny"]
    assert args.controlnet_strength == pytest.approx([0.7, 0.4])


def test_one_strength_applies_to_every_stacked_controlnet():
    args = _parse([
        "--prompt", "x",
        "--controlnet-image-path", "a.png", "--controlnet-path", "org/a",
        "--controlnet-image-path", "b.png", "--controlnet-path", "org/b",
        "--controlnet-strength", "0.5",
    ])
    assert args.controlnet_image_path == ["a.png", "b.png"]
    assert args.controlnet_strength == pytest.approx(0.5)


def test_mismatched_strength_count_is_rejected():
    with pytest.raises(SystemExit):
        _parse([
            "--prompt", "x",
            "--controlnet-image-path", "a.png", "--controlnet-path", "org/a",
            "--controlnet-image-path", "b.png", "--controlnet-path", "org/b",
            "--controlnet-strength", "0.5", "--controlnet-strength", "0.6", "--controlnet-strength", "0.7",
        ])


def test_controlnet_path_without_a_control_image_is_rejected():
    # --controlnet-image-path is not required on every controlnet CLI, so a checkpoint with no
    # control image must be caught here rather than at load time.
    parser = CommandLineParser(description="controlnet")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_controlnet_arguments(mode="canny")  # require_image defaults to False here
    parser.add_output_arguments()
    with patch("sys.argv", ["prog", "--prompt", "x", "--controlnet-path", "org/a"]):
        with pytest.raises(SystemExit):
            parser.parse_args()


def test_canny_preprocessing_follows_the_config_only_when_the_config_picks_the_controlnet():
    # The config's is_canny describes the controlnet IT names. An explicitly supplied checkpoint may
    # be anything (a depth net), so its control image must not be run through the Canny detector.
    from mflux.models.common.config.model_config import ModelConfig

    canny_cfg = ModelConfig.from_name("dev-controlnet-canny")
    assert canny_cfg.is_canny() is True

    from_config = Flux1Controlnet.__new__(Flux1Controlnet)
    from_config.controlnet_from_config = True
    from_config.model_config = canny_cfg
    assert canny_cfg.is_canny() and from_config.controlnet_from_config

    explicit = Flux1Controlnet.__new__(Flux1Controlnet)
    explicit.controlnet_from_config = False
    explicit.model_config = canny_cfg
    assert not (canny_cfg.is_canny() and explicit.controlnet_from_config)


def test_saving_a_stack_is_rejected_rather_than_dropping_nets():
    # The saved layout holds a single controlnet, so saving a stack must not silently write only
    # the first net.
    flux = Flux1Controlnet.__new__(Flux1Controlnet)
    flux.transformer_controlnets = [object(), object()]
    with pytest.raises(ValueError, match="stacked controlnets"):
        flux.save_model("/tmp/whatever")


def test_stacking_without_controlnet_paths_is_rejected():
    # The model config names a single controlnet, so several control images cannot be satisfied.
    # This must fail during parsing, not after the model has been loaded.
    with pytest.raises(SystemExit):
        _parse([
            "--prompt", "x",
            "--controlnet-image-path", "a.png", "--controlnet-image-path", "b.png",
        ])


def test_mismatched_controlnet_path_count_is_rejected():
    with pytest.raises(SystemExit):
        _parse([
            "--prompt", "x",
            "--controlnet-image-path", "a.png", "--controlnet-image-path", "b.png",
            "--controlnet-path", "org/only-one",
        ])


def test_one_image_with_several_strengths_is_rejected():
    # The counts must be checked before a single image collapses to a scalar, or this invalid
    # combination would sail through parsing and only fail once the model is being loaded.
    with pytest.raises(SystemExit):
        _parse([
            "--prompt", "x", "--controlnet-image-path", "a.png",
            "--controlnet-strength", "0.5", "--controlnet-strength", "0.6",
        ])


def test_one_image_with_several_controlnet_paths_is_rejected():
    with pytest.raises(SystemExit):
        _parse([
            "--prompt", "x", "--controlnet-image-path", "a.png",
            "--controlnet-path", "org/a", "--controlnet-path", "org/b",
        ])


# --------------------------------------------------------------------------- #
# checkpoint sources: --controlnet-path takes a local directory or an HF repo
# --------------------------------------------------------------------------- #
def test_load_single_resolves_a_local_directory_without_hitting_huggingface(tmp_path):
    # A local --controlnet-path must not be handed to snapshot_download as a repo id.
    from mflux.models.common.weights.loading.weight_loader import WeightLoader

    local_dir = tmp_path / "my-controlnet"
    local_dir.mkdir()
    (local_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"")

    component = SimpleNamespace(name="transformer_controlnet")
    with patch("mflux.models.common.weights.loading.weight_loader.snapshot_download") as dl:
        with patch.object(WeightLoader, "_load_component", return_value=({}, None, None)) as load_component:
            WeightLoader.load_single(component=component, repo_id=str(local_dir))

    dl.assert_not_called()
    assert load_component.call_args.args[0] == local_dir
