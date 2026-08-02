import importlib
import json
import sys

import pytest

from mflux.cli import capabilities


@pytest.fixture(scope="module")
def caps():
    return capabilities.build_capabilities()


@pytest.mark.fast
def test_every_generate_entrypoint_has_full_coverage(caps):
    # The dump discovers commands from the installed entry points, so this fails the
    # moment someone adds a generate CLI without the build_parser() convention: the
    # command shows up on its own (self-healing coverage) and this test names it.
    assert caps["commands"], "entry-point discovery found nothing"
    not_full = [(c["command"], c.get("coverage")) for c in caps["commands"] if c.get("coverage") != "full"]
    assert not_full == [], f"commands without full capability coverage: {not_full}"


@pytest.mark.fast
def test_declared_flags_exist_in_their_parsers(caps):
    # A declaration for a flag the parser no longer takes is a lie waiting to be
    # printed; catch it at the source.
    for command in caps["commands"]:
        module = importlib.import_module(command["module"])
        parser = module.build_parser()
        known_flags = {flag for action in parser._actions for flag in action.option_strings}
        declared = set(getattr(module, "IGNORED_OPTIONS", {})) | set(getattr(module, "CONDITIONAL_OPTIONS", {}))
        missing = declared - known_flags
        assert not missing, f"{command['command']} declares options its parser does not take: {sorted(missing)}"


@pytest.mark.fast
def test_option_records_are_well_formed(caps):
    for command in caps["commands"]:
        for option in command["options"]:
            assert option["flag"].startswith("--")
            assert option["status"] in ("honored", "ignored", "conditional")
            if option["status"] == "ignored":
                assert option["reason"]
            if option["status"] == "conditional":
                assert option["condition"]
                assert option["reason"]


@pytest.mark.fast
def test_known_statuses_survive(caps):
    def status_of(command_name: str, flag: str) -> dict:
        command = next(c for c in caps["commands"] if c["command"] == command_name)
        return next(o for o in command["options"] if o["flag"] == flag)

    assert status_of("mflux-generate", "--negative-prompt")["status"] == "ignored"
    assert status_of("mflux-generate-ideogram4", "--steps")["status"] == "ignored"
    assert status_of("mflux-generate-z-image-turbo", "--guidance")["status"] == "ignored"
    assert status_of("mflux-generate-z-image", "--negative-prompt")["status"] == "conditional"
    assert status_of("mflux-generate-z-image", "--seed")["status"] == "honored"


@pytest.mark.fast
def test_json_output_is_valid(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mflux-capabilities", "--command", "mflux-generate-z-image-turbo"])
    capabilities.main()
    dumped = json.loads(capsys.readouterr().out)
    assert dumped["schema_version"] == capabilities.SCHEMA_VERSION
    assert len(dumped["commands"]) == 1
    assert dumped["commands"][0]["command"] == "mflux-generate-z-image-turbo"
    assert dumped["commands"][0]["coverage"] == "full"


@pytest.mark.fast
def test_full_dump_serializes_to_every_format(caps):
    # A parser default that is not JSON-native (qwen-layered's Path --output-dir)
    # truncated the streamed JSON mid-value; defaults are normalized at record-build
    # time so every wire format serializes the whole document.
    text = json.dumps(caps)
    assert text.endswith("}")
    for command in caps["commands"]:
        for option in command["options"]:
            default = option["parser_default"]
            assert default is None or isinstance(default, (str, int, float, bool, list)), (
                f"{command['command']} {option['flag']} default is {type(default).__name__}"
            )


@pytest.mark.fast
def test_flux_family_negative_prompt_gaps_are_declared(caps):
    # The controlnet and depth variants never read negative_prompt (zero mentions in
    # their variant trees), same as the base FLUX.1 CLI that already declared it.
    for command_name in (
        "mflux-generate-controlnet",
        "mflux-generate-depth",
        "mflux-generate-fill",
        "mflux-generate-redux",
        "mflux-generate-kontext",
        "mflux-generate-in-context",
        "mflux-generate-in-context-catvton",
        "mflux-generate-in-context-edit",
    ):
        command = next(c for c in caps["commands"] if c["command"] == command_name)
        option = next(o for o in command["options"] if o["flag"] == "--negative-prompt")
        assert option["status"] == "ignored", command_name


@pytest.mark.fast
def test_flux_guidance_is_conditional_on_the_resolved_model(caps):
    # dev honours --guidance; schnell has supports_guidance=False and builds no
    # guidance embedder, so the same CLI gives two answers keyed on --base-model.
    command = next(c for c in caps["commands"] if c["command"] == "mflux-generate")
    option = next(o for o in command["options"] if o["flag"] == "--guidance")
    assert option["status"] == "conditional"
    assert "schnell" in option["condition"]


@pytest.mark.fast
def test_controlnet_guidance_is_conditional_on_the_resolved_model(caps):
    # --model schnell resolves to schnell_controlnet_canny (supports_guidance=False),
    # same shape as the base CLI: two answers keyed on the model flag.
    command = next(c for c in caps["commands"] if c["command"] == "mflux-generate-controlnet")
    option = next(o for o in command["options"] if o["flag"] == "--guidance")
    assert option["status"] == "conditional"
    assert "schnell" in option["condition"]
    assert option["reason"]


@pytest.mark.fast
def test_jsonable_preserves_mappings():
    # A dict default must stay a JSON object, not become a Python repr string.
    from pathlib import Path

    assert capabilities._jsonable({"width": 1024, "p": Path("x")}) == {"width": 1024, "p": "x"}
    assert capabilities._jsonable({1: {"p": Path("x")}}) == {"1": {"p": "x"}}
    assert capabilities._jsonable([Path("a"), 2]) == ["a", 2]
    with pytest.raises(ValueError, match="loses keys"):
        capabilities._jsonable({1: "a", "1": "b"})
