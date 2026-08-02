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
                assert option["condition"] and option["reason"]


@pytest.mark.fast
def test_known_statuses_survive(caps):
    def status_of(command_name, flag):
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
    assert dumped["commands"][0]["coverage"] == "full"
