import sys
import warnings

import pytest

from mflux.cli.parser.parsers import CommandLineParser


@pytest.mark.fast
def test_provided_option_warns(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "x", "--negative-prompt", "blurry"])
    with pytest.warns(UserWarning, match=r"--negative-prompt is ignored; because reasons\."):
        CommandLineParser.warn_ignored_options({"--negative-prompt": "because reasons."})


@pytest.mark.fast
def test_equals_form_warns(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--negative-prompt=blurry"])
    with pytest.warns(UserWarning, match="--negative-prompt is ignored"):
        CommandLineParser.warn_ignored_options({"--negative-prompt": "because reasons."})


@pytest.mark.fast
def test_absent_option_is_silent(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "x"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CommandLineParser.warn_ignored_options({"--negative-prompt": "because reasons."})


@pytest.mark.fast
def test_each_provided_option_warns_once(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--guidance", "3.5", "--negative-prompt", "blurry"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CommandLineParser.warn_ignored_options(
            {
                "--guidance": "reason a.",
                "--negative-prompt": "reason b.",
                "--steps": "reason c.",
            }
        )
    messages = [str(w.message) for w in caught]
    assert messages == [
        "--guidance is ignored; reason a.",
        "--negative-prompt is ignored; reason b.",
    ]
