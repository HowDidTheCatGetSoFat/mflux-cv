import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig


@pytest.mark.fast
def test_base_model_keyword_alone_resolves_a_real_model_name():
    # `--base-model schnell` with no --model asks for the base model itself. A test over
    # the static table stays green through this bug: the table is right and from_name is
    # what was wrong, handing back model_name=None when only the base_model keyword named
    # the model, which every initializer then fails to resolve a weights path from.
    for key, config in AVAILABLE_MODELS.items():
        if config.base_model is not None:
            continue
        for alias in config.aliases:
            resolved = ModelConfig.from_name(model_name=None, base_model=alias)
            assert resolved.model_name == config.model_name, (
                f"base_model={alias!r} resolved model_name={resolved.model_name!r}, "
                f"expected {config.model_name!r}"
            )


@pytest.mark.fast
def test_both_keywords_yield_the_same_shape():
    # Whichever keyword names the model, the resolved config must look the same.
    by_model = ModelConfig.from_name(model_name="schnell", base_model=None)
    by_base = ModelConfig.from_name(model_name=None, base_model="schnell")
    assert by_base.model_name == by_model.model_name
    assert by_base.supports_guidance == by_model.supports_guidance


@pytest.mark.fast
def test_custom_checkpoint_with_base_model_keeps_both_fields():
    # The two-keyword form (a third-party checkpoint on a known architecture) is the
    # case the explicit-base rule exists for; it must keep working unchanged.
    resolved = ModelConfig.from_name(model_name="some-user/some-finetune", base_model="schnell")
    assert resolved.model_name == "some-user/some-finetune"
    assert resolved.base_model == "black-forest-labs/FLUX.1-schnell"
