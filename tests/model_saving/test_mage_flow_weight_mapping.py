import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.mage_flow.model.mage_flow_text_encoder import MageFlowTextEncoder
from mflux.models.mage_flow.model.mage_flow_transformer import MageFlowTransformer
from mflux.models.mage_flow.model.mage_flow_vae.vae import MageVAE, _DConvDenoiser, _DConvEncoder
from mflux.models.mage_flow.weights import MageFlowWeightDefinition, MageFlowWeightMapping


def _tiny_transformer() -> MageFlowTransformer:
    return MageFlowTransformer(
        in_channels=16,
        out_channels=16,
        context_in_dim=64,
        hidden_size=96,
        num_attention_heads=3,
        depth=12,
        axes_dim=(8, 12, 12),
    )


def _tiny_text_encoder() -> MageFlowTextEncoder:
    return MageFlowTextEncoder(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=36,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        head_dim=16,
        mrope_section=(4, 2, 2),
        vision_config={
            "hidden_size": 32,
            "num_heads": 4,
            "intermediate_size": 64,
            "depth": 24,
            "num_position_embeddings": 64,
        },
    )


def _tiny_vae() -> MageVAE:
    encoder = _DConvEncoder(
        z_ch=4,
        hidden_size=32,
        num_blocks=1,
        patch_size=4,
        mlp_ratio=2.0,
        head_size=32,
        num_head_blocks=1,
    )
    decoder = _DConvDenoiser(
        patch_size=4,
        hidden_size=32,
        hidden_size_x=8,
        mlp_ratio=2.0,
        num_blocks=2,
        num_cond_blocks=1,
        bottleneck_dim=4,
        attention_patch_size=2,
    )
    return MageVAE(sample_posterior=False, encoder=encoder, decoder_model=decoder)


def test_mage_flow_components_expose_subdirs_coverage_and_post_load_metadata():
    components = {component.name: component for component in MageFlowWeightDefinition.get_components()}

    assert set(components) == {"transformer", "text_encoder", "vae"}
    assert components["transformer"].hf_subdir == "transformer"
    assert components["text_encoder"].hf_subdir == "text_encoder"
    assert components["vae"].hf_subdir == "vae"
    assert components["transformer"].expected_hf_weight_count == 397
    assert components["text_encoder"].expected_hf_weight_count == 713
    assert components["vae"].expected_hf_weight_count == 728
    assert components["vae"].folded_weight_count == 686
    assert components["vae"].post_load_hook == "freeze_adaln_cache"
    assert components["text_encoder"].skip_quantization is True


def test_mage_flow_tokenizer_uses_the_checkpoint_text_encoder_processor():
    tokenizer = MageFlowWeightDefinition.get_tokenizers()[0]

    assert tokenizer.name == "mage"
    assert tokenizer.hf_subdir == "text_encoder"
    assert tokenizer.max_length == 2112
    assert tokenizer.encoder_class.__name__ == "VisionLanguageTokenizer"
    assert tokenizer.processor_class.__name__ == "MageFlowQwen3VLProcessor"
    assert tokenizer.download_patterns == ["text_encoder/**"]


def test_mage_flow_hf_loaded_weight_coverage_is_strict_but_local_metadata_is_supported():
    components = {
        component_name: {f"weight_{index}": mx.array(0) for index in range(expected_count)}
        for component_name, expected_count in MageFlowWeightMapping.EXPECTED_HF_WEIGHT_COUNTS.items()
    }
    hf_weights = LoadedWeights(components=components, meta_data=MetaData())
    MageFlowWeightDefinition.validate_loaded_weights(hf_weights)

    components["transformer"].pop("weight_0")
    try:
        MageFlowWeightDefinition.validate_loaded_weights(hf_weights)
    except ValueError as error:
        assert "transformer expected 397 weights, got 396" in str(error)
    else:
        raise AssertionError("Incomplete Hugging Face weights must fail coverage validation")

    local_weights = LoadedWeights(
        components={},
        meta_data=MetaData(quantization_level=4, mflux_version="0.18.0"),
    )
    MageFlowWeightDefinition.validate_loaded_weights(local_weights)


def test_mage_flow_transformer_mapping_is_direct_for_all_397_tensors():
    target_keys = {key for key, _ in tree_flatten(_tiny_transformer().parameters())}

    mapped_keys = MageFlowWeightMapping.validate_hf_coverage("transformer", target_keys)

    assert len(target_keys) == 397
    assert mapped_keys == target_keys


def test_mage_flow_text_mapping_strips_model_prefix_for_all_713_checkpoint_tensors():
    model_keys = {key for key, _ in tree_flatten(_tiny_text_encoder().parameters())}
    checkpoint_targets = model_keys
    source_keys = {f"model.{key}" for key in checkpoint_targets}

    mapped_keys = MageFlowWeightMapping.validate_hf_coverage("text_encoder", source_keys)

    assert len(model_keys) == 713
    assert len(checkpoint_targets) == 713
    assert mapped_keys == checkpoint_targets
    assert MageFlowWeightMapping.transform_text_encoder_key("lm_head.weight") is None


def test_mage_flow_text_mapping_transposes_only_the_visual_conv3d_kernel():
    source = mx.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)

    mapped = MageFlowWeightMapping.transform_text_encoder_weight(
        "visual.patch_embed.proj.weight",
        source,
    )
    unchanged = MageFlowWeightMapping.transform_text_encoder_weight(
        "visual.blocks.0.attn.qkv.weight",
        source,
    )

    assert mapped.shape == (2, 4, 5, 6, 3)
    assert mapped[1, 2, 3, 4, 0].item() == source[1, 0, 2, 3, 4].item()
    assert unchanged is source


def test_mage_flow_vae_mapping_covers_728_used_tensors_and_ignores_only_legacy_encoder():
    target_keys = {key for key, _ in tree_flatten(MageVAE(sample_posterior=False).parameters())}
    source_keys = {
        (
            f"student.dconv_encoder.{key[len('encoder.') :]}"
            if key.startswith("encoder.")
            else f"pipeline.{key[len('decoder_model.') :]}"
        )
        for key in target_keys
    }
    source_keys.add("pipeline.y_embedder.encoder.legacy.weight")

    mapped_keys = MageFlowWeightMapping.validate_hf_coverage("vae", source_keys)

    assert len(target_keys) == 728
    assert mapped_keys == target_keys


def test_mage_flow_vae_mapping_transposes_conv2d_kernels():
    source = mx.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    mapped = MageFlowWeightMapping.transform_vae_weight("encoder.proj_out.weight", source)
    bias = mx.zeros((2,))

    assert mapped.shape == (2, 4, 5, 3)
    assert mapped[1, 2, 3, 0].item() == source[1, 0, 2, 3].item()
    assert MageFlowWeightMapping.transform_vae_weight("encoder.proj_out.bias", bias) is bias


def test_mage_flow_folded_vae_mflux_save_round_trip(tmp_path):
    original = _tiny_vae()
    assert original.freeze_adaln_cache() == 2
    ModelSaver._save_weights(str(tmp_path), None, original, "vae")

    loaded, quantization_level, version = WeightLoader._try_load_mflux_format(tmp_path / "vae")
    assert loaded is not None
    assert quantization_level is None
    assert version is not None
    assert MageFlowWeightDefinition.is_folded_vae_weights(loaded)

    reloaded = _tiny_vae()
    assert MageFlowWeightDefinition.prepare_vae_for_loading(reloaded, loaded)
    reloaded.update(loaded, strict=True)
    assert MageFlowWeightDefinition.finalize_vae_after_loading(reloaded, loaded) == 0

    original_parameters = dict(tree_flatten(original.parameters()))
    reloaded_parameters = dict(tree_flatten(reloaded.parameters()))
    assert set(reloaded_parameters) == set(original_parameters)
    modulation_keys = [key for key in original_parameters if key.endswith(".adaLN_modulation.modulation")]
    assert len(modulation_keys) == 2
    for key in modulation_keys:
        assert bool(mx.array_equal(reloaded_parameters[key], original_parameters[key]))


def test_mage_flow_quantization_keeps_fold_inputs_unquantized():
    assert MageFlowWeightDefinition.quantization_predicate("transformer.proj_out", nn.Linear(64, 64))
    assert not MageFlowWeightDefinition.quantization_predicate("vae.encoder.t_embedder.mlp.0", nn.Linear(64, 64))
    assert not MageFlowWeightDefinition.quantization_predicate(
        "vae.encoder.blocks.0.adaLN_modulation.1",
        nn.Linear(64, 64),
    )
    assert not MageFlowWeightDefinition.quantization_predicate("vae.decoder.final_layer", nn.Linear(63, 64))
    assert not MageFlowWeightDefinition.quantization_predicate("vae.encoder.conv", nn.Conv2d(3, 32, 3))
