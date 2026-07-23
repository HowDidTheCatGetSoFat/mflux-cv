from collections.abc import Iterable

import mlx.core as mx


class MageFlowWeightMapping:
    EXPECTED_HF_WEIGHT_COUNTS = {
        "transformer": 397,
        "text_encoder": 713,
        "vae": 728,
    }
    EXPECTED_FOLDED_VAE_WEIGHT_COUNT = 686
    LEGACY_VAE_ENCODER_PREFIX = "pipeline.y_embedder.encoder."

    @staticmethod
    def transform_transformer_key(key: str) -> str:
        return key

    @staticmethod
    def transform_text_encoder_key(key: str) -> str | None:
        if key in {"lm_head.weight", "model.visual.rotary_pos_emb.inv_freq"}:
            return None
        if key.startswith(("model.language_model.", "model.visual.")):
            return key[len("model.") :]
        raise ValueError(f"Unexpected Mage Flow text encoder weight: {key}")

    @staticmethod
    def transform_vae_key(key: str) -> str | None:
        if key.startswith(MageFlowWeightMapping.LEGACY_VAE_ENCODER_PREFIX):
            return None
        if key.startswith("student.dconv_encoder."):
            return f"encoder.{key[len('student.dconv_encoder.') :]}"
        if key.startswith("pipeline."):
            return f"decoder_model.{key[len('pipeline.') :]}"
        raise ValueError(f"Unexpected Mage Flow VAE weight: {key}")

    @staticmethod
    def transform_text_encoder_weight(key: str, value: mx.array) -> mx.array:
        if key != "visual.patch_embed.proj.weight":
            return value
        if value.ndim != 5:
            raise ValueError(
                "visual.patch_embed.proj.weight must have OITHW layout before conversion",
            )
        return value.transpose(0, 2, 3, 4, 1)

    @staticmethod
    def transform_vae_weight(key: str, value: mx.array) -> mx.array:
        if value.ndim == 4:
            return value.transpose(0, 2, 3, 1)
        return value

    @classmethod
    def transform_key(cls, component_name: str, key: str) -> str | None:
        if component_name == "transformer":
            return cls.transform_transformer_key(key)
        if component_name == "text_encoder":
            return cls.transform_text_encoder_key(key)
        if component_name == "vae":
            return cls.transform_vae_key(key)
        raise ValueError(f"Unknown Mage Flow component: {component_name}")

    @classmethod
    def validate_hf_coverage(cls, component_name: str, source_keys: Iterable[str]) -> frozenset[str]:
        if component_name not in cls.EXPECTED_HF_WEIGHT_COUNTS:
            raise ValueError(f"Unknown Mage Flow component: {component_name}")

        mapped_keys = [
            mapped
            for source_key in source_keys
            if (mapped := cls.transform_key(component_name, source_key)) is not None
        ]
        expected_count = cls.EXPECTED_HF_WEIGHT_COUNTS[component_name]
        if len(mapped_keys) != expected_count:
            raise ValueError(
                f"Mage Flow {component_name} expected {expected_count} mapped weights, got {len(mapped_keys)}",
            )
        unique_keys = frozenset(mapped_keys)
        if len(unique_keys) != len(mapped_keys):
            raise ValueError(f"Mage Flow {component_name} mapping produced duplicate target keys")
        return unique_keys
