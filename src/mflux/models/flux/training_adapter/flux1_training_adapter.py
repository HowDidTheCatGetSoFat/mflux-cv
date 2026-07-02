from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.utils import TrainingUtil
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping
from mflux.utils.version_util import VersionUtil


class Flux1TrainingAdapter(TrainingAdapter):
    """LoRA training adapter for FLUX.1 (dev / schnell) on MLX.

    Re-enables FLUX.1 LoRA training in mflux (removed when the trainer moved to the
    adapter architecture). Single-class (no edit variant), mirroring the flux2
    adapter; only the model-specific pieces are supplied — the flow-matching loss,
    timestep sampling and LoRA injection live in the generic trainer.
    """

    def __init__(self, *, model_config: ModelConfig, quantize: int | None, model_path: str | None = None):
        self._model_config = model_config
        self._flux = Flux1(quantize=quantize, model_path=model_path, model_config=model_config)
        self._guidance: float = 0.0

    def model(self):
        return self._flux

    def transformer(self):
        return self._flux.transformer

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config:
        # FLUX.1 inference uses the "linear" scheduler (flux.py:60), not flow-match.
        self._guidance = training_spec.guidance
        return Config(
            model_config=self._model_config,
            num_inference_steps=training_spec.steps,
            width=width,
            height=height,
            guidance=training_spec.guidance,
            scheduler="linear",
        )

    def freeze_base(self) -> None:
        self._flux.vae.freeze()
        self._flux.transformer.freeze()
        self._flux.t5_text_encoder.freeze()
        self._flux.clip_text_encoder.freeze()

    def encode_data(
        self,
        *,
        data_id: int,
        image_path: Path,
        prompt: str,
        width: int,
        height: int,
        input_image_path: Path | None = None,  # txt2img: unused
    ) -> tuple[mx.array, Any]:
        # image -> VAE latents (1,16,H/8,W/8) -> FLUX.1 packing (1, seq, 64)
        encoded = LatentCreator.encode_image(
            vae=self._flux.vae,
            image_path=image_path,
            height=height,
            width=width,
            tiling_config=self._flux.tiling_config,
        )
        clean_latents = FluxLatentCreator.pack_latents(encoded, height=height, width=width)
        # prompt -> (prompt_embeds [1,seq,4096], pooled_prompt_embeds [1,768])
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache={},
            t5_tokenizer=self._flux.tokenizers["t5"],
            clip_tokenizer=self._flux.tokenizers["clip"],
            t5_text_encoder=self._flux.t5_text_encoder,
            clip_text_encoder=self._flux.clip_text_encoder,
        )
        mx.eval(clean_latents, prompt_embeds, pooled_prompt_embeds)
        return clean_latents, {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config, sigma: float | None = None) -> mx.array:  # noqa: ARG002
        # FLUX.1 transformer takes the INTEGER step index t (it reads config.scheduler.sigmas[t]
        # internally) — unlike flux2, which is passed config.scheduler.timesteps[t].
        return self._flux.transformer(
            t=t,
            config=config,
            hidden_states=latents_t,
            prompt_embeds=cond["prompt_embeds"],
            pooled_prompt_embeds=cond["pooled_prompt_embeds"],
        )

    def generate_preview_image(
        self,
        *,
        seed: int,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        image_paths: list[Path | str] | None = None,  # txt2img: unused
    ):
        with self._assistant_disabled():
            image = self._flux.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=self._guidance,
            )
        self._flux.prompt_cache = {}
        return image.image

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None:  # noqa: ARG002
        weights: dict[str, mx.array] = {}
        for target in training_spec.lora_layers.targets:
            if target.blocks is not None:
                for b in target.blocks.get_blocks():
                    self._append_train_lora_weights(weights, self._flux.transformer, target.module_path.format(block=b))
            else:
                self._append_train_lora_weights(weights, self._flux.transformer, target.module_path)

        mx.save_safetensors(
            str(path),
            weights,
            metadata={
                "mflux_version": VersionUtil.get_mflux_version(),
                "model": training_spec.model,
            },
        )

    def load_lora_adapter(self, *, path: str | Path) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=FluxLoRAMapping.get_mapping(),
            transformer=self._flux.transformer,
            lora_paths=[str(path)],
            lora_scales=[1.0],
            role="train",
            bake_lora=False,
        )

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=FluxLoRAMapping.get_mapping(),
            transformer=self._flux.transformer,
            lora_paths=[str(path)],
            lora_scales=[float(scale)],
            role="assistant",
            bake_lora=False,
        )

    def _assistant_disabled(self):
        return TrainingUtil.assistant_disabled(self._flux.transformer)

    @staticmethod
    def _append_train_lora_weights(weights: dict[str, mx.array], transformer, module_path: str) -> None:
        train_lora = TrainingUtil.get_train_lora(transformer, module_path)
        weights[f"transformer.{module_path}.lora_A.weight"] = mx.transpose(train_lora.lora_A)
        weights[f"transformer.{module_path}.lora_B.weight"] = mx.transpose(train_lora.lora_B)
        if train_lora.dora_scale is not None:
            weights[f"transformer.{module_path}.dora_scale"] = train_lora.dora_scale
