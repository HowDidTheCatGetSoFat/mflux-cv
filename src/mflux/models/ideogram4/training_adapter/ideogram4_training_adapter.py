from __future__ import annotations
import math

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
from mflux.models.ideogram4.latent_creator.ideogram4_latent_creator import Ideogram4LatentCreator
from mflux.models.ideogram4.model.ideogram4_text_encoder.caption import Ideogram4Caption
from mflux.models.ideogram4.model.ideogram4_text_encoder.prompt_encoder import Ideogram4PromptEncoder
from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4
from mflux.models.ideogram4.weights.ideogram4_lora_mapping import Ideogram4LoRAMapping
from mflux.utils.version_util import VersionUtil


class Ideogram4TrainingAdapter(TrainingAdapter):
    """QLoRA training adapter for Ideogram-4 (fp8) on MLX.

    Ideogram-4 ships fp8-only (no bf16 base), so training is QLoRA: float LoRA over the
    frozen Fp8Linear base (the LoRA injector was taught about Fp8Linear). Specifics vs
    the flux adapters:
      - JSON-structured captions (Ideogram4Caption.prepare).
      - the conditional transformer takes (llm_features, x, t, position_ids, segment_ids,
        indicator) with text zero-padding prepended; the image region is sliced back out.
      - save prefix is diffusion_model.* (z_image / ai-toolkit convention).
    The timestep fed to the transformer is the SAME sigma the trainer used to noise the
    latents (sigmas[t]), which keeps the noise level and the timestep consistent.
    """

    def __init__(self, *, model_config: ModelConfig, quantize: int | None, model_path: str | None = None):
        self._model_config = model_config
        # ideogram-4-fp8 weights are already Fp8Linear (uint8+scale); quantize does not
        # re-quantize them. The LoRA trains in float over the frozen fp8 base = QLoRA.
        self._ideo = Ideogram4(quantize=quantize, model_config=model_config, model_path=model_path)
        self._guidance: float | None = None

    def model(self):
        return self._ideo

    def transformer(self):
        # Train/inject only the conditional transformer (unconditional is preview-CFG only).
        return self._ideo.conditional_transformer

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config:
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
        self._ideo.vae.freeze()
        self._ideo.conditional_transformer.freeze()
        self._ideo.unconditional_transformer.freeze()
        if getattr(self._ideo, "text_encoder", None) is not None:
            self._ideo.text_encoder.freeze()

    def sample_sigma(self, *, width: int, height: int, rng) -> float:  # noqa: ARG002
        # Match ai-toolkit's VALIDATED Ideogram-4 run (which produces LoRAs that DO imprint
        # identity). When `timestep_type` is unset (our CLI/default case) ai-toolkit defaults to
        # 'sigmoid' (config_modules.py:534): the per-step grid is sigmoid(randn) and 'balanced'
        # draws a uniform index into it, so the net training prior over sigma = timestep/1000 is
        # 1 - sigmoid(N(0,1)) — MID-concentrated, NOT uniform.
        # A code-level diff vs ai-toolkit found this is the single dominant divergence: our former
        # UNIFORM sampling under-sampled the mid/low-sigma band where subject-identity gradient
        # lives, which the per-sigma validator measured as -12.6% low-sigma degradation. (Earlier
        # "uniform is correct" was a misread of the UI default 'linear', not what actually ran.)
        z = rng.gauss(0.0, 1.0)
        sigma = 1.0 - 1.0 / (1.0 + math.exp(-z))  # 1 - sigmoid(randn): center/mid-peaked
        return float(min(max(sigma, 1e-3), 1.0 - 1e-3))

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
        # 1) image -> VAE latents (1, 32, H/8, W/8) -> Ideogram packing (1, seq, 128)
        encoded = LatentCreator.encode_image(
            vae=self._ideo.vae,
            image_path=image_path,
            height=height,
            width=width,
            tiling_config=getattr(self._ideo, "tiling_config", None),
        )
        clean_latents = self._pack_latents(encoded, height=height, width=width)

        # 2) JSON caption -> normalized string -> Qwen3 embeddings
        prepared = Ideogram4Caption.prepare(prompt)
        prompt_str = prepared.prompt
        inputs = Ideogram4PromptEncoder.build_inputs(
            self._ideo.tokenizers["ideogram4"],
            [prompt_str],
            height=height,
            width=width,
        )
        llm_features = Ideogram4PromptEncoder.encode_prompt(
            prompt=prompt_str,
            width=width,
            height=height,
            inputs=inputs,
            text_encoder=self._ideo.text_encoder,
            prompt_cache={},  # fresh per item; never share/poison the cache during encode
        )

        # 3) text zero-padding the conditional transformer prepends to the image latents
        max_text = int(inputs["max_text_tokens"])
        in_channels = self._ideo.conditional_transformer.config.in_channels
        text_z_padding = mx.zeros((1, max_text, in_channels), dtype=mx.float32)

        mx.eval(clean_latents, llm_features)
        cond = {
            "llm_features": llm_features,
            "position_ids": inputs["position_ids"],
            "segment_ids": inputs["segment_ids"],
            "indicator": inputs["indicator"],
            "max_text_tokens": max_text,
            "text_z_padding": text_z_padding,
        }
        return clean_latents, cond

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config, sigma: float | None = None) -> mx.array:  # noqa: ARG002
        # Ideogram-4 parametrizes its flow by the CLEAN fraction: z_t = t*clean + (1-t)*noise,
        # so the timestep is 1 - sigma (sigma = the trainer's NOISE fraction), and the model
        # predicts the velocity toward clean (clean - noise). The generic loss expects
        # (noise - clean), so we feed t = 1 - sigma and negate the output. (Verified by an
        # empirical timestep/sign sweep: cosine(-out, noise-clean) peaks ~+0.8 at t = 1-sigma.)
        # sigma is the continuous logit-normal draw from sample_sigma; fall back to the
        # index grid only if the trainer didn't provide it.
        sig = float(sigma) if sigma is not None else float(sigmas[t])
        t_arr = mx.full((1,), 1.0 - sig, dtype=mx.float32)
        pos_z = mx.concatenate([cond["text_z_padding"], latents_t], axis=1)
        out = self._ideo.conditional_transformer(
            llm_features=cond["llm_features"],
            x=pos_z,
            t=t_arr,
            position_ids=cond["position_ids"],
            segment_ids=cond["segment_ids"],
            indicator=cond["indicator"],
        )
        return -out[:, cond["max_text_tokens"]:, :]

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
            image = self._ideo.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=self._guidance,
            )
        self._ideo.prompt_cache = {}
        return image.image

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None:  # noqa: ARG002
        weights: dict[str, mx.array] = {}
        for target in training_spec.lora_layers.targets:
            if target.blocks is not None:
                for b in target.blocks.get_blocks():
                    self._append_train_lora_weights(weights, self._ideo.conditional_transformer, target.module_path.format(block=b))
            else:
                self._append_train_lora_weights(weights, self._ideo.conditional_transformer, target.module_path)
        mx.save_safetensors(
            str(path),
            weights,
            metadata={"mflux_version": VersionUtil.get_mflux_version(), "model": training_spec.model},
        )

    def load_lora_adapter(self, *, path: str | Path) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Ideogram4LoRAMapping.get_mapping(),
            transformer=self._ideo.conditional_transformer,
            lora_paths=[str(path)],
            lora_scales=[1.0],
            role="train",
            bake_lora=False,
        )

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Ideogram4LoRAMapping.get_mapping(),
            transformer=self._ideo.conditional_transformer,
            lora_paths=[str(path)],
            lora_scales=[float(scale)],
            role="assistant",
            bake_lora=False,
        )

    def _assistant_disabled(self):
        return TrainingUtil.assistant_disabled(self._ideo.conditional_transformer)

    @staticmethod
    def _append_train_lora_weights(weights: dict[str, mx.array], transformer, module_path: str) -> None:
        train_lora = TrainingUtil.get_train_lora(transformer, module_path)
        weights[f"diffusion_model.{module_path}.lora_A.weight"] = mx.transpose(train_lora.lora_A)
        weights[f"diffusion_model.{module_path}.lora_B.weight"] = mx.transpose(train_lora.lora_B)

    @staticmethod
    def _pack_latents(encoded: mx.array, *, height: int, width: int) -> mx.array:
        # Exact inverse of Ideogram4LatentCreator.unpack_latents (geometry + latent norm):
        # unpack does z*scale+shift -> reshape(b,gh,gw,2,2,c) -> transpose(0,5,1,3,2,4) -> (b,c,gh*2,gw*2).
        b, ae_c, _, _ = encoded.shape
        grid_h, grid_w = height // 16, width // 16
        x = encoded.reshape(b, ae_c, grid_h, 2, grid_w, 2)
        x = x.transpose(0, 2, 4, 3, 5, 1)
        x = x.reshape(b, grid_h * grid_w, ae_c * 4)
        shift, scale = Ideogram4LatentCreator.get_latent_norm()
        x = (x - shift.astype(x.dtype)[None, None, :]) / scale.astype(x.dtype)[None, None, :]
        return x.astype(mx.float32)
