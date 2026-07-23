import hashlib
from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.mage_flow.latent_creator import MageFlowLatentCreator
from mflux.models.mage_flow.mage_flow_initializer import MageFlowInitializer
from mflux.models.mage_flow.model.mage_flow_text_encoder import MageFlowTextEncoder
from mflux.models.mage_flow.model.mage_flow_text_encoder.policy import (
    FilterVerdict,
    make_refusal_image,
)
from mflux.models.mage_flow.model.mage_flow_transformer import MageFlowTransformer
from mflux.models.mage_flow.model.mage_flow_vae import MageVAE
from mflux.models.mage_flow.variants.conditioning import MageFlowConditioning
from mflux.models.mage_flow.variants.edit.util import MageFlowEditUtil
from mflux.models.mage_flow.variants.pipeline_helpers import (
    make_velocity_predictor,
    resolve_generation_parameters,
    resolve_seed,
)
from mflux.models.mage_flow.weights import MageFlowWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

ImageInput = Path | str | Image.Image
ReferenceCacheKey = tuple[tuple[tuple[int, int], bytes], ...]


class MageFlowEdit(nn.Module):
    """Native MLX multi-reference image-edit pipeline for Mage-Flow Edit."""

    vae: MageVAE
    transformer: MageFlowTransformer
    text_encoder: MageFlowTextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
    ):
        super().__init__()
        MageFlowInitializer.init(
            model=self,
            model_config=model_config or ModelConfig.mage_flow_edit(),
            quantize=quantize,
            model_path=model_path,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        image_paths: ImageInput | list[ImageInput],
        num_inference_steps: int | None = None,
        height: int | None = None,
        width: int | None = None,
        max_size: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        renormalization: bool = False,
        gaussian_shading_key: int | str | None = None,
        scheduler: str = "mage_flow",
    ) -> GeneratedImage:
        seed = resolve_seed(seed)
        num_inference_steps, guidance = resolve_generation_parameters(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
        )
        references = MageFlowEditUtil.load_references(image_paths)
        width, height = MageFlowEditUtil.resolve_target_size(
            references[0],
            width=width,
            height=height,
            max_size=max_size,
        )

        raw_paths = image_paths if isinstance(image_paths, list) else [image_paths]
        metadata_paths = [path for path in raw_paths if isinstance(path, (str, Path))]
        primary_path = metadata_paths[0] if metadata_paths else None
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=primary_path,
            scheduler=scheduler,
        )
        reference_key = self._reference_cache_key(references)
        verdict = self._screen_edit(
            prompt=prompt,
            references=references,
            reference_key=reference_key,
        )
        if verdict.violates:
            print(verdict.banner())
            return self._refusal_result(
                verdict=verdict,
                config=config,
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                primary_path=primary_path,
                metadata_paths=metadata_paths,
            )

        target_latents = MageFlowLatentCreator.create_noise(
            seed=seed,
            height=config.height,
            width=config.width,
            gaussian_shading_key=gaussian_shading_key,
            dtype=ModelConfig.precision,
        )
        reference_latents = MageFlowEditUtil.encode_references(
            self.vae,
            references,
            width=config.width,
            height=config.height,
            seed=seed,
        )
        text_embeddings, text_attention_mask = self._encode_prompt_pair(
            prompt=prompt,
            negative_prompt=negative_prompt,
            references=references,
            reference_key=reference_key,
            guidance=guidance,
        )
        mx.eval(target_latents, reference_latents, text_embeddings, text_attention_mask)

        latent_height = config.height // 16
        latent_width = config.width // 16
        target_length = target_latents.shape[1]
        image_shapes = [(1, latent_height, latent_width)] * (1 + len(references))
        predict = make_velocity_predictor(
            transformer=self.transformer,
            text_embeddings=text_embeddings,
            text_attention_mask=text_attention_mask,
            image_shapes=image_shapes,
            guidance=guidance,
            target_length=target_length,
            renormalization=renormalization,
        )

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(target_latents)
        for step in config.time_steps:
            try:
                model_input = mx.concatenate([target_latents, reference_latents], axis=1)
                velocity = predict(model_input, config.scheduler.sigmas[step])
                target_latents = config.scheduler.step(
                    noise=velocity,
                    timestep=step,
                    latents=target_latents,
                    sigmas=config.scheduler.sigmas,
                )
                ctx.in_loop(step, target_latents)
                mx.eval(target_latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(step, target_latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {step + 1}/{config.num_inference_steps}"
                )
        # The predictor closes over the transformer. Release it and the final
        # evaluated denoising graph before low-RAM callbacks evict the model.
        del predict, velocity, model_input
        ctx.after_loop(target_latents)

        decoded = self.vae.decode(
            MageFlowLatentCreator.unpack_latents(
                target_latents,
                height=config.height,
                width=config.width,
            )
        )
        mx.eval(decoded)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=primary_path,
            image_paths=metadata_paths or None,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    def _encode_prompt_pair(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        references: list[Image.Image],
        reference_key: ReferenceCacheKey | None = None,
        guidance: float,
    ) -> tuple[mx.array, mx.array]:
        normalized_negative = negative_prompt if negative_prompt and negative_prompt.strip() else " "
        reference_key = reference_key or self._reference_cache_key(references)
        cache_key = (prompt, normalized_negative, guidance, reference_key)
        cached = self.prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        prompts = [normalized_negative, prompt] if guidance > 1.0 else [prompt]
        image_groups = [references] * len(prompts)
        result = MageFlowConditioning.encode_edit(
            prompts=prompts,
            images_per_prompt=image_groups,
            tokenizer=self.tokenizers["mage"],
            text_encoder=self.text_encoder,
            max_sequence_length=self.model_config.max_sequence_length or 2048,
        )
        mx.eval(*result)
        self.prompt_cache[cache_key] = result
        self.prompt_cache[prompt] = result
        return result

    def _screen_edit(
        self,
        *,
        prompt: str,
        references: list[Image.Image],
        reference_key: ReferenceCacheKey,
    ) -> FilterVerdict:
        cache = getattr(self, "policy_cache", None)
        if cache is None:
            cache = self.policy_cache = {}
        cache_key = ("edit", prompt, reference_key)
        verdict = cache.get(cache_key)
        if verdict is None:
            verdict = self.text_encoder.screen_edit(
                prompt,
                references,
                self.tokenizers["mage"],
            )
            cache[cache_key] = verdict
            mx.clear_cache()
        return verdict

    @staticmethod
    def _reference_cache_key(references: list[Image.Image]) -> ReferenceCacheKey:
        return tuple((reference.size, hashlib.sha256(reference.tobytes()).digest()) for reference in references)

    def _refusal_result(
        self,
        *,
        verdict: FilterVerdict,
        config: Config,
        seed: int,
        prompt: str,
        negative_prompt: str | None,
        primary_path: Path | str | None,
        metadata_paths: list[Path | str],
    ) -> GeneratedImage:
        return GeneratedImage(
            image=make_refusal_image(verdict, height=config.height, width=config.width),
            model_config=config.model_config,
            seed=seed,
            prompt=prompt,
            steps=config.num_inference_steps,
            guidance=config.guidance,
            precision=config.precision,
            quantization=self.bits,
            generation_time=0.0,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            height=config.height,
            width=config.width,
            image_path=primary_path,
            image_paths=metadata_paths or None,
            negative_prompt=negative_prompt,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=MageFlowWeightDefinition,
        )
