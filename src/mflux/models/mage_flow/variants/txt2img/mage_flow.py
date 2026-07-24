import mlx.core as mx
from mlx import nn

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
from mflux.models.mage_flow.variants.pipeline_helpers import (
    make_velocity_predictor,
    normalize_image_dimension,
    resolve_generation_parameters,
    resolve_seed,
)
from mflux.models.mage_flow.weights import MageFlowWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class MageFlow(nn.Module):
    """Native MLX text-to-image pipeline for all released Mage-Flow checkpoints."""

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
            model_config=model_config or ModelConfig.mage_flow(),
            quantize=quantize,
            model_path=model_path,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int = 1024,
        width: int = 1024,
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
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=normalize_image_dimension(height),
            width=normalize_image_dimension(width),
            guidance=guidance,
            scheduler=scheduler,
        )
        verdict = self._screen_prompt(prompt)
        if verdict.violates:
            print(verdict.banner())
            return self._refusal_result(
                verdict=verdict,
                config=config,
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
            )

        latents = MageFlowLatentCreator.create_noise(
            seed=seed,
            height=config.height,
            width=config.width,
            gaussian_shading_key=gaussian_shading_key,
            dtype=ModelConfig.precision,
        )
        text_embeddings, text_attention_mask = self._encode_prompt_pair(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
        )
        mx.eval(latents, text_embeddings, text_attention_mask)

        latent_height = config.height // 16
        latent_width = config.width // 16
        image_shapes = [(1, latent_height, latent_width)]
        predict = make_velocity_predictor(
            transformer=self.transformer,
            text_embeddings=text_embeddings,
            text_attention_mask=text_attention_mask,
            image_shapes=image_shapes,
            guidance=guidance,
            renormalization=renormalization,
        )

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for step in config.time_steps:
            try:
                velocity = predict(latents, config.scheduler.sigmas[step])
                latents = config.scheduler.step(
                    noise=velocity,
                    timestep=step,
                    latents=latents,
                    sigmas=config.scheduler.sigmas,
                )
                ctx.in_loop(step, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(step, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {step + 1}/{config.num_inference_steps}"
                )
        # The predictor closes over the transformer. Release the closure (and
        # the final evaluated graph) before low-RAM callbacks evict the model.
        del predict, velocity
        ctx.after_loop(latents)

        decoded = self.vae.decode(
            MageFlowLatentCreator.unpack_latents(
                latents,
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
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    def _encode_prompt_pair(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array]:
        normalized_negative = negative_prompt if negative_prompt and negative_prompt.strip() else " "
        cache_key = (prompt, normalized_negative, guidance)
        cached = self.prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        prompts = [normalized_negative, prompt] if guidance > 1.0 else [prompt]
        result = MageFlowConditioning.encode_text_to_image(
            prompts=prompts,
            tokenizer=self.tokenizers["mage"],
            text_encoder=self.text_encoder,
            max_sequence_length=self.model_config.max_sequence_length or 2048,
        )
        mx.eval(*result)
        self.prompt_cache[cache_key] = result
        # MemorySaver uses the positive prompt as its generic "safe to evict"
        # signal. The exact tuple remains the lookup key for correctness.
        self.prompt_cache[prompt] = result
        return result

    def _screen_prompt(self, prompt: str) -> FilterVerdict:
        cache = getattr(self, "policy_cache", None)
        if cache is None:
            cache = self.policy_cache = {}
        cache_key = ("text", prompt)
        verdict = cache.get(cache_key)
        if verdict is None:
            verdict = self.text_encoder.screen_text(prompt, self.tokenizers["mage"])
            cache[cache_key] = verdict
            # The autoregressive KV cache is no longer live; release its Metal
            # buffers before the diffusion conditioning pass.
            mx.clear_cache()
        return verdict

    def _refusal_result(
        self,
        *,
        verdict: FilterVerdict,
        config: Config,
        seed: int,
        prompt: str,
        negative_prompt: str | None,
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
            negative_prompt=negative_prompt,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=MageFlowWeightDefinition,
        )
