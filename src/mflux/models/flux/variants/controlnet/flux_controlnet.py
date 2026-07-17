import math

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux.flux_initializer import FluxInitializer
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.flux.variants.controlnet.controlnet_util import ControlnetUtil
from mflux.models.flux.variants.controlnet.transformer_controlnet import TransformerControlnet
from mflux.models.flux.weights.flux_weight_definition import FluxControlnetWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil, StrOrBytesPath
from mflux.utils.metadata_reader import MetadataReader


class Flux1Controlnet(nn.Module):
    vae: VAE
    transformer: Transformer
    transformer_controlnets: list[TransformerControlnet]
    t5_text_encoder: T5Encoder
    clip_text_encoder: CLIPEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        controlnet_path: str | None = None,
        controlnet_paths: list[str] | None = None,
        model_config: ModelConfig = ModelConfig.dev_controlnet_canny(),
    ):
        super().__init__()
        # Stack several controlnets by passing controlnet_paths (each is a separate checkpoint, e.g.
        # depth + canny). controlnet_path stays as the single-checkpoint form. With neither, the
        # controlnet named by the model config is used.
        sources = controlnet_paths or ([controlnet_path] if controlnet_path else None)
        FluxInitializer.init_controlnet(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            model_config=model_config,
            controlnet_paths=sources,
        )

    @property
    def transformer_controlnet(self) -> TransformerControlnet | None:
        """The first (or only) controlnet. Kept so single-controlnet callers and the model saver
        keep working now that the nets are stored as a list."""
        nets = getattr(self, "transformer_controlnets", None)
        return nets[0] if nets else None

    @staticmethod
    def _broadcast_samples(samples: list[mx.array], num_blocks: int) -> list[mx.array] | None:
        """Spread one controlnet's residuals over the transformer's blocks using the same rule the
        transformer itself applies, so nets with different block counts become directly summable."""
        if not samples:
            return None
        interval_control = int(math.ceil(num_blocks / len(samples)))
        return [samples[idx // interval_control] for idx in range(num_blocks)]

    @staticmethod
    def _combine_samples(per_net_samples: list[list[mx.array]], num_blocks: int) -> list[mx.array]:
        """Sum the residuals of every stacked controlnet, block by block. Returns a list as long as
        the transformer's block list (so it indexes 1:1 there), or empty when no net contributed."""
        expanded = [e for e in (Flux1Controlnet._broadcast_samples(s, num_blocks) for s in per_net_samples) if e is not None]  # fmt: off
        if not expanded:
            return []
        total = expanded[0]
        for other in expanded[1:]:
            total = [a + b for a, b in zip(total, other)]
        return total

    def generate_image(
        self,
        seed: int,
        prompt: str,
        controlnet_image_path: StrOrBytesPath | list[StrOrBytesPath],
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        controlnet_strength: float | list[float] = 1.0,
        scheduler: str = "linear",
    ) -> GeneratedImage:
        # 0. Normalize the per-controlnet inputs. A single path/strength keeps the original
        #    single-controlnet call shape; lists drive the stacked controlnets loaded at init.
        nets = self.transformer_controlnets
        image_paths = list(controlnet_image_path) if isinstance(controlnet_image_path, (list, tuple)) else [controlnet_image_path]  # fmt: off
        if isinstance(controlnet_strength, (list, tuple)):
            strengths = [float(s) for s in controlnet_strength]
        else:
            strengths = [float(controlnet_strength)] * len(image_paths)

        if len(image_paths) != len(nets):
            raise ValueError(
                f"Got {len(image_paths)} controlnet image(s) but {len(nets)} controlnet(s) are loaded. "
                f"Pass one image per controlnet (and load them with controlnet_paths)."
            )
        if len(strengths) != len(nets):
            raise ValueError(
                f"Got {len(strengths)} controlnet strength(s) for {len(nets)} controlnet(s). "
                f"Pass one strength per controlnet, or a single value for all."
            )

        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            controlnet_strength=strengths[0],
        )

        # 1. Encode each controlnet reference image. The canny preprocessing is a property of the
        #    model config and can only describe one controlnet, so it is applied for the single-net
        #    case; a stack expects control images that are already in each net's input form.
        conditions, canny_images = [], []
        for path in image_paths:
            condition, canny_image = ControlnetUtil.encode_image(
                vae=self.vae,
                width=config.width,
                height=config.height,
                controlnet_image_path=path,
                is_canny=self.model_config.is_canny() and len(nets) == 1,
            )
            conditions.append(condition)
            canny_images.append(canny_image)
        canny_image = canny_images[0]

        # 2. Create the initial latents
        latents = FluxLatentCreator.create_noise(
            seed=seed,
            width=config.width,
            height=config.height,
        )

        # 3. Encode the prompt
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            t5_tokenizer=self.tokenizers["t5"],
            clip_tokenizer=self.tokenizers["clip"],
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
        )

        # 4. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents, canny_image=canny_image)

        for t in config.time_steps:
            try:
                # Scale model input if needed by the scheduler
                latents = config.scheduler.scale_model_input(latents, t)

                # 5.t Compute the controlnet samples of every stacked net, then sum them per block.
                per_net_blocks, per_net_single_blocks = [], []
                for net, condition, strength in zip(nets, conditions, strengths):
                    block_samples, single_block_samples = net(
                        t=t,
                        config=config,
                        hidden_states=latents,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        controlnet_condition=condition,
                        controlnet_strength=strength,
                    )
                    per_net_blocks.append(block_samples)
                    per_net_single_blocks.append(single_block_samples)

                controlnet_block_samples = Flux1Controlnet._combine_samples(
                    per_net_blocks, len(self.transformer.transformer_blocks)
                )
                controlnet_single_block_samples = Flux1Controlnet._combine_samples(
                    per_net_single_blocks, len(self.transformer.single_transformer_blocks)
                )

                # 6.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                )

                # 7.t Take one denoise step
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)

                # 8.t Call subscribers in-loop
                ctx.in_loop(t, latents)

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # 9. Call subscribers after loop
        ctx.after_loop(latents)

        # 10. Decode the latent array and return the image
        latents = FluxLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)

        # 11. Read metadata from the (first) controlnet image if available
        primary_image_path = image_paths[0]
        init_metadata = MetadataReader.read_all_metadata(primary_image_path) if primary_image_path else None

        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            controlnet_image_path=primary_image_path,
            generation_time=config.time_steps.format_dict["elapsed"],
            init_metadata=init_metadata,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=FluxControlnetWeightDefinition,
        )
