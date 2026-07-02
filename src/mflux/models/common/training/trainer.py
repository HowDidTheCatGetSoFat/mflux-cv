from __future__ import annotations

import gc
import math
import random
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.optimizers import clip_grad_norm
from mlx.utils import tree_map, tree_unflatten
from PIL import Image as PILImage
from tqdm import tqdm

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.dataset.batch import Batch, DataItem
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.state.training_state import TrainingState
from mflux.models.common.training.statistics.plotter import Plotter
from mflux.models.common.training.utils import TrainingUtil


class TrainingTrainer:
    @staticmethod
    def _sample_timestep_index(timestep_type: str, low: int, high: int, rng) -> int:
        # Map a [0,1) draw shaped by the distribution to a timestep index in [low, high).
        import math

        span = high - low
        if span <= 0:
            return low
        if timestep_type == "sigmoid":  # mid-concentrated (ai-toolkit default; best for identity)
            frac = 1.0 / (1.0 + math.exp(-rng.gauss(0.0, 1.0)))
        elif timestep_type == "content":  # cubic, favors low noise (fine detail)
            frac = rng.random() ** 3
        elif timestep_type == "style":  # favors high noise (coarse style)
            frac = 1.0 - rng.random() ** 3
        else:
            frac = rng.random()
        return min(max(low + int(frac * span), low), high - 1)

    @staticmethod
    def compute_loss(
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        base_config,
        batch: Batch,
    ) -> mx.float16:
        losses = [
            TrainingTrainer._single_example_loss(adapter, training_spec, base_config, item, batch.rng)
            for item in batch.data
        ]
        return mx.mean(mx.array(losses))

    @staticmethod
    def _single_example_loss(
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        base_config,
        item: DataItem,
        rng: random.Random,
    ) -> mx.float16:
        # Create a config matching this item's spatial dimensions.
        # Flux uses config.width/height for rotary embeddings, so this must match the latent layout.
        config = adapter.create_config(training_spec, width=item.width, height=item.height)

        # Reuse the base scheduler only when compatible with the item's dimensions.
        # Some schedulers depend on image seq len when sigma shift is enabled.
        if not config.model_config.requires_sigma_shift or config.image_seq_len == base_config.image_seq_len:
            config._scheduler = base_config.scheduler  # type: ignore[attr-defined]
        else:
            _ = config.scheduler

        time_seed = rng.randint(0, 2**32 - 1)
        noise_seed = rng.randint(0, 2**32 - 1)

        low = int(training_spec.training_loop.timestep_low)
        high = int(
            config.num_inference_steps
            if training_spec.training_loop.timestep_high is None
            else training_spec.training_loop.timestep_high
        )

        timestep_type = training_spec.training_loop.timestep_type
        if timestep_type and timestep_type != "uniform":
            # Non-uniform timestep-index sampling (sigmoid/content/style). Identity learning
            # lives in the mid/low-noise band that flat sampling under-weights.
            t = TrainingTrainer._sample_timestep_index(timestep_type, low, high, rng)
        else:
            t = int(
                mx.random.randint(
                    low=low,
                    high=high,
                    shape=[],
                    key=mx.random.key(time_seed),
                )
            )

        clean_image = item.clean_latents
        pure_noise = mx.random.normal(
            shape=clean_image.shape,
            dtype=config.precision,
            key=mx.random.key(noise_seed),
        )

        # Noise level: an adapter may define its own training timestep distribution
        # (e.g. Ideogram-4's resolution-shifted logit-normal schedule — matching how the
        # model was trained and how it samples at inference). Otherwise fall back to the
        # uniform index into config.scheduler.sigmas (flux / z-image behavior).
        if hasattr(adapter, "sample_sigma"):
            sigma = float(adapter.sample_sigma(width=config.width, height=config.height, rng=rng))
        else:
            sigma = float(config.scheduler.sigmas[t])

        latents_t = LatentCreator.add_noise_by_interpolation(
            clean=clean_image,
            noise=pure_noise,
            sigma=sigma,
        )

        # Caption dropout: with probability caption_dropout_rate, swap this example's text condition
        # for the precomputed empty-caption condition, so the model also learns unconditional
        # generation. Decided per example per step (stochastic via rng).
        cond = item.cond
        dropout_rate = training_spec.training_loop.caption_dropout_rate
        null_cond = getattr(adapter, "_caption_dropout_null_cond", None)
        if null_cond is not None and dropout_rate > 0 and rng.random() < dropout_rate:
            cond = null_cond

        predicted_noise = adapter.predict_noise(
            t=t,
            latents_t=latents_t,
            sigmas=config.scheduler.sigmas,
            cond=cond,
            config=config,
            sigma=sigma,
        )

        error = (clean_image + predicted_noise - pure_noise).square()
        # Masked loss: when a per-example mask is attached, weight the loss by it so training focuses
        # on the masked region (e.g. a subject/face) rather than the whole frame. Broadcast the mask
        # over the channel axis before normalizing so the result stays on the same scale as .mean()
        # (an all-ones mask reproduces the plain mean exactly).
        if item.mask is not None:
            weights = mx.broadcast_to(item.mask, error.shape)
            loss = (error * weights).sum() / weights.sum()
        else:
            loss = error.mean()
        # Prior preservation: scale regularization images' loss by reg_weight relative to subject images.
        if item.is_reg:
            loss = loss * training_spec.training_loop.reg_weight
        return loss

    @staticmethod
    def train(
        *,
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ) -> None:
        first_preview = None
        if training_spec.monitoring is not None and training_spec.monitoring.preview_images:
            first_preview = training_spec.monitoring.preview_images[0]
        preview_width, preview_height = TrainingTrainer._preview_dimensions(training_spec, preview_image=first_preview)
        base_config = adapter.create_config(training_spec, width=preview_width, height=preview_height)
        # Ensure scheduler is initialized once and can be reused in per-item configs.
        _ = base_config.scheduler

        # Freeze base weights and unfreeze LoRA weights
        adapter.freeze_base()
        TrainingTrainer._unfreeze_lora_layers(adapter.transformer())

        # EMA of the trained (LoRA) weights (optional): a shadow copy taken after unfreezing,
        # updated after each optimizer step, and swapped in at save time. None when ema_decay is off.
        ema_decay = training_spec.training_loop.ema_decay
        ema = TrainingTrainer._init_ema(adapter, ema_decay)

        # With previews off, the encode/preview-only submodules (text encoder, VAE, CFG
        # transformer) are dead weight for the whole loop — the dataset is already encoded.
        # Release them to the OS (~16GB on Ideogram-4's Qwen3-VL alone); zero train-speed cost.
        previews_off = training_spec.monitoring is None or not training_spec.monitoring.preview_prompts
        if previews_off:
            release = getattr(adapter, "release_encoders", None)
            if release is not None:
                release()

        train_step_function = nn.value_and_grad(
            model=adapter.model(),
            fn=lambda b: TrainingTrainer.compute_loss(adapter, training_spec, base_config, b),
        )

        if training_spec.monitoring is not None and training_state.iterator.num_iterations == 0:
            TrainingTrainer._generate_previews_with_optimizer_offload(adapter, training_spec, training_state)
            validation_batch = training_state.iterator.get_validation_batch()
            validation_loss = TrainingTrainer.compute_loss(adapter, training_spec, base_config, validation_batch)
            training_state.statistics.append_values(step=training_state.iterator.num_iterations, loss=float(validation_loss))  # fmt: off
            Plotter.update_loss_plot(training_spec=training_spec, training_state=training_state)
            del validation_loss
            TrainingTrainer._save_checkpoint(training_state, adapter, training_spec, ema)

        batches = tqdm(
            training_state.iterator,
            total=training_state.iterator.total_number_of_steps(),
            initial=training_state.iterator.num_iterations,
        )

        max_grad_norm = training_spec.optimizer.max_grad_norm
        accum_steps = max(1, training_spec.optimizer.gradient_accumulation_steps)
        accumulated_grads = None
        nonfinite_skips = 0
        for batch in batches:
            loss, grads = train_step_function(batch)
            # The training loss is already computed here (with the gradients); capture it for the
            # plot instead of paying for a separate forward pass later. float() also forces the
            # single scalar eval needed for the non-finite check below.
            train_loss = float(loss)
            del loss
            # Skip non-finite steps (bf16 activation spikes / NaN loss) so a single bad batch
            # can't poison the LoRA weights or optimizer state; the run continues on the last
            # good state. clip_grad_norm handles ordinary spikes; this catches Inf/NaN.
            if not math.isfinite(train_loss):
                del grads
                nonfinite_skips += 1
                if training_spec.low_ram:
                    mx.clear_cache()
                continue

            # Gradient accumulation: average grads across accum_steps micro-batches and only step
            # the optimizer on the window boundary, for an effective batch of batch_size *
            # accum_steps. num_iterations counts micro-batches, so the boundary is every
            # accum_steps of them; bookkeeping below still runs each iteration on valid weights.
            at_step_boundary = training_state.iterator.num_iterations % accum_steps == 0
            if accum_steps > 1:
                grads = tree_map(lambda g: g / accum_steps, grads)
                if accumulated_grads is not None:
                    grads = tree_map(lambda a, g: a + g, accumulated_grads, grads)
                accumulated_grads = None if at_step_boundary else grads

            if accum_steps == 1 or at_step_boundary:
                if max_grad_norm is not None:
                    grads, _ = clip_grad_norm(grads, max_grad_norm)
                training_state.optimizer.optimizer.update(model=adapter.model(), gradients=grads)
                mx.eval(adapter.model().parameters(), training_state.optimizer.optimizer.state)
                ema = TrainingTrainer._update_ema(ema, ema_decay, adapter)
            else:
                # Keep the partial sum materialized so the graph doesn't grow across the window.
                mx.eval(accumulated_grads)
            del grads

            if training_state.should_plot_loss(training_spec):
                # Plot the already-computed training loss (free) instead of a separate forward
                # pass over the same samples every plot step (there is no held-out validation
                # set); that recompute was the dominant per-step cost on larger models.
                training_state.statistics.append_values(step=training_state.iterator.num_iterations, loss=train_loss)  # fmt: off
                Plotter.update_loss_plot(training_spec=training_spec, training_state=training_state)

            if training_state.should_generate_image(training_spec):
                TrainingTrainer._generate_previews_with_optimizer_offload(adapter, training_spec, training_state)

            if training_state.should_save(training_spec):
                TrainingTrainer._save_checkpoint(training_state, adapter, training_spec, ema)

            if training_spec.low_ram:
                mx.clear_cache()

        if nonfinite_skips:
            print(f"Skipped {nonfinite_skips} non-finite (NaN/Inf) training step(s).")
        TrainingTrainer._save_checkpoint(training_state, adapter, training_spec, ema)

    @staticmethod
    def _init_ema(adapter, ema_decay):
        # Shadow copy of the trainable (LoRA) params, or None when EMA is off.
        if not ema_decay:
            return None
        return tree_map(lambda p: mx.array(p), adapter.model().trainable_parameters())

    @staticmethod
    def _update_ema(ema, ema_decay, adapter):
        # ema = decay*ema + (1-decay)*live, over the trainable params.
        if ema is None:
            return None
        updated = tree_map(
            lambda e, p: ema_decay * e + (1.0 - ema_decay) * p,
            ema,
            adapter.model().trainable_parameters(),
        )
        mx.eval(updated)
        return updated

    @staticmethod
    def _save_checkpoint(training_state, adapter, training_spec, ema) -> None:
        # Save the checkpoint. With EMA enabled, swap the EMA weights into the model for the save and
        # restore the live training weights afterwards so training continues from the live weights.
        if ema is None:
            training_state.save(adapter, training_spec)
            return
        model = adapter.model()
        live = model.trainable_parameters()
        model.update(ema)
        try:
            training_state.save(adapter, training_spec)
        finally:
            model.update(live)
            mx.eval(model.trainable_parameters())

    @staticmethod
    def _unfreeze_lora_layers(module: nn.Module) -> None:
        for _, child in module.named_modules():
            if isinstance(child, LoRALinear):
                if getattr(child, "_mflux_lora_role", None) == "train":
                    child.unfreeze(keys=["lora_A", "lora_B", "dora_scale"], strict=False)
            elif isinstance(child, FusedLoRALinear):
                for lora in child.loras:
                    if getattr(lora, "_mflux_lora_role", None) == "train":
                        lora.unfreeze(keys=["lora_A", "lora_B", "dora_scale"], strict=False)

    @staticmethod
    def _preview_dimensions(training_spec: TrainingSpec, *, preview_image: Path | None = None) -> tuple[int, int]:
        if training_spec.monitoring is None:
            return 1024, 1024
        if preview_image is not None:
            with PILImage.open(preview_image) as img:
                width, height = img.size
        else:
            width = int(training_spec.monitoring.preview_width)
            height = int(training_spec.monitoring.preview_height)

        return TrainingUtil.resolve_dimensions(
            width=width,
            height=height,
            max_resolution=None,
            error_template=(
                f"Preview image too small for training (needs >=16px). Got {{width}}x{{height}} from {preview_image}"
            ),
        )

    @staticmethod
    def _generate_previews(
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ) -> None:
        if training_spec.monitoring is None:
            return
        preview_prompts = training_spec.monitoring.preview_prompts
        preview_names = training_spec.monitoring.preview_prompt_names
        preview_images = training_spec.monitoring.preview_images
        for idx, prompt in enumerate(preview_prompts):
            image_paths = None
            if training_spec.is_edit:
                if not preview_images or idx >= len(preview_images):
                    raise ValueError("Edit training requires data/preview.* for each preview prompt.")
                image_paths = [preview_images[idx]]
                preview_width, preview_height = TrainingTrainer._preview_dimensions(
                    training_spec, preview_image=preview_images[idx]
                )
            else:
                preview_width, preview_height = TrainingTrainer._preview_dimensions(training_spec)
            image = adapter.generate_preview_image(
                seed=training_spec.seed,
                prompt=prompt,
                width=preview_width,
                height=preview_height,
                steps=training_spec.steps,
                image_paths=image_paths,
            )
            preview_name = preview_names[idx] if idx < len(preview_names) else None
            image.save(
                training_state.get_current_preview_image_path(
                    training_spec,
                    preview_index=idx,
                    preview_name=preview_name,
                )
            )
            del image

    @staticmethod
    def _generate_previews_with_optimizer_offload(
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ) -> None:
        optimizer = training_state.optimizer
        with tempfile.TemporaryDirectory() as tmp_dir:
            offload_path = Path(tmp_dir) / "optimizer_offload.safetensors"
            optimizer.save(offload_path)
            optimizer.optimizer.state = []

            gc.collect()
            mx.clear_cache()
            try:
                TrainingTrainer._generate_previews(adapter, training_spec, training_state)
            finally:
                restored_state = tree_unflatten(list(mx.load(str(offload_path)).items()))
                optimizer.optimizer.state = restored_state
                gc.collect()
                mx.clear_cache()
