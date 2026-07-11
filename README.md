# mflux-CV

A drop-in community build of [mflux](https://github.com/filipstrand/mflux) by Filip Strand (MIT). It
stays current with upstream `main` and adds a curated set of fixes, training features, and useful
community PRs, with prebuilt wheels on GitHub Releases so improvements stay available to everyone even
when upstream maintenance is slow. The Python package is still `mflux` and every CLI keeps its name,
so this is a direct replacement in ComfyUI, image-studio, or any existing setup.

**Install**

```bash
pip install git+https://github.com/HowDidTheCatGetSoFat/mflux-cv.git@v.0.18.15-CV
```

or download a wheel from [Releases](https://github.com/HowDidTheCatGetSoFat/mflux-cv/releases).

**Relationship to upstream:** kept rebased on `filipstrand/mflux` so changes merge back cleanly. Every
upstream commit is included and nothing is removed. Credit for the base project and every pulled PR
goes to their authors.

## Changelog (on top of upstream 0.18.0)

### 0.18.16-CV
- Pulled in two upstream bug fixes (credited under [Community PRs pulled in](#community-prs-pulled-in)):
  fused-qkv LoRA loading (#459) and ERNIE / Krea 2 tiled img2img (#463).
- Repo automation: CodeQL security scanning, structured issue forms, and PR / issue auto-labeling.

### 0.18.15-CV
First release under the `mflux-CV` name, in the community home at
[HowDidTheCatGetSoFat/mflux-cv](https://github.com/HowDidTheCatGetSoFat/mflux-cv). Same codebase as the
prior `+fxd0h` builds (0.18.1 through 0.18.5); this is the rebrand plus everything listed below.

### 0.18.5
- **Krea 2 `--krea2-uncensor <k>`**: scales the text-fusion projector's refusal layers (tapped Qwen3-VL
  9/10/11) so explicit prompts render instead of being dodged. `k=1` is off, `~6` neutralises the filter.

### 0.18.4
- **LoRA on quantized bases**: keep the adapter live at inference instead of baking it into the quantized
  weights (baking re-quantized and badly diverged the output on the `--quantize` default).
- **LoKr**: load LyCORIS LoKr adapters for Krea 2, Qwen, and Ideogram 4.
- **Ideogram 4**: fix the stepwise-preview VAE decode crash on already-unpacked latents; guard the
  injected-LoRA scan against transformers without `named_modules`.
- **Krea 2 Raw**: download the diffusers transformer from HuggingFace.
- **z-image**: shared `--saveinfo` filename builder; fix numeric-tag collisions.
- **qwen-edit**: clearer error on empty `image_paths`; regenerated golden references.

### 0.18.3
- Review fixes: fp8-aware fused DoRA, training guards (LR, grad-accum reset on skip, qwen VAE flag),
  route the CFG negative through an injected LoRA in training previews, surface LoRA bake failures on
  save, EMA resume from live weights.

### 0.18.2
- **Krea 2 sigma schedule**: use the official dynamic exponential shift instead of a linear 1.15.

### 0.18.1
- **Training suite**: DoRA (weight-decomposed LoRA) for Krea 2, Ideogram 4, z-image, flux, flux2;
  gradient accumulation; EMA of trained weights; caption dropout; masked loss; regularization /
  prior-preservation images; continue training from an existing LoRA; non-finite-step guard; utf-8-safe
  captions; free training-loss plot.
- **Krea 2**: LoRA training, Raw variant, and diffusers-format loading.

### Community PRs pulled in
- **[filipstrand/mflux#459](https://github.com/filipstrand/mflux/pull/459) by Sahil Tanveer** — fix LoRA
  loading for fused qkv layers: keep the shared rank/down projection whole and slice only the up
  projection, so kohya/BFL FLUX LoRAs with a rank divisible by 3/4 load correctly.
- **[filipstrand/mflux#463](https://github.com/filipstrand/mflux/pull/463) by Mike Wallio** — fix ERNIE
  and Krea 2 img2img with tiled VAE latents: the 5D tiled-VAE pack path took the wrong slice; keep the
  singleton temporal axis so tiled-decode img2img reconstructs correctly.

---

> The rest of this file is the upstream mflux documentation.

![image](src/mflux/assets/logo.jpg)

[![MFLUX](https://img.shields.io/pypi/v/mflux?label=MFLUX&logo=pypi&logoColor=white)](https://pypi.org/project/mflux/)
[![MLX](https://img.shields.io/pypi/v/mlx?label=MLX&logo=pypi&logoColor=white)](https://pypi.org/project/mlx/)
[![CI](https://github.com/filipstrand/mflux/actions/workflows/tests.yml/badge.svg)](https://github.com/filipstrand/mflux/actions/workflows/tests.yml)

### About

Run the latest state-of-the-art generative image models locally on your Mac in native MLX!

### Table of contents

- [💡 Philosophy](#-philosophy)
- [💿 Installation](#-installation)
- [🎨 Models](#-models)
- [✨ Features](#-features)
- [🌱 Related projects](#related-projects)
- [🙏 Acknowledgements](#-acknowledgements)
- [⚖️ License](#%EF%B8%8F-license)

---

### 💡 Philosophy

MFLUX is a line-by-line MLX port of several state-of-the-art generative image models from the [Huggingface Diffusers](https://github.com/huggingface/diffusers) and [Huggingface Transformers](https://github.com/huggingface/transformers) libraries. All models are implemented from scratch in MLX, using only tokenizers from the [Huggingface Transformers](https://github.com/huggingface/transformers) library. MFLUX is purposefully kept minimal and explicit, [@karpathy](https://gist.github.com/awni/a67d16d50f0f492d94a10418e0592bde?permalink_comment_id=5153531#gistcomment-5153531) style.

---

### 💿 Installation
If you haven't already, [install `uv`](https://github.com/astral-sh/uv?tab=readme-ov-file#installation), then run:

```sh
uv tool install --upgrade mflux
```

After installation, the following command shows all available MFLUX CLI commands: 

```sh
uv tool list 
```

To generate your first image using, for example, the z-image-turbo model, run

```
mflux-generate-z-image-turbo \
  --prompt "A puffin standing on a cliff" \
  --width 1280 \
  --height 500 \
  --seed 42 \
  --steps 9 \
  -q 8
```

![Puffin](src/mflux/assets/puffin.png)

The first time you run this, the model will automatically download which can take some time. See the [model section](#-models) for the different options and features, and the [common README](src/mflux/models/common/README.md) for shared CLI patterns and examples.

<details>
<summary>Python API</summary>

Create a standalone `generate.py` script with inline `uv` dependencies:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mflux",
# ]
# ///
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(quantize=8)
image = model.generate_image(
    prompt="A puffin standing on a cliff",
    seed=42,
    num_inference_steps=9,
    width=1280,
    height=500,
)
image.save("puffin.png")
```

Run it with:

```sh
uv run generate.py
```

For more Python API inspiration, look at the [CLI entry points](src/mflux/models/z_image/cli/z_image_turbo_generate.py) for the respective models.
</details>

<details>
<summary>⚠️ Troubleshooting: hf_transfer error</summary>

If you encounter a `ValueError: Fast download using 'hf_transfer' is enabled (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available`, you can install MFLUX with the `hf_transfer` package included:

```sh
uv tool install --upgrade mflux --with hf_transfer
```

This will enable faster model downloads from Hugging Face.

</details>

<details>
<summary>DGX / NVIDIA (uv tool install)</summary>

```sh
uv tool install --python 3.13 mflux
```
</details>

---

### 🎨 Models

MFLUX supports the following model families. They have different strengths and weaknesses; see each model’s README for full usage details.

| Model | Release date | Size | Type | Training | Description |
| --- | --- | --- | --- | --- | --- |
|[Z-Image](src/mflux/models/z_image/README.md) | Nov 2025 | 6B | Distilled & Base | Yes | Fast, small, very good quality and realism. |
|[Krea 2](src/mflux/models/krea2/README.md) | Jun 2026 | 12B | Turbo (distilled) | No | Very good quality with a wide range of styles; good for creative exploration. |
|[FLUX.2](src/mflux/models/flux2/README.md) | Jan 2026 | 4B & 9B | Distilled & Base | Yes | Fastest + smallest with very good qaility and edit capabilities. |
|[Ideogram 4](src/mflux/models/ideogram4/README.md) | Jun 2026 | 9B | Base | No | JSON-caption-native, typography-focused text-to-image generation. |
|[ERNIE-Image](src/mflux/models/ernie_image/README.md) | Apr 2026 | 8B | Distilled & Base | No | Single-stream DiT from Baidu. Vivid, high-contrast output. |
|[FIBO](src/mflux/models/fibo/README.md) | Oct 2025+ | 8B | Distilled & Base | No | Very good JSON-based prompt understanding. Has edit capabilities. |
|[SeedVR2](src/mflux/models/seedvr2/README.md) | Jun 2025 | 3B & 7B | — | No | Best upscaling model. |
|[Qwen Image](src/mflux/models/qwen/README.md) | Aug 2025+ | 20B | Base | No | Large model (slower); strong prompt understanding and world knowledge. Has edit capabilities |
|[Depth Pro](src/mflux/models/depth_pro/README.md) | Oct 2024 | — | — | No | Very fast and accurate depth estimation model from Apple. |
|[FLUX.1](src/mflux/models/flux/README.md) | Aug 2024 | 12B | Distilled & Base | No (legacy) | Legacy option with decent quality. Has edit capabilities with 'Kontext' model and upscaling support via ControlNet |

---

### ✨ Features

**General**
- Quantization and local model loading
- LoRA support (multi-LoRA, scales, library lookup), including LyCORIS LoKr on FLUX.1 and FLUX.2
- Metadata export + reuse, plus prompt file support

**Model-specific highlights**
- Text-to-image and image-to-image generation.
- LoRA finetuning
- In-context editing, multi-image editing, and virtual try-on
- ControlNet (Canny), depth conditioning, fill/inpainting, and Redux
- Upscaling (SeedVR2 and Flux ControlNet)
- Depth map extraction and FIBO prompt tooling (VLM inspire/refine)

See the [common README](src/mflux/models/common/README.md) for detailed usage and examples, and use the model section above to browse specific models and capabilities.

> [!NOTE]
> As MFLUX supports a wide variety of CLI tools and options, the easiest way to navigate the CLI in 2026 is to use a coding agent (like [Cursor](https://cursor.com), [Claude Code](https://www.anthropic.com/claude-code), or similar). Ask questions like: “Can you help me generate an image using z-image?”


---

<a id="related-projects"></a>

### 🌱 Related projects

- [MindCraft Studio](https://themindstudio.cc/mindcraft#models) — macOS app built on mflux by [@shaoju](https://github.com/shaoju)
- [Mflux-ComfyUI](https://github.com/raysers/Mflux-ComfyUI) by [@raysers](https://github.com/raysers)
- [MFLUX-WEBUI](https://github.com/CharafChnioune/MFLUX-WEBUI) by [@CharafChnioune](https://github.com/CharafChnioune)
- [mflux-fasthtml](https://github.com/anthonywu/mflux-fasthtml) by [@anthonywu](https://github.com/anthonywu)
- [mflux-streamlit](https://github.com/elitexp/mflux-streamlit) by [@elitexp](https://github.com/elitexp)
- [mlx-taef](https://github.com/IonDen/mlx-taef) — TAESD/TAEF tiny-autoencoder live previews and low-memory FLUX decode for mflux, by [@IonDen](https://github.com/IonDen)
- [mlx-teacache](https://github.com/IonDen/mlx-teacache) — TeaCache step-skipping to speed up FLUX generation in mflux, by [@IonDen](https://github.com/IonDen)

---

### 🙏 Acknowledgements

MFLUX would not be possible without the great work of:

- The MLX Team for [MLX](https://github.com/ml-explore/mlx) and [MLX examples](https://github.com/ml-explore/mlx-examples)
- Black Forest Labs for the [FLUX project](https://github.com/black-forest-labs/flux)
- Bria for the [FIBO project](https://huggingface.co/briaai/FIBO)
- Tongyi Lab for the [Z-Image project](https://tongyi-mai.github.io/Z-Image-blog/)
- Baidu for the [ERNIE-Image project](https://huggingface.co/baidu/ERNIE-Image)
- Ideogram for the [Ideogram 4 project](https://huggingface.co/ideogram-ai/ideogram-4-fp8)
- Krea.ai for the [Krea 2 project](https://www.krea.ai/blog/krea-2-technical-report)
- Qwen Team for the [Qwen Image project](https://qwen.ai/blog?id=a6f483777144685d33cd3d2af95136fcbeb57652&from=research.research-list)
- ByteDance, @numz and @adrientoupet for the [SeedVR2 project](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
- Hugging Face for the [Diffusers library implementations](https://github.com/huggingface/diffusers) 
- Depth Pro authors for the [Depth Pro model](https://github.com/apple/ml-depth-pro?tab=readme-ov-file#citation)
- The MLX community and all [contributors and testers](https://github.com/filipstrand/mflux/graphs/contributors)

---

### ⚖️ License

This project is licensed under the [MIT License](LICENSE).
