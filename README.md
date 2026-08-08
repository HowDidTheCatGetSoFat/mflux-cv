# mflux-CV

A drop-in community build of [mflux](https://github.com/filipstrand/mflux) by Filip Strand (MIT). It
stays current with upstream `main` and adds a curated set of fixes, training features, and useful
community PRs, with prebuilt wheels on GitHub Releases so improvements are easy to pick up between
upstream releases. The Python package is still `mflux` and every CLI keeps its name,
so this is a direct replacement in ComfyUI, image-studio, or any existing setup.

**Install**

```bash
pip install mflux-cv
```

To track a specific tag instead, or to pick up work that has not been released yet:

```bash
pip install git+https://github.com/HowDidTheCatGetSoFat/mflux-cv.git@v.0.18.38-CV
```

Wheels are also attached to every [Release](https://github.com/HowDidTheCatGetSoFat/mflux-cv/releases).

> **Install this or upstream `mflux`, never both.** The distribution is named `mflux-cv` so it can
> live on PyPI alongside upstream, though both provide the same `mflux` import package. With both
> installed, pip raises no error while the two overwrite each other's files, and uninstalling either
> one leaves the other broken. Run `pip uninstall mflux` before installing this, and
> `pip uninstall mflux-cv` before going back.

**Relationship to upstream:** kept rebased on `filipstrand/mflux` so changes merge back cleanly. Every
upstream commit is included and nothing is removed. Credit for the base project and every pulled PR
goes to their authors.

## Changelog (on top of upstream 0.18.0)

### 0.18.38-CV

- **Fix: Qwen-Image 4-bit accumulated quantization noise across steps** (upstream
  [#484](https://github.com/filipstrand/mflux/issues/484)): more steps made 4-bit output grainier,
  not better (flat-field sigma 5.06 to 16.07 from 4 to 50 steps). The whole effect traced to the
  adaLN modulation layers; `-q 4` now keeps `img_mod_linear` at 8-bit (~1.8 GB), restoring the
  full-q8 noise floor (sigma 1.10/1.37). Edit and Flash inherit it; mixed saves round-trip
  pixel-identically via per-layer bits inference on load.

### 0.18.37-CV

- **Fix: Klein edit stretched reference images whenever aspects differed** (upstream
  [#385](https://github.com/filipstrand/mflux/issues/385)): each reference is now encoded at its own
  aspect-preserved size (capped near 1MP and snapped to multiples of 16 by center-crop), matching
  diffusers. It bit hardest on multi-image edits and explicit `--width`/`--height`. All three Klein
  edit goldens validated, including 9b-kv.
- **Fix: Flash follow-ups**: `--base-model` survives qwen CLI resolution, and Flash through the edit
  CLI no longer double-applies CFG.
- **mlx floor raised to 0.32.0 on macOS** (upstream
  [#489](https://github.com/filipstrand/mflux/pull/489)): older mlx silently corrupts
  `quantized_matmul` past 32768 rows; reproduced here, the whole result garbage at 40000 rows.
- **Releases can publish to PyPI via trusted publishing (OIDC)**, gated by a repository variable
  until the PyPI side is configured; the manual path keeps working.

<details>
<summary><b>Older releases (0.18.1 to 0.18.36)</b></summary>

### 0.18.36-CV

- **New model: nvidia/Qwen-Image-Flash** (`--model qwen-image-flash`): the DMD2 4-step distillation
  of Qwen-Image, transformer byte-identical to 2512, so the 20B Qwen drops from minutes to ~24s of
  denoising. CFG is internalized: guidance forced to 1.0, the per-step negative pass skipped, and
  both flags reported honestly by the warnings and `mflux-capabilities`.
- **Fix: golden-image comparator atol** (upstream [#467](https://github.com/filipstrand/mflux/issues/467)/[#491](https://github.com/filipstrand/mflux/issues/491)):
  near-black references no longer fail on 1-2 count noise; `MFLUX_IMAGE_ALLCLOSE_ATOL` overrides.
- **`--no-metadata`** (upstream [#437](https://github.com/filipstrand/mflux/issues/437)): opt out of
  embedding generation parameters in the output image.

### 0.18.35-CV

- **Fix: `--base-model <alias>` alone crashed blaming the vae.** `from_name` handed back
  `model_name=None` through the `base_model` keyword; the alias now resolves to its own table entry,
  both keywords yield the same config shape, and a missing weights location is reported once as the
  whole-model condition it is. Field-reported; offered upstream as
  [#501](https://github.com/filipstrand/mflux/pull/501).
- **Fix: ComfyUI-format LoRAs (`lora_A`/`lora_B`, no trailing `.weight`) loaded nowhere but FLUX.1.**
  The matcher now accepts the bare spelling for every `.weight` pattern in all seven families, and
  the zero-match error shows the key endings it saw against what the mapping expects. Field-reported
  with a working patch and a fixed-seed A/B; offered upstream as
  [#505](https://github.com/filipstrand/mflux/pull/505).

### 0.18.34-CV

- **`mflux-capabilities`: the CLI contract, machine-readable.** A versioned JSON dump of what each
  command actually honours, because `--help` overstates: some models read options and discard them.
  Self-healing by construction: commands come from the installed console scripts, options and
  defaults from each CLI's live parser, and the honoured/ignored/conditional classification is the
  same constant the runtime warnings read, so the dump and the warnings cannot disagree.
  `--format markdown` for humans, `--format yaml` when PyYAML is present. Battle-tested by a
  downstream consumer against their hand-maintained table (13 of 16 models agreed, and the dump
  caught a regression on their side); their three findings and our review's three more are fixed
  and regression-pinned. Offered upstream as
  [#499](https://github.com/filipstrand/mflux/pull/499).
- **Eight flux-family commands now tell the truth about `--negative-prompt` and `--guidance`**:
  six CLIs beyond the base one accepted a negative prompt they never read, and dev honours
  `--guidance` where schnell silently drops it. All declared and warned at runtime, keyed on the
  resolved model.
- **Synced with upstream** ([#444](https://github.com/filipstrand/mflux/pull/444) by
  [@plz12345](https://github.com/plz12345)): the stepwise VAE-routing fix and its regression test
  file; `cv/main..upstream/main` is empty again. The 0.18.28 stepwise lora-paths guard is now
  test-pinned, and the fix is offered upstream as
  [#500](https://github.com/filipstrand/mflux/pull/500), where the crash still reproduces.

### 0.18.33-CV

- **NVIDIA PiD pixel-diffusion decoder** (upstream [#490](https://github.com/filipstrand/mflux/pull/490)
  by [@azrahello](https://github.com/azrahello), credited under
  [Community PRs pulled in](#community-prs-pulled-in)): `--pid-decode` replaces the VAE decode with a
  4x super-resolving re-render of the final latents, so a 512x512 generation decodes straight to
  2048x2048. Wired on FLUX.1, FLUX.2 Klein, Qwen Image, Krea 2, ERNIE, Ideogram 4 and Z-Image.
  Opt-in, off by default, normal decode untouched. Weights download at runtime (one checkpoint per
  VAE family plus the gated `google/gemma-2-2b-it`). `--pid-degrade-sigma` trades source fidelity for
  invented detail. The 4x output re-draws rather than sharpens; portraits can over-texture. One
  deliberate divergence from the upstream branch: the sampler threads explicit RNG keys instead of
  reseeding the global stream, so multi-seed runs stay reproducible with and without `--pid-decode`.

### 0.18.32-CV

- **Fix: FLUX.2 CLIs discarded all image metadata.** `--metadata` wrote a sidecar JSON containing
  literal `null`; both CLIs now route through the same save path as every other entry point.
  Cherry-pick of upstream [#492](https://github.com/filipstrand/mflux/pull/492), fix by
  [@plz12345](https://github.com/plz12345).
- **Fix: EXIF orientation is applied when loading images.** Photos straight off a phone reached the
  model sideways, and edit models are conditioned on the rotated pixels, so they produced wrong
  output rather than output that merely needs a flip. Reported upstream as
  [#495](https://github.com/filipstrand/mflux/issues/495).
- **Fix: `--steps 1` crashed the shared flow-match scheduler.** The 1-step schedule is now the one
  full-denoise Euler step the class's sibling paths already define; every other step count is
  byte-identical. Reported upstream as [#494](https://github.com/filipstrand/mflux/issues/494).
- **CLIs warn when an option the model cannot honour is dropped.** `--negative-prompt` on FLUX.1 and
  Ideogram 4, `--guidance` and `--negative-prompt` on the guidance-distilled models, and the base
  Z-Image case where an omitted `--guidance` (default 0.0) disables CFG and drops the negative
  prompt. Options stay accepted so scripts keep working; the drop is just no longer silent.
  Abbreviated long options are now rejected parser-wide.

### 0.18.31-CV

- **Fix: Ideogram 4 models saved with a LoRA could not be loaded back.** Baking a LoRA over the fp8
  base folds the adapted layers to MLX q8, and the fresh `Fp8Linear` modules could neither hold the
  folded tensors nor pass the native checkpoint validation, which also misread the deferred fp8
  placeholders as shape mismatches. Folded layers are now rebuilt as `QuantizedLinear` before
  validation and zero-size placeholders are exempt from the shape check.
- **Fix: a reloaded Ideogram save generated with the wrong CFG negative.** Baking strips the LoRA
  wrappers, so the reloaded model no longer knew it was a LoRA model and ran the empty prompt through
  the clean unconditional transformer, which amplifies the baked LoRA at full guidance: the subject
  holds but the prompt scene washes out. `mflux-save` now records the baked LoRA in
  `mflux_model_config.json` and the loader keeps the negative routed through the conditional
  transformer. A save/reload round-trip now reproduces the live-LoRA generation pixel-identically.
  Older saves carry no marker; re-save them to pick up the routing.

### 0.18.30-CV

- **Fix: natively saved Qwen-VAE models could not be loaded back.** Any model saved with `mflux-save`
  that uses the Qwen VAE (Qwen Image, Qwen Image Edit, Qwen-Image-Layered, and Krea 2 Turbo, Raw and
  Depth) failed on load with a `shape_mismatches` error on five `decoder.mid_block.*` tensors.
  `QwenImageRMSNorm` created its gamma with spatial trailing dimensions while the checkpoints store it
  one-dimensional. That difference had been harmless since upstream #269, because the forward path
  reshapes either form, until the native integrity check that arrived with Mage Flow turned it into a
  hard load failure. Gamma now takes the checkpoint's shape, and output is bit-identical on the 4D
  image and 5D video paths. Reported and fixed by [@fortinmike](https://github.com/fortinmike).

### 0.18.29-CV

- **Faster SeedVR2 histogram matching** (#488, credited under [Community PRs pulled in](#community-prs-pulled-in)):
  the inverse permutation was built with `argsort(argsort(x))`, an O(n log n) sort standing in for what a
  single scatter gives in O(n). Measured on an M5 Max at 4M elements, roughly one channel of a 1080p frame:
  **880 ms → 17.3 ms per channel**. Output is identical, checked with repeated values where the stable-sort
  semantics could have broken.
- **[Who wrote what](#-who-wrote-what)**: a new contributors section, an avatar grid of the fourteen people
  whose ports and fixes make up this build, over tables read out of the git history with every row linking
  its PR. Three attributions that a hand-written table gets wrong: Krea 2 is @plz12345's #453, Ideogram 4 is
  @omercelik's #433, ERNIE-Image is @azrahello's #417.
- **A move and a move back**: the repo spent a few hours in the [mflux-community](https://github.com/mflux-community)
  org on the day this was released, and came back the same night while the question of who runs mflux is
  still open. Both moves were transfers rather than forks, so stars, releases, issues and pull requests
  travelled each time and every old URL still redirects for web and for git. Existing clones and
  `pip install git+` lines were never affected. The package URLs shipped in 0.18.29 still name the org;
  PyPI does not allow replacing a published version, so they are corrected from the next release on.

### 0.18.28-CV

- Fixed a crash in the stepwise preview for models that do not take LoRA: the handler read `lora_paths`
  off the model unconditionally, so asking for step images on FIBO ended the generation instead of
  writing them.

### 0.18.27-CV

- **Mage Flow** (#483 by @ivanfioravanti, credited under [Community PRs pulled in](#community-prs-pulled-in)):
  Microsoft's Mage Flow family, text-to-image and instruction edit, ported to MLX
  (`mflux-generate-mage-flow`, `mflux-generate-mage-flow-edit`). Validated on turbo and on the RL/CFG
  variants, both t2i and edit.
- Allow `mlx` 0.32.x.

### 0.18.26-CV

- **Z-Image Turbo Union ControlNet, native in MLX.** The first Z-Image ControlNet running in MLX, with
  all five modalities computed locally (no pre-made control image needed):

  ```bash
  mflux-generate-z-image-controlnet \
      --control canny:room.png:1.0 --controlnet-strength 0.6 \
      --prompt "a cozy bedroom, photorealistic" --steps 8 --output out.png
  ```

  `--control type:path[:strength]` is repeatable to stack controls. Types: `canny`, `mlsd`, `depth`,
  `hed`, `pose`.
- Preprocessors: `canny` and `mlsd` via OpenCV, `depth` via the native DepthPro mflux already ships, and
  native `mlx.nn` ports of `hed` (ControlNetHED) and `pose` (OpenPose body). Only weight loading touches
  torch; every forward pass is MLX.
- The controlnet inference is numerically matched to the diffusers `ZImageControlNetModel` (block-by-block
  residual cosine 1.00000). A correctness fix carries the refined control tokens into the main control
  layers, which the original port dropped.

### 0.18.25-CV

- **Multi-ControlNet for FLUX.1.** Several controlnets can now be stacked, each with its own
  checkpoint, control image and strength (for example depth + canny). Repeat `--controlnet-path`
  and `--controlnet-image-path` (and optionally `--controlnet-strength`, or give one value for all):

  ```bash
  mflux-generate-controlnet -m dev --prompt "a modern living room" \
      --controlnet-path org/depth --controlnet-image-path depth.png --controlnet-strength 0.7 \
      --controlnet-path org/canny --controlnet-image-path canny.png --controlnet-strength 0.4
  ```

  The residual path is additive, so each net's residuals are spread over the transformer's blocks
  with the rule the transformer already applies and then summed. Controlnets with different block
  counts stack correctly, and a single controlnet renders exactly as before (the expansion
  reproduces the transformer's own selection index for index, which is pinned by a test).
- `--controlnet-path` is new: it selects a controlnet checkpoint (local path or HF repo) instead of
  the one named by the model config, which also makes `Flux1Controlnet(controlnet_path=...)` work.
  It was previously accepted by the constructor and silently ignored.
- A single `--controlnet-image-path` / `--controlnet-strength` keeps its scalar shape, so existing
  commands and the metadata round-trip are unchanged.
- Canny preprocessing is decided per controlnet, from that net's own checkpoint name (the same match
  `is_canny()` makes). A depth + canny stack therefore preprocesses only the canny image, and a
  config-driven canny run behaves exactly as before.
- Known limitation: the image metadata holds a single controlnet, so a stacked run records only its
  first net there. Metadata-driven re-runs of a stack are not supported; pass the flags explicitly.
  Single-controlnet metadata is unchanged.

### 0.18.24-CV

- FLUX.2-klein now exposes `flux2-klein-edit` / `flux2-edit` / `klein-edit` aliases (it does txt2img
  and edit from the same weights), so the edit variant is selectable by name — used by the
  ComfyUI-mflux-AnyModel node.
- Qwen-Image-Edit: skip the unconditional pass at guidance 1.0 (it reduces to the conditional noise
  there), halving per-step compute for CFG-distilled setups like the Lightning step-reduction LoRAs.

### 0.18.23-CV

- Updated the default Qwen models to the latest releases (based on #475's sibling, #474, credited
  under [Community PRs pulled in](#community-prs-pulled-in)): `qwen-image` now loads Qwen-Image-2512 and
  `qwen-image-edit` loads Qwen-Image-Edit-2511. The architecture is identical to the prior releases, so
  they are drop-in; both were validated end-to-end on Apple Silicon. Unlike the upstream PR, the old
  `qwen-edit-2509` alias still resolves to the actual 2509 weights (kept as its own entry) instead of
  silently pointing at the new default.

### 0.18.22-CV

- Pulled in configurable VAE decode tiling (#475, credited under [Community PRs pulled in](#community-prs-pulled-in)):
  `--vae-tiling` / `--vae-tile-size` decouple tiled decoding from full low-RAM mode, so you can cut peak
  memory on large generations without the rest of the low-RAM penalty.

### 0.18.21-CV

- Krea 2 depth ControlNet: fix the estimated-depth convention (Depth Pro already outputs near = white,
  like Depth-Anything-V2, so its map is now used un-inverted — an earlier inversion put the background in
  front). `--save-depth-map` now writes the estimated map (`<output>_depth_map.png`).

### 0.18.20-CV

- Krea 2 depth ControlNet now supports quantization (`-q 8` / `-q 4`): the control deltas are baked
  into the base weights before quantization, so the packed model keeps the depth control. Validated
  end-to-end at int8.

### 0.18.19-CV
- New: **Krea 2 depth ControlNet** (`mflux-generate-krea2-controlnet`). Runs the community
  [Krea-2-controlnet](https://github.com/Tanmaypatil123/Krea-2-controlnet) depth checkpoint natively in
  MLX: the input projection is widened to take a depth latent concatenated on the channel axis, and the
  attention/MLP deltas are merged into the base weights. Depth is taken from a supplied map
  (`--depth-image-path`) or estimated with the native Depth Pro. See the Krea 2 depth ControlNet section
  below.

### 0.18.18-CV
- Pulled in two new upstream models (credited under [Community PRs pulled in](#community-prs-pulled-in)):
  Boogu-Image-0.1-Turbo (#446, `mflux-generate-boogu`) and Qwen-Image-Layered (#302,
  `mflux-generate-qwen-layered`) for decomposing an image into RGBA layers.

### 0.18.17-CV
- Pulled in Ideogram 4 mlx-forge checkpoint loading (#445) and mixed-quant FLUX.2 inference (#436),
  both credited under [Community PRs pulled in](#community-prs-pulled-in).

### 0.18.16-CV
- Pulled in two upstream bug fixes (credited under [Community PRs pulled in](#community-prs-pulled-in)):
  fused-qkv LoRA loading (#459) and ERNIE / Krea 2 tiled img2img (#463).
- Repo automation: CodeQL security scanning, structured issue forms, and PR / issue auto-labeling.

### 0.18.15-CV
First release under the `mflux-CV` name. Same codebase as the prior `+fxd0h` builds (0.18.1 through
0.18.5); this is the rebrand plus everything listed below.

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

</details>

### Community PRs pulled in
- **[filipstrand/mflux#459](https://github.com/filipstrand/mflux/pull/459) by Sahil Tanveer** — fix LoRA
  loading for fused qkv layers: keep the shared rank/down projection whole and slice only the up
  projection, so kohya/BFL FLUX LoRAs with a rank divisible by 3/4 load correctly.
- **[filipstrand/mflux#463](https://github.com/filipstrand/mflux/pull/463) by Mike Wallio** — fix ERNIE
  and Krea 2 img2img with tiled VAE latents: the 5D tiled-VAE pack path took the wrong slice; keep the
  singleton temporal axis so tiled-decode img2img reconstructs correctly.
- **[filipstrand/mflux#445](https://github.com/filipstrand/mflux/pull/445) by plz12345** — load
  Ideogram 4 from mlx-forge converted checkpoints (bf16 / int8) by HF repo id, skipping the fp8-only
  validation; plus a once-built boolean attention keep-mask. Merged with our gradient-checkpointing.
- **[filipstrand/mflux#436](https://github.com/filipstrand/mflux/pull/436) by Ian Scrivener** — mixed-quant
  inference for FLUX.2: quantize the transformer and text encoder to different levels, or load each from
  its own path (`--model-transformer` / `--model-text-encoder`); the VAE stays bf16. Merged alongside our
  LoKr flux2 changes.
- **[filipstrand/mflux#446](https://github.com/filipstrand/mflux/pull/446) by plz12345** — new model:
  Boogu-Image-0.1-Turbo (`mflux-generate-boogu`). Applied cleanly.
- **[filipstrand/mflux#302](https://github.com/filipstrand/mflux/pull/302) by ZimengXiong** — new model:
  Qwen-Image-Layered (`mflux-generate-qwen-layered`) for decomposing an image into RGBA layers, with a
  low-memory chunked save path. We kept our README; the PR's stale old-structure README changes were dropped.
- **[filipstrand/mflux#475](https://github.com/filipstrand/mflux/pull/475) by azrahello** — configurable
  VAE decode tiling: `--vae-tiling` and `--vae-tile-size` (min 128, multiple of 16) enable tiled decoding
  on its own, decoupled from `--low-ram`, to lower peak memory during the VAE decode phase. Applied cleanly.
- **[filipstrand/mflux#474](https://github.com/filipstrand/mflux/pull/474) by imbible** — bump the default
  Qwen models to Qwen-Image-2512 and Qwen-Image-Edit-2511. Both validated generating end-to-end on Apple
  Silicon (identical architecture, drop-in). We diverged in one place: kept a separate `qwen-image-edit-2509`
  entry so `qwen-edit-2509` still resolves to the real 2509 weights instead of the new default.
- **[filipstrand/mflux#483](https://github.com/filipstrand/mflux/pull/483) by Ivan Fioravanti** — new model
  family: Mage Flow (Microsoft), text-to-image and instruction edit (`mflux-generate-mage-flow`,
  `mflux-generate-mage-flow-edit`). Integrated onto this build with the shared-file conflict resolutions
  and review fixes; validated on turbo and on the RL/CFG variants, both t2i and edit.
- **[filipstrand/mflux#488](https://github.com/filipstrand/mflux/pull/488) by Unmilan Mukherjee** — build
  the inverse permutation in SeedVR2's histogram matching with a scatter instead of a second sort. Verified
  before integrating: identical output including with repeated values, and 880 ms → 17.3 ms per channel at
  4M elements on an M5 Max.
- **[filipstrand/mflux#492](https://github.com/filipstrand/mflux/pull/492) by plz12345** — the FLUX.2 CLIs
  discarded all image metadata: `--metadata` wrote a sidecar containing literal `null`. Both CLIs now route
  through the same save path as every other entry point. Cherry-picked with authorship preserved.
- **[filipstrand/mflux#490](https://github.com/filipstrand/mflux/pull/490) by azrahello** — the NVIDIA PiD
  pixel-diffusion decoder (`--pid-decode`): a 4x super-resolving re-render of the final latents, wired across
  seven model families. Pulled at its post-review state, with the `--pid-degrade-sigma` piece co-authored by
  plz12345; one deliberate divergence keeps multi-seed runs reproducible (explicit RNG keys in the sampler).
- **[filipstrand/mflux#444](https://github.com/filipstrand/mflux/pull/444) by plz12345** — the stepwise
  preview's VAE routing fix and its regression test file, synced so `cv/main..upstream/main` stays empty.
- **[filipstrand/mflux#489](https://github.com/filipstrand/mflux/pull/489) by plz12345** — require mlx
  0.32.0 or newer on macOS: below that, `quantized_matmul` silently corrupts its output once the input
  passes 32768 rows, and quantized SeedVR2 at large output sizes runs into it. Reproduced independently
  before porting the floor bump.

### Krea 2 depth ControlNet

Steer a Krea 2 generation with the depth of a reference image, running natively in MLX. Uses the
community [Krea-2-controlnet](https://github.com/Tanmaypatil123/Krea-2-controlnet) depth checkpoint by
Tanmay Patil (base weights: `krea/Krea-2-Raw` / `krea/Krea-2-Turbo`).

```bash
mflux-generate-krea2-controlnet \
  --model krea2 \
  --controlnet-path /path/to/depth-control-lora.safetensors \
  --image-path reference.png \
  --prompt "a glowing crystal orb on a wooden table, studio photo" \
  --steps 8 --seed 42 --height 1024 --width 1024 \
  --output out.png
```

- `--image-path` estimates depth with the native Depth Pro (near = white, used as-is, the same
  convention the checkpoint was trained on). Add `--save-depth-map` to also write the estimated map. For
  the closest match to the training data, pass a Depth-Anything-V2 map directly with `--depth-image-path`.
- `--controlnet-strength` scales how strongly the control deltas are merged (default `1.0`).
- `--krea2-uncensor` is supported here too.
- `-q 8` / `-q 4` quantize the variant: the control deltas are baked into the base weights before
  quantization, so the packed model keeps the depth control.

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
- [👥 Who wrote what](#-who-wrote-what)
- [🙏 Acknowledgements](#-acknowledgements)
- [⚖️ License](#%EF%B8%8F-license)

---

### 💡 Philosophy

MFLUX is a line-by-line MLX port of several state-of-the-art generative image models from the [Huggingface Diffusers](https://github.com/huggingface/diffusers) and [Huggingface Transformers](https://github.com/huggingface/transformers) libraries. All models are implemented from scratch in MLX, using only tokenizers from the [Huggingface Transformers](https://github.com/huggingface/transformers) library. MFLUX is purposefully kept minimal and explicit, [@karpathy](https://gist.github.com/awni/a67d16d50f0f492d94a10418e0592bde?permalink_comment_id=5153531#gistcomment-5153531) style.

---

### 💿 Installation
If you haven't already, [install `uv`](https://github.com/astral-sh/uv?tab=readme-ov-file#installation), then run:

```sh
uv tool install --upgrade mflux-cv
```

> For this community build the distribution is `mflux-cv` (the upstream original is `uv tool install mflux`;
> install one or the other, never both — see the note at the top of this file).

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
|[Mage Flow](src/mflux/models/mage_flow/README.md) | Jul 2026 | 8.7B | Base, RL & Turbo | No | Native-resolution generation with natural-language, multi-image editing. |
|[Z-Image](src/mflux/models/z_image/README.md) | Nov 2025 | 6B | Distilled & Base | Yes | Fast, small, very good quality and realism. |
|[Krea 2](src/mflux/models/krea2/README.md) | Jun 2026 | 12B | Turbo (distilled) | No | Very good quality with a wide range of styles; good for creative exploration. |
|[FLUX.2](src/mflux/models/flux2/README.md) | Jan 2026 | 4B & 9B | Distilled & Base | Yes | Fastest + smallest with very good qaility and edit capabilities. |
|[Ideogram 4](src/mflux/models/ideogram4/README.md) | Jun 2026 | 9B | Base | No | JSON-caption-native, typography-focused text-to-image generation. |
|[ERNIE-Image](src/mflux/models/ernie_image/README.md) | Apr 2026 | 8B | Distilled & Base | No | Single-stream DiT from Baidu. Vivid, high-contrast output. |
|[Boogu](src/mflux/models/boogu) | Jun 2026 | 4B | Turbo (distilled) | No | 4-step DMD generation; fast drafts up to ~768px, use 8 steps at 1024. |
|[FIBO](src/mflux/models/fibo/README.md) | Oct 2025+ | 8B | Distilled & Base | No | Very good JSON-based prompt understanding. Has edit capabilities. |
|[SeedVR2](src/mflux/models/seedvr2/README.md) | Jun 2025 | 3B & 7B | — | No | Best upscaling model. |
|[Qwen Image](src/mflux/models/qwen/README.md) | Aug 2025+ | 20B | Base & Flash (distilled) | No | Strong prompt understanding and world knowledge; the Flash variant (`--model qwen-image-flash`) cuts denoising to 4 steps. Has edit capabilities |
|[Depth Pro](src/mflux/models/depth_pro/README.md) | Oct 2024 | — | — | No | Very fast and accurate depth estimation model from Apple. |
|[FLUX.1](src/mflux/models/flux/README.md) | Aug 2024 | 12B | Distilled & Base | No (legacy) | Legacy option with decent quality. Has edit capabilities with 'Kontext' model and upscaling support via ControlNet |

---

### ✨ Features

**General**
- Quantization and local model loading
- LoRA support (multi-LoRA, scales, library lookup), including LyCORIS LoKr on FLUX.1, FLUX.2, Qwen, Ideogram 4 and Krea 2, and ComfyUI-format adapters (bare `lora_A`/`lora_B` tensor names) in every family
- Metadata export + reuse, plus prompt file support
- `mflux-capabilities`: a machine-readable JSON contract of what each CLI actually honours (honoured / ignored / conditional per option), self-healing from the installed entry points and live parsers; `--format markdown` for humans. CLIs also warn at runtime when an option they cannot honour is dropped
- NVIDIA PiD pixel-diffusion decoding (`--pid-decode`): replace the VAE decode with a 4x super-resolving re-render on seven model families

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
- [mflux-paint](https://github.com/Amo643/mflux-paint) — native macOS inpaint/edit app (pywebview), 16 models across edit/inpaint/text-to-image, mask painting, multi-seed batch, by [@Amo643](https://github.com/Amo643)
- [ComfyUI-mflux-AnyModel](https://github.com/fxd0h/ComfyUI-mflux-AnyModel) — run any mflux model in ComfyUI via MLX (live previews, LoRA, ControlNet stacking, PiD decode), built on this distro, by [@fxd0h](https://github.com/fxd0h)
- [image-studio](https://github.com/MLXBits/image-studio) — native SwiftUI image studio for Apple Silicon running on mflux-cv, by [@plz12345](https://github.com/plz12345)
- [Mflux-ComfyUI](https://github.com/raysers/Mflux-ComfyUI) by [@raysers](https://github.com/raysers)
- [MFLUX-WEBUI](https://github.com/CharafChnioune/MFLUX-WEBUI) by [@CharafChnioune](https://github.com/CharafChnioune)
- [mflux-fasthtml](https://github.com/anthonywu/mflux-fasthtml) by [@anthonywu](https://github.com/anthonywu)
- [mflux-streamlit](https://github.com/elitexp/mflux-streamlit) by [@elitexp](https://github.com/elitexp)
- [mlx-taef](https://github.com/IonDen/mlx-taef) — TAESD/TAEF tiny-autoencoder live previews and low-memory FLUX decode for mflux, by [@IonDen](https://github.com/IonDen)
- [mlx-teacache](https://github.com/IonDen/mlx-teacache) — TeaCache step-skipping to speed up FLUX generation in mflux, by [@IonDen](https://github.com/IonDen)

---

### 👥 Who wrote what

The people whose work you run when you use this build:

<table>
<tr>
<td align="center" width="150">
<a href="https://github.com/filipstrand"><img src="https://github.com/filipstrand.png?size=100" width="72" alt="filipstrand"><br><b>Filip Strand</b></a><br>
<sub>created mflux<br>FLUX.1 · FLUX.2 · Z-Image<br>FIBO · SeedVR2 · Qwen · Depth Pro</sub>
</td>
<td align="center" width="150">
<a href="https://github.com/plz12345"><img src="https://github.com/plz12345.png?size=100" width="72" alt="plz12345"><br><b>plz12345</b></a><br>
<sub>Krea 2 · Boogu-Image<br>Ideogram 4 mlx-forge loading</sub>
</td>
<td align="center" width="150">
<a href="https://github.com/ivanfioravanti"><img src="https://github.com/ivanfioravanti.png?size=100" width="72" alt="ivanfioravanti"><br><b>Ivan Fioravanti</b></a><br>
<sub>Mage Flow<br>text-to-image and edit</sub>
</td>
<td align="center" width="150">
<a href="https://github.com/azrahello"><img src="https://github.com/azrahello.png?size=100" width="72" alt="azrahello"><br><b>Alessandro Rizzo</b></a><br>
<sub>ERNIE-Image<br>configurable VAE tiling</sub>
</td>
<td align="center" width="150">
<a href="https://github.com/omercelik"><img src="https://github.com/omercelik.png?size=100" width="72" alt="omercelik"><br><b>omercelik</b></a><br>
<sub>Ideogram 4</sub>
</td>
</tr>
<tr>
<td align="center" width="150">
<a href="https://github.com/ZimengXiong"><img src="https://github.com/ZimengXiong.png?size=100" width="72" alt="ZimengXiong"><br><b>Zimeng Xiong</b></a><br>
<sub>Qwen-Image-Layered<br>RGBA decomposition</sub>
</td>
<td align="center" width="150">
<a href="https://github.com/JanGrohn"><img src="https://github.com/JanGrohn.png?size=100" width="72" alt="JanGrohn"><br><b>Jan Grohn</b></a><br>
<sub>LyCORIS LoKr adapters</sub>
</td>
<td align="center" width="150">
<a href="https://github.com/michaeltrefry"><img src="https://github.com/michaeltrefry.png?size=100" width="72" alt="michaeltrefry"><br><b>michaeltrefry</b></a><br>
<sub>FLUX.2 KV-cache<br>klein-9b-kv</sub>
</td>
<td align="center" width="150">
<a href="https://github.com/ianscrivener"><img src="https://github.com/ianscrivener.png?size=100" width="72" alt="ianscrivener"><br><b>Ian Scrivener</b></a><br>
<sub>FLUX.2 mixed-quant inference</sub>
</td>
<td align="center" width="150">
<a href="https://github.com/deadmansahil"><img src="https://github.com/deadmansahil.png?size=100" width="72" alt="deadmansahil"><br><b>Sahil Tanveer</b></a><br>
<sub>fused-qkv LoRA loading</sub>
</td>
</tr>
<tr>
<td align="center" width="150">
<a href="https://github.com/scaryrawr"><img src="https://github.com/scaryrawr.png?size=100" width="72" alt="scaryrawr"><br><b>Mike Wallio</b></a><br>
<sub>ERNIE / Krea 2<br>img2img tiled latents</sub>
</td>
<td align="center" width="150">
<a href="https://github.com/imbible"><img src="https://github.com/imbible.png?size=100" width="72" alt="imbible"><br><b>George</b></a><br>
<sub>Qwen model version defaults</sub>
</td>
<td align="center" width="150">
<a href="https://github.com/Missing-Identity"><img src="https://github.com/Missing-Identity.png?size=100" width="72" alt="Missing-Identity"><br><b>Unmilan Mukherjee</b></a><br>
<sub>SeedVR2<br>linear-time histogram matching</sub>
</td>
<td align="center" width="150">
<a href="https://github.com/fxd0h"><img src="https://github.com/fxd0h.png?size=100" width="72" alt="fxd0h"><br><b>Mariano Abad</b></a><br>
<sub>Z-Image and Krea 2 ControlNets<br>multi-ControlNet · training suite</sub>
</td>
<td align="center" width="150">
<a href="https://github.com/fortinmike"><img src="https://github.com/fortinmike.png?size=100" width="72" alt="fortinmike"><br><b>Michaël Fortin</b></a><br>
<sub>Qwen VAE native<br>save/load fix</sub>
</td>
</tr>
<tr>
<td align="center" width="150">
<a href="https://github.com/filipstrand/mflux/graphs/contributors"><b>everyone else</b></a><br>
<sub>every contributor and tester<br>in the upstream graph</sub>
</td>
</tr>
</table>

<details>
<summary><b>The provenance behind that grid</b></summary>

The provenance behind that grid, read out of the git history rather than filled in by hand. Dates are
the **merge date of the PR that brought the work in**, which is months away from the model's own public
release date in several cases. Anyone can re-derive a row:
`gh pr view <n> -R filipstrand/mflux --json author,mergedAt`.

Ported upstream and inherited here:

| Model | Component | Contributor | Merged | PR |
|---|---|---|---|---|
| FLUX.1 | | [@filipstrand](https://github.com/filipstrand) | 2024-08-12 | initial release |
| Depth Pro | | [@filipstrand](https://github.com/filipstrand) | 2025-03-23 | [#159](https://github.com/filipstrand/mflux/pull/159) |
| Qwen Image | | [@filipstrand](https://github.com/filipstrand) | 2025-10-06 | [#269](https://github.com/filipstrand/mflux/pull/269) |
| FIBO | | [@filipstrand](https://github.com/filipstrand) | 2025-11-27 | [#279](https://github.com/filipstrand/mflux/pull/279) |
| Z-Image | | [@filipstrand](https://github.com/filipstrand) | 2025-12-03 | [#284](https://github.com/filipstrand/mflux/pull/284) |
| SeedVR2 | | [@filipstrand](https://github.com/filipstrand) | 2025-12-31 | [#297](https://github.com/filipstrand/mflux/pull/297) |
| FLUX.2 | | [@filipstrand](https://github.com/filipstrand) | 2026-01-18 | [#323](https://github.com/filipstrand/mflux/pull/323) |
| ERNIE-Image | | [@azrahello](https://github.com/azrahello) | 2026-06-06 | [#417](https://github.com/filipstrand/mflux/pull/417) |
| Ideogram 4 | | [@omercelik](https://github.com/omercelik) | 2026-06-06 | [#433](https://github.com/filipstrand/mflux/pull/433) |
| FLUX.2 | KV-cache (klein-9b-kv) | [@michaeltrefry](https://github.com/michaeltrefry) | 2026-06-07 | [#426](https://github.com/filipstrand/mflux/pull/426) |
| Krea 2 | | [@plz12345](https://github.com/plz12345) | 2026-06-30 | [#453](https://github.com/filipstrand/mflux/pull/453) |

Still open upstream, running here because this build pulled them in. Each is described in detail under
[Community PRs pulled in](#community-prs-pulled-in):

| Model | Component | Contributor | PR |
|---|---|---|---|
| Qwen-Image-Layered | | [@ZimengXiong](https://github.com/ZimengXiong) | [#302](https://github.com/filipstrand/mflux/pull/302) |
| | LyCORIS LoKr | [@JanGrohn](https://github.com/JanGrohn) | [#422](https://github.com/filipstrand/mflux/pull/422) |
| FLUX.2 | mixed-quant inference | [@ianscrivener](https://github.com/ianscrivener) | [#436](https://github.com/filipstrand/mflux/pull/436) |
| Ideogram 4 | mlx-forge checkpoint loading | [@plz12345](https://github.com/plz12345) | [#445](https://github.com/filipstrand/mflux/pull/445) |
| Boogu-Image | | [@plz12345](https://github.com/plz12345) | [#446](https://github.com/filipstrand/mflux/pull/446) |
| | fused-qkv LoRA loading | [@deadmansahil](https://github.com/deadmansahil) | [#459](https://github.com/filipstrand/mflux/pull/459) |
| ERNIE / Krea 2 | img2img tiled latents | [@scaryrawr](https://github.com/scaryrawr) | [#463](https://github.com/filipstrand/mflux/pull/463) |
| Qwen | model version defaults | [@imbible](https://github.com/imbible) | [#474](https://github.com/filipstrand/mflux/pull/474) |
| | configurable VAE decode tiling | [@azrahello](https://github.com/azrahello) | [#475](https://github.com/filipstrand/mflux/pull/475) |
| SeedVR2 | linear-time histogram matching | [@Missing-Identity](https://github.com/Missing-Identity) | [#488](https://github.com/filipstrand/mflux/pull/488) |
| | mlx 0.32.0 floor (quantized_matmul corruption) | [@plz12345](https://github.com/plz12345) | [#489](https://github.com/filipstrand/mflux/pull/489) |
| | NVIDIA PiD pixel-diffusion decoder (`--pid-decode`) | [@azrahello](https://github.com/azrahello) | [#490](https://github.com/filipstrand/mflux/pull/490) |
| FLUX.2 | CLI metadata embedding fix | [@plz12345](https://github.com/plz12345) | [#492](https://github.com/filipstrand/mflux/pull/492) |
| Mage Flow | | [@ivanfioravanti](https://github.com/ivanfioravanti) | [#483](https://github.com/filipstrand/mflux/pull/483) |

Written in this build:

| Model | Component | Contributor | Where |
|---|---|---|---|
| Z-Image | ControlNet: Union (canny/mlsd/depth/hed/pose) | [@fxd0h](https://github.com/fxd0h) | mirrored upstream as [#482](https://github.com/filipstrand/mflux/pull/482) |
| Krea 2 | ControlNet: Depth | [@fxd0h](https://github.com/fxd0h) | this build only, no upstream PR |
| Qwen VAE | native save/load shape fix | [@fortinmike](https://github.com/fortinmike) | [mflux-cv#33](https://github.com/HowDidTheCatGetSoFat/mflux-cv/pull/33) |
| FLUX.1 | multi-ControlNet stacking | [@fxd0h](https://github.com/fxd0h) | this build only |
| | training suite (DoRA, LR schedules, optimizers, LoRA alpha, grad clipping/accumulation) | [@fxd0h](https://github.com/fxd0h) | open upstream as [#442](https://github.com/filipstrand/mflux/pull/442), [#447](https://github.com/filipstrand/mflux/pull/447), [#448](https://github.com/filipstrand/mflux/pull/448), [#449](https://github.com/filipstrand/mflux/pull/449), [#450](https://github.com/filipstrand/mflux/pull/450), [#451](https://github.com/filipstrand/mflux/pull/451), [#452](https://github.com/filipstrand/mflux/pull/452), [#464](https://github.com/filipstrand/mflux/pull/464) |
| Ideogram 4 | LoRA inference fixes (uncond routing, fp8 bake, CFG truncation) | [@fxd0h](https://github.com/fxd0h) | open upstream as [#439](https://github.com/filipstrand/mflux/pull/439), [#440](https://github.com/filipstrand/mflux/pull/440), [#441](https://github.com/filipstrand/mflux/pull/441) |
| Krea 2 | Raw variant, LoRA training, dynamic sigma schedule | [@fxd0h](https://github.com/fxd0h) | open upstream as [#462](https://github.com/filipstrand/mflux/pull/462), [#465](https://github.com/filipstrand/mflux/pull/465) |

One caveat on the method, stated so nobody trusts it further than it deserves: a first-commit-in-directory
lookup is wrong for anything introduced before #269, because that PR created the `src/mflux/models/`
layout and the lookup returns the restructure instead of the original work. Those rows were confirmed by
commit subject and then against the PR itself.

</details>

---

### 🙏 Acknowledgements

MFLUX would not be possible without the great work of:

- The MLX Team for [MLX](https://github.com/ml-explore/mlx) and [MLX examples](https://github.com/ml-explore/mlx-examples)
- Black Forest Labs for the [FLUX project](https://github.com/black-forest-labs/flux)
- Bria for the [FIBO project](https://huggingface.co/briaai/FIBO)
- Tongyi Lab for the [Z-Image project](https://tongyi-mai.github.io/Z-Image-blog/)
- Baidu for the [ERNIE-Image project](https://huggingface.co/baidu/ERNIE-Image)
- Microsoft for the [Mage Flow project](https://huggingface.co/collections/microsoft/mage-flow)
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
