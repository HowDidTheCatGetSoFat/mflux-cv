# Mage Flow

MFLUX includes a native MLX implementation of Microsoft's
[Mage Flow](https://huggingface.co/collections/microsoft/mage-flow) family for
text-to-image generation and instruction-based, multi-image editing. Mage Flow
uses a Qwen3-VL text encoder, a native-resolution flow transformer, and its own
VAE. See the [technical report](https://arxiv.org/abs/2607.19064) for architecture
and training details.

## Models and defaults

| Model alias | Hugging Face checkpoint | Steps | Guidance |
| --- | --- | ---: | ---: |
| `mage-flow-base` | `microsoft/Mage-Flow-Base` | 30 | 5.0 |
| `mage-flow` | `microsoft/Mage-Flow` | 20 | 5.0 |
| `mage-flow-turbo` | `microsoft/Mage-Flow-Turbo` | 4 | 1.0 |
| `mage-flow-edit-base` | `microsoft/Mage-Flow-Edit-Base` | 30 | 5.0 |
| `mage-flow-edit` | `microsoft/Mage-Flow-Edit` | 30 | 5.0 |
| `mage-flow-edit-turbo` | `microsoft/Mage-Flow-Edit-Turbo` | 4 | 1.0 |

The commands select these recommended step counts automatically. The equivalent
`mageflow-*` aliases are also accepted. Full BF16 checkpoints are roughly 17 GB;
use `--quantize` when a smaller transformer footprint is useful.

## Text-to-image

The default command uses the RL-enhanced `mage-flow` checkpoint:

```sh
mflux-generate-mage-flow \
  --prompt "A glass greenhouse glowing at blue hour, tropical plants, rain on the panes, cinematic natural light" \
  --width 1024 \
  --height 1024 \
  --seed 42 \
  -q 8
```

Select Base or Turbo with `--model`:

```sh
mflux-generate-mage-flow \
  --model mage-flow-turbo \
  --prompt "Editorial photograph of a red fox crossing a frozen lake at sunrise" \
  --seed 42
```

Turbo automatically uses 4 steps and guidance 1.0. Base and RL checkpoints use
classifier-free guidance and accept `--negative-prompt`. At high guidance,
`--renormalization` can reduce over-saturation.

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.mage_flow.variants.txt2img.mage_flow import MageFlow

model = MageFlow(
    model_config=ModelConfig.mage_flow(),
    quantize=8,
)
image = model.generate_image(
    seed=42,
    prompt="A glass greenhouse glowing at blue hour, tropical plants, rain on the panes",
    num_inference_steps=20,
    width=1024,
    height=1024,
    guidance=5.0,
)
image.save("mage_flow.png")
```

</details>

## Image editing

Pass one or more reference images. The first image defines the output aspect
ratio when `--max-size` is used:

```sh
mflux-generate-mage-flow-edit \
  --image-paths scene.png object.png \
  --prompt "Place the object from Image 2 naturally on the table in Image 1" \
  --max-size 1024 \
  --seed 42 \
  -q 8
```

Without `--width`, `--height`, or `--max-size`, the output keeps the primary
reference image's resolution, rounded to a multiple of 16. Supplying both
`--width` and `--height` overrides `--max-size`. Reference images used by the
VAE retain the output resolution; the vision-language conditioning copies are
downscaled internally to match training.

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.mage_flow.variants.edit.mage_flow_edit import MageFlowEdit

model = MageFlowEdit(
    model_config=ModelConfig.mage_flow_edit(),
    quantize=8,
)
image = model.generate_image(
    seed=42,
    prompt="Place the object from Image 2 naturally on the table in Image 1",
    image_paths=["scene.png", "object.png"],
    num_inference_steps=30,
    max_size=1024,
    guidance=5.0,
)
image.save("mage_flow_edit.png")
```

</details>

## Watermark, memory, and safety

Mage Flow's released Gaussian-Shading watermark is embedded in the initial noise
by default. Use `--gaussian-shading-key KEY` to supply your own detection key.
If omitted, MFLUX checks `MAGEFLOW_GS_KEY`, then `MAGEFLOW_GS_KEY_FILE`, then
`~/.mageflow/gs_key`, and finally uses the released checkpoint key.

The text encoder and transformer are large. Add `--low-ram` if memory pressure is
more important than generation speed. Use `--stepwise-image-output-dir DIR` to
inspect intermediate denoising images.

To save a reusable quantized checkpoint:

```sh
mflux-save \
  --model mage-flow-turbo \
  --quantize 4 \
  --path mage-flow-turbo-4bit
```

Load it by passing that directory to `--model`. LoRA adapters are not currently
supported for Mage Flow.

As in Microsoft's release, every positive generation prompt is screened by the
same Qwen3-VL weights before denoising. Edit requests screen both the instruction
and the original-resolution source images. The classifier is fail-closed and a
blocked request returns a plain white image; its autoregressive pass can be a
noticeable part of Turbo-model latency.
