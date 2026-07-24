import os
from pathlib import Path

import platformdirs

BATTERY_PERCENTAGE_STOP_LIMIT = 5
CONTROLNET_STRENGTH = 0.4
DEFAULT_DEV_FILL_GUIDANCE = 30
DEFAULT_DEPTH_GUIDANCE = 10
DIMENSION_STEP_PIXELS = 16
GUIDANCE_SCALE = 3.5
GUIDANCE_SCALE_KONTEXT = 2.5
HEIGHT, WIDTH = 1024, 1024
IMAGE_STRENGTH = 0.4
MODEL_CHOICES = [
    "dev",
    "schnell",
    "krea-dev",
    "dev-krea",
    "krea-2",
    "krea2",
    "qwen",
    "qwen-image-layered",
    "mage-flow-base",
    "mageflow-base",
    "mage-flow",
    "mageflow",
    "mage-flow-turbo",
    "mageflow-turbo",
    "mage-flow-edit-base",
    "mageflow-edit-base",
    "mage-flow-edit",
    "mageflow-edit",
    "mage-flow-edit-turbo",
    "mageflow-edit-turbo",
    "fibo",
    "fibo-lite",
    "fibo-edit",
    "fibo-edit-rmbg",
    "z-image",
    "z-image-turbo",
    "z-image-controlnet",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux2-klein-9b-kv",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
    "ernie-image-turbo",
    "ernie-image",
    "ideogram4",
    "boogu-image-turbo",
    "boogu",
]
MODEL_INFERENCE_STEPS = {
    "dev": 25,
    "schnell": 4,
    "krea-dev": 25,
    "qwen": 20,
    "qwen-image": 20,
    "qwen-image-edit": 20,
    "qwen-image-layered": 50,
    "mage-flow-base": 30,
    "mageflow-base": 30,
    "microsoft/Mage-Flow-Base": 30,
    "mage-flow": 20,
    "mageflow": 20,
    "microsoft/Mage-Flow": 20,
    "mage-flow-turbo": 4,
    "mageflow-turbo": 4,
    "microsoft/Mage-Flow-Turbo": 4,
    "mage-flow-edit-base": 30,
    "mageflow-edit-base": 30,
    "microsoft/Mage-Flow-Edit-Base": 30,
    "mage-flow-edit": 30,
    "mageflow-edit": 30,
    "microsoft/Mage-Flow-Edit": 30,
    "mage-flow-edit-turbo": 4,
    "mageflow-edit-turbo": 4,
    "microsoft/Mage-Flow-Edit-Turbo": 4,
    "fibo": 50,
    "fibo-lite": 8,
    "fibo-edit": 50,
    "fibo-edit-rmbg": 10,
    "z-image": 50,
    "z-image-turbo": 9,
    "z-image-controlnet": 8,
    "krea-2": 8,
    "krea2": 8,
    "ernie-image-turbo": 8,
    "ernie-image": 50,
    "boogu-image-turbo": 4,
    "flux2-klein-4b": 4,
    "flux2-klein-9b": 4,
    "flux2-klein-9b-kv": 4,
    "flux2-klein-base-4b": 50,
    "flux2-klein-base-9b": 50,
    "ideogram4": 20,
    "ideogram-4-fp8": 20,
}
QUANTIZE_CHOICES = [3, 5, 4, 6, 8]

if os.environ.get("MFLUX_CACHE_DIR"):
    MFLUX_CACHE_DIR = Path(os.environ["MFLUX_CACHE_DIR"]).resolve()
else:
    MFLUX_CACHE_DIR = Path(platformdirs.user_cache_dir(appname="mflux"))

MFLUX_LORA_CACHE_DIR = MFLUX_CACHE_DIR / "loras"
