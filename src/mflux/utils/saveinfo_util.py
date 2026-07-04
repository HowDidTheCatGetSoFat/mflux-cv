from datetime import datetime
from pathlib import Path


def nodot(val) -> str:
    # Encode a numeric value as a filename-safe tag without a literal dot, e.g. 1.05 -> "1p05",
    # 10.5 -> "10p5", 105 -> "105". Replaces the dot with 'p' rather than stripping it, which
    # collided distinct values (1.05 and 10.5 both became "105") and defeated --saveinfo's purpose
    # of encoding exact run parameters in the filename.
    return f"{val:g}".replace(".", "p")


def build_saveinfo_filename(args, seed) -> str:
    # Build an output path that encodes the run parameters (seed, steps, LoRA(s), scheduler, sigma
    # options) in the filename. Shared by the Z-Image and Z-Image-Turbo CLIs.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sched_tag = getattr(args, "scheduler", "linear")
    sa_parts = []
    if getattr(args, "cosine", False):
        sa_parts.append("cos")
    if getattr(args, "karras", False):
        sa_parts.append("karras")
    if getattr(args, "exponential", False):
        sa_parts.append("exp")
    if getattr(args, "shift", None) is not None:
        sa_parts.append(f"shift_{nodot(args.shift)}")
    if getattr(args, "mcf_max_change", None) is not None:
        sa_parts.append(f"mcf_{nodot(args.mcf_max_change)}")
    sa = "_".join(sa_parts)
    lora_tag = "NoLora"
    if getattr(args, "lora_paths", None):
        names = [Path(p).stem for p in args.lora_paths]
        lora_tag = "+".join(names)
        if getattr(args, "lora_scales", None):
            sc = "+".join(nodot(s) for s in args.lora_scales)
            lora_tag = f"{lora_tag}-{sc}"
    parts = [timestamp, str(seed), f"S{args.steps}", lora_tag, sched_tag]
    if sa:
        parts.append(sa)
    output_dir = str(Path(args.output).parent)
    return str(Path(output_dir) / ("_".join(parts) + ".png"))
