from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Protocol

import mlx.core as mx


@dataclass
class LoRATarget:
    model_path: str
    possible_up_patterns: List[str]
    possible_down_patterns: List[str]
    possible_alpha_patterns: List[str] = field(default_factory=list)
    possible_lokr_w1_patterns: List[str] = field(default_factory=list)
    possible_lokr_w2_patterns: List[str] = field(default_factory=list)
    possible_dora_scale_patterns: List[str] = field(default_factory=list)
    up_transform: Callable[[mx.array], mx.array] | None = None
    down_transform: Callable[[mx.array], mx.array] | None = None
    lokr_w1_transform: Callable[[mx.array], mx.array] | None = None
    lokr_w2_transform: Callable[[mx.array], mx.array] | None = None


# Trailing matrix suffixes a LoRA/LyCORIS pattern can carry, longest-first so a base is
# stripped cleanly (e.g. ".lora_A.default.weight" before ".weight" would matter if present).
_MATRIX_SUFFIXES = (
    ".lora_A.weight",
    ".lora_A.default.weight",
    ".lora_B.weight",
    ".lora_B.default.weight",
    ".lora_down.weight",
    ".lora_down.default.weight",
    ".lora_up.weight",
    ".lora_up.default.weight",
    ".lora.down.weight",
    ".lora.down.default.weight",
    ".lora.up.weight",
    ".lora.up.default.weight",
)


def derive_lokr_patterns(targets: List[LoRATarget]) -> List[LoRATarget]:
    """Populate each target's LyCORIS/LoKr factor patterns (lokr_w1 / lokr_w2) from its
    existing up/down matrix patterns, unless it already defines them. A LoKr adapter stores
    lokr_w1/lokr_w2 instead of lora_A/lora_B, so a mapping that only lists matrix patterns
    would silently skip it. Deriving from the same bases keeps prefix handling in one place
    and avoids a hand-written parallel list per module. Returns the same list for chaining."""
    for target in targets:
        if target.possible_lokr_w1_patterns or target.possible_lokr_w2_patterns:
            continue
        bases: List[str] = []
        seen: set[str] = set()
        for pattern in list(target.possible_down_patterns) + list(target.possible_up_patterns):
            base: str | None = None
            for suffix in _MATRIX_SUFFIXES:
                if pattern.endswith(suffix):
                    base = pattern[: -len(suffix)]
                    break
            if base is None or base in seen:
                continue
            seen.add(base)
            bases.append(base)
        target.possible_lokr_w1_patterns = [f"{base}.lokr_w1" for base in bases]
        target.possible_lokr_w2_patterns = [f"{base}.lokr_w2" for base in bases]
    return targets


class LoRAMapping(Protocol):
    @staticmethod
    def get_mapping() -> List[LoRATarget]:
        return
