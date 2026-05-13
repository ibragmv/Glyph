from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DataGenConfig:
    output_dir: Path
    train_per_class: int = 1200
    val_per_class: int = 300
    canvas_size: int = 144
    output_size: int = 64
    seed: int = 42
    train_hardness: float = 0.95
    val_hardness: float = 1.35
    real_val_fraction: float = 0.25
    train_profiles: tuple[str, ...] = ()
    val_profiles: tuple[str, ...] = ()
    preview_samples_per_group: int = 16


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    severity_scale: float
    blur_scale: float
    warp_scale: float
    crop_scale: float
    damage_scale: float
    texture_scale: float
    lighting_scale: float
    color_scale: float


@dataclass(frozen=True)
class ClassBuildResult:
    index: int
    label_dir: str
    name: str
    synthetic_count: int
    real_count: int
    profile_histogram: dict[str, dict[str, int]]
    previews: dict[str, dict[str, list[np.ndarray]]]


GENERATION_PROFILES = {
    "clean": GenerationProfile(
        name="clean",
        severity_scale=0.70,
        blur_scale=0.35,
        warp_scale=0.45,
        crop_scale=0.18,
        damage_scale=0.22,
        texture_scale=0.35,
        lighting_scale=0.40,
        color_scale=0.35,
    ),
    "scan": GenerationProfile(
        name="scan",
        severity_scale=1.00,
        blur_scale=0.85,
        warp_scale=0.95,
        crop_scale=0.55,
        damage_scale=0.65,
        texture_scale=0.90,
        lighting_scale=0.90,
        color_scale=0.75,
    ),
    "damaged": GenerationProfile(
        name="damaged",
        severity_scale=1.20,
        blur_scale=1.05,
        warp_scale=1.15,
        crop_scale=0.95,
        damage_scale=1.30,
        texture_scale=1.10,
        lighting_scale=1.10,
        color_scale=0.95,
    ),
    "harsh": GenerationProfile(
        name="harsh",
        severity_scale=1.45,
        blur_scale=1.30,
        warp_scale=1.25,
        crop_scale=1.30,
        damage_scale=1.55,
        texture_scale=1.25,
        lighting_scale=1.25,
        color_scale=1.10,
    ),
}

DEFAULT_SPLIT_PROFILES = {
    "train": ("clean", "scan", "damaged"),
    "val": ("scan", "damaged", "harsh"),
}

DEFAULT_SPLIT_PROFILE_WEIGHTS = {
    "train": {"clean": 0.42, "scan": 0.38, "damaged": 0.20},
    "val": {"scan": 0.22, "damaged": 0.34, "harsh": 0.44},
}


def severity_value(train_hardness: float, val_hardness: float, split: str) -> float:
    base = train_hardness if split == "train" else val_hardness
    return max(0.35, float(base))


def resolve_profile_names(split: str, selected: tuple[str, ...]) -> tuple[str, ...]:
    names = selected or DEFAULT_SPLIT_PROFILES[split]
    normalized = []
    for name in names:
        profile_name = name.strip().lower()
        if not profile_name:
            continue
        if profile_name not in GENERATION_PROFILES:
            raise ValueError(
                f"Unknown generation profile {name!r}. "
                f"Available: {', '.join(sorted(GENERATION_PROFILES))}"
            )
        if profile_name not in normalized:
            normalized.append(profile_name)
    if not normalized:
        raise ValueError(f"No generation profiles configured for split {split!r}.")
    return tuple(normalized)


def _profile_weights(split: str, profile_names: tuple[str, ...]) -> list[float]:
    default_weights = DEFAULT_SPLIT_PROFILE_WEIGHTS[split]
    weights = [float(default_weights.get(name, 1.0)) for name in profile_names]
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(profile_names)] * len(profile_names)
    return [weight / total for weight in weights]


def choose_profile(
    rng: np.random.Generator, split: str, profile_names: tuple[str, ...]
) -> GenerationProfile:
    weights = _profile_weights(split, profile_names)
    selected = str(rng.choice(np.asarray(profile_names, dtype=object), p=weights))
    return GENERATION_PROFILES[selected]
