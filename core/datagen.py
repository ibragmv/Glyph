from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from core.console import create_progress, write_progress_line
from core.constants import (
    CODEPOINTS,
    DEFAULT_FONT_CANDIDATES,
    DEFAULT_FONT_SEARCH_PATHS,
    IMPERIAL_ARAMAIC_SYMBOLS,
    LABEL_DIRS,
)
from core.utils import ensure_dir, save_json, seed_everything

RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
TRANSFORM_AFFINE = Image.Transform.AFFINE
TRANSFORM_QUAD = Image.Transform.QUAD


@dataclass
class DataGenConfig:
    output_dir: Path
    train_per_class: int = 1200
    val_per_class: int = 300
    canvas_size: int = 128
    output_size: int = 64
    seed: int = 42
    explicit_fonts: tuple[Path, ...] = ()
    extra_font_dirs: tuple[Path, ...] = ()
    train_hardness: float = 0.95
    val_hardness: float = 1.35
    holdout_font_fraction: float = 0.35
    train_profiles: tuple[str, ...] = ()
    val_profiles: tuple[str, ...] = ()
    preview_samples_per_group: int = 16


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    severity_scale: float
    background_scale: float
    blur_scale: float
    contrast_scale: float
    dropout_scale: float
    crop_scale: float
    bleed_scale: float
    shadow_scale: float
    lighting_scale: float
    damage_scale: float
    warp_scale: float


GENERATION_PROFILES = {
    "clean": GenerationProfile(
        name="clean",
        severity_scale=0.72,
        background_scale=0.45,
        blur_scale=0.35,
        contrast_scale=0.45,
        dropout_scale=0.30,
        crop_scale=0.10,
        bleed_scale=0.15,
        shadow_scale=0.20,
        lighting_scale=0.35,
        damage_scale=0.30,
        warp_scale=0.50,
    ),
    "scan": GenerationProfile(
        name="scan",
        severity_scale=0.98,
        background_scale=0.90,
        blur_scale=0.95,
        contrast_scale=0.95,
        dropout_scale=0.85,
        crop_scale=0.55,
        bleed_scale=0.85,
        shadow_scale=0.85,
        lighting_scale=0.95,
        damage_scale=0.85,
        warp_scale=0.95,
    ),
    "damaged": GenerationProfile(
        name="damaged",
        severity_scale=1.18,
        background_scale=1.20,
        blur_scale=1.10,
        contrast_scale=1.15,
        dropout_scale=1.30,
        crop_scale=1.20,
        bleed_scale=1.05,
        shadow_scale=1.00,
        lighting_scale=1.10,
        damage_scale=1.45,
        warp_scale=1.10,
    ),
    "harsh": GenerationProfile(
        name="harsh",
        severity_scale=1.42,
        background_scale=1.35,
        blur_scale=1.35,
        contrast_scale=1.40,
        dropout_scale=1.55,
        crop_scale=1.55,
        bleed_scale=1.30,
        shadow_scale=1.35,
        lighting_scale=1.35,
        damage_scale=1.70,
        warp_scale=1.25,
    ),
}

DEFAULT_SPLIT_PROFILES = {
    "train": ("clean", "scan", "damaged"),
    "val": ("scan", "damaged", "harsh"),
}

DEFAULT_SPLIT_PROFILE_WEIGHTS = {
    "train": {"clean": 0.44, "scan": 0.38, "damaged": 0.18},
    "val": {"scan": 0.20, "damaged": 0.34, "harsh": 0.46},
}


def font_supports_codepoints(font_path: Path, codepoints: Iterable[int]) -> bool:
    try:
        ttfont = TTFont(str(font_path))
        cmap = set((ttfont.getBestCmap() or {}).keys())
        ttfont.close()
    except Exception:
        return False
    return all(codepoint in cmap for codepoint in codepoints)


def discover_fonts(
    explicit_fonts: Iterable[Path], extra_font_dirs: Iterable[Path]
) -> list[Path]:
    candidates: list[Path] = []

    for font_path in explicit_fonts:
        if font_path.exists():
            candidates.append(font_path)

    for font_dir in [*DEFAULT_FONT_SEARCH_PATHS, *extra_font_dirs]:
        if not font_dir.exists():
            continue
        for candidate_name in DEFAULT_FONT_CANDIDATES:
            font_path = font_dir / candidate_name
            if font_path.exists():
                candidates.append(font_path)
        for font_path in font_dir.rglob("*.ttf"):
            if "aramaic" in font_path.name.lower():
                candidates.append(font_path)
        for font_path in font_dir.rglob("*.otf"):
            if "aramaic" in font_path.name.lower():
                candidates.append(font_path)

    unique_candidates = sorted(set(candidates))
    supported = [
        font_path
        for font_path in unique_candidates
        if font_supports_codepoints(font_path, CODEPOINTS)
    ]
    if not supported:
        raise FileNotFoundError(
            "No TTF/OTF font with full Imperial Aramaic support was found. "
            "Provide --font or place the font into ./fonts."
        )
    return supported


def _center_by_mass(image: Image.Image) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    threshold = max(float(array.max()) * 0.15, 8.0)
    weights = np.where(array >= threshold, array, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return image

    y_grid, x_grid = np.indices(weights.shape)
    center_x = float((x_grid * weights).sum() / total)
    center_y = float((y_grid * weights).sum() / total)
    shift_x = int(round((weights.shape[1] / 2.0) - center_x))
    shift_y = int(round((weights.shape[0] / 2.0) - center_y))

    translated = ImageChops.offset(image, shift_x, shift_y)
    if shift_x > 0:
        translated.paste(0, (0, 0, shift_x, translated.height))
    elif shift_x < 0:
        translated.paste(
            0, (translated.width + shift_x, 0, translated.width, translated.height)
        )
    if shift_y > 0:
        translated.paste(0, (0, 0, translated.width, shift_y))
    elif shift_y < 0:
        translated.paste(
            0, (0, translated.height + shift_y, translated.width, translated.height)
        )
    return translated


def _severity_value(
    train_hardness: float,
    val_hardness: float,
    split: str,
) -> float:
    base = train_hardness if split == "train" else val_hardness
    return max(0.35, float(base))


def _resolve_profile_names(split: str, selected: tuple[str, ...]) -> tuple[str, ...]:
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


def _choose_profile(
    rng: np.random.Generator,
    split: str,
    profile_names: tuple[str, ...],
) -> GenerationProfile:
    weights = _profile_weights(split, profile_names)
    selected = str(rng.choice(np.asarray(profile_names, dtype=object), p=weights))
    return GENERATION_PROFILES[selected]


def _image_from_grayscale_array(array: np.ndarray) -> Image.Image:
    grayscale = np.asarray(array, dtype=np.uint8)
    if grayscale.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale array, received shape {grayscale.shape!r}."
        )
    return Image.fromarray(grayscale)


def _sample_noise_map(
    size: tuple[int, int],
    rng: np.random.Generator,
    min_grid: int,
    max_grid: int,
) -> np.ndarray:
    width, height = size
    low_w = int(rng.integers(min_grid, max_grid + 1))
    low_h = int(rng.integers(min_grid, max_grid + 1))
    noise = rng.normal(0.0, 1.0, size=(low_h, low_w)).astype(np.float32)
    noise = (noise - noise.min()) / max(float(noise.max() - noise.min()), 1e-6)
    noise_image = _image_from_grayscale_array(noise * 255.0)
    resized = noise_image.resize((width, height), resample=RESAMPLE_BILINEAR)
    return (np.asarray(resized, dtype=np.float32) / 127.5) - 1.0


def _paper_background(
    size: tuple[int, int],
    rng: np.random.Generator,
    severity: float,
    profile: GenerationProfile,
) -> np.ndarray:
    width, height = size
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    texture_scale = severity * profile.background_scale
    paper_level = float(rng.uniform(178.0, 244.0))
    coarse = (
        _sample_noise_map(size, rng, min_grid=4, max_grid=9)
        * float(rng.uniform(14.0, 34.0))
        * texture_scale
    )
    medium = (
        _sample_noise_map(size, rng, min_grid=10, max_grid=18)
        * float(rng.uniform(6.0, 18.0))
        * texture_scale
    )
    fine = rng.normal(
        0.0,
        float(rng.uniform(1.5, 5.8)) * texture_scale,
        size=(height, width),
    )

    angle = float(rng.uniform(0.0, np.pi))
    gradient = (
        np.cos(angle) * (x_grid - (width / 2.0))
        + np.sin(angle) * (y_grid - (height / 2.0))
    ) / max(width, height)
    gradient *= float(rng.uniform(24.0, 66.0)) * texture_scale

    fibers = np.zeros((height, width), dtype=np.float32)
    fiber_count = int(
        rng.integers(2, 6 + int(np.ceil(8.0 * profile.damage_scale * severity)))
    )
    for _ in range(fiber_count):
        x0 = int(rng.integers(0, width))
        x1 = int(np.clip(x0 + rng.integers(-32, 33), 0, width - 1))
        y0 = int(rng.integers(0, height))
        y1 = int(np.clip(y0 + rng.integers(-32, 33), 0, height - 1))
        line = Image.new("L", size, color=0)
        line_draw = ImageDraw.Draw(line)
        line_draw.line(
            (x0, y0, x1, y1),
            fill=int(rng.integers(6, 24)),
            width=int(rng.integers(1, 3)),
        )
        if rng.random() < 0.7:
            line = line.filter(
                ImageFilter.GaussianBlur(
                    radius=float(rng.uniform(0.35, 1.35) * profile.damage_scale)
                )
            )
        fibers += np.asarray(line, dtype=np.float32)

    paper = paper_level + coarse + medium + fine + gradient + fibers
    return np.clip(paper, 120.0, 255.0)


def _draw_glyph_mask(
    symbol_char: str,
    font_path: Path,
    config: DataGenConfig,
    rng: np.random.Generator,
    severity: float,
) -> Image.Image:
    canvas = Image.new("L", (config.canvas_size, config.canvas_size), color=0)
    draw = ImageDraw.Draw(canvas)

    font_size = int(rng.integers(low=68, high=108))
    stroke_width = int(rng.integers(low=0, high=4))
    font = ImageFont.truetype(str(font_path), size=font_size)

    bbox = draw.textbbox((0, 0), symbol_char, font=font, stroke_width=stroke_width)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    jitter = 5.0 + (2.5 * severity)
    offset_x = (
        (config.canvas_size - width) / 2 - bbox[0] + float(rng.uniform(-jitter, jitter))
    )
    offset_y = (
        (config.canvas_size - height) / 2
        - bbox[1]
        + float(rng.uniform(-jitter, jitter))
    )

    draw.text(
        (offset_x, offset_y),
        symbol_char,
        fill=255,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=255,
    )
    return canvas


def _random_affine(
    image: Image.Image, rng: np.random.Generator, severity: float
) -> Image.Image:
    shear = float(rng.uniform(-0.10, 0.10)) * severity
    scale = float(
        rng.uniform(max(0.72, 0.96 - 0.10 * severity), 1.04 + 0.08 * severity)
    )
    tx = float(rng.uniform(-4.0 - 3.5 * severity, 4.0 + 3.5 * severity))
    ty = float(rng.uniform(-4.0 - 3.5 * severity, 4.0 + 3.5 * severity))
    matrix = (scale, shear, tx, shear, scale, ty)
    return image.transform(
        image.size,
        TRANSFORM_AFFINE,
        matrix,
        resample=RESAMPLE_BILINEAR,
        fillcolor=0,
    )


def _random_quad(
    image: Image.Image, rng: np.random.Generator, severity: float
) -> Image.Image:
    width, height = image.size
    warp = 6.0 + (5.0 * severity)
    quad = [
        float(rng.uniform(-warp, warp)),
        float(rng.uniform(-warp, warp)),
        float(width + rng.uniform(-warp, warp)),
        float(rng.uniform(-warp, warp)),
        float(width + rng.uniform(-warp, warp)),
        float(height + rng.uniform(-warp, warp)),
        float(rng.uniform(-warp, warp)),
        float(height + rng.uniform(-warp, warp)),
    ]
    return image.transform(
        image.size,
        TRANSFORM_QUAD,
        quad,
        resample=RESAMPLE_BILINEAR,
        fillcolor=0,
    )


def _subtract_overlay(base: Image.Image, overlay: Image.Image) -> Image.Image:
    base_array = np.asarray(base, dtype=np.int16)
    overlay_array = np.asarray(overlay, dtype=np.int16)
    merged = np.clip(base_array - overlay_array, 0, 255).astype(np.uint8)
    return _image_from_grayscale_array(merged)


def _apply_stroke_loss(
    image: Image.Image,
    rng: np.random.Generator,
    severity: float,
    profile: GenerationProfile,
) -> Image.Image:
    overlay = Image.new("L", image.size, color=0)
    draw = ImageDraw.Draw(overlay)
    line_count = int(
        rng.integers(0, 2 + int(np.ceil(4.0 * severity * profile.dropout_scale)))
    )
    for _ in range(line_count):
        x0 = int(rng.integers(0, image.width))
        y0 = int(rng.integers(0, image.height))
        length = int(rng.integers(10, int(24 + 32 * severity * profile.dropout_scale)))
        angle = float(rng.uniform(0.0, np.pi))
        x1 = int(np.clip(x0 + np.cos(angle) * length, 0, image.width - 1))
        y1 = int(np.clip(y0 + np.sin(angle) * length, 0, image.height - 1))
        draw.line(
            (x0, y0, x1, y1),
            fill=int(rng.integers(160, 256)),
            width=int(rng.integers(1, max(2, int(2 + 4 * profile.dropout_scale)))),
        )
    if line_count == 0:
        return image
    overlay = overlay.filter(
        ImageFilter.GaussianBlur(
            radius=float(rng.uniform(0.3, 1.4) * profile.dropout_scale)
        )
    )
    return _subtract_overlay(image, overlay)


def _apply_mask_damage(
    image: Image.Image,
    rng: np.random.Generator,
    severity: float,
    split: str,
    profile: GenerationProfile,
) -> Image.Image:
    working = image

    filter_bias = profile.damage_scale * max(0.8, severity)
    if rng.random() < min(0.92, (0.42 if split == "train" else 0.66) * filter_bias):
        filter_size = 3 if rng.random() < 0.6 else 5
        working = working.filter(ImageFilter.MaxFilter(size=filter_size))
    if rng.random() < min(0.92, (0.34 if split == "train" else 0.62) * filter_bias):
        filter_size = 3 if rng.random() < 0.7 else 5
        working = working.filter(ImageFilter.MinFilter(size=filter_size))
    if rng.random() < min(0.95, (0.28 if split == "train" else 0.52) * filter_bias):
        working = working.filter(
            ImageFilter.GaussianBlur(
                radius=float(rng.uniform(0.35, 1.55) * severity * profile.blur_scale)
            )
        )

    draw = ImageDraw.Draw(working)
    dropout_count = int(
        rng.integers(0, 2 + int(np.ceil(3.0 * severity * profile.dropout_scale)))
    )
    for _ in range(dropout_count):
        if rng.random() > min(0.96, 0.26 + 0.24 * severity * profile.dropout_scale):
            continue
        x0 = int(rng.integers(0, max(1, working.width - 8)))
        y0 = int(rng.integers(0, max(1, working.height - 8)))
        w = int(rng.integers(4, int(14 + 14 * severity * profile.dropout_scale)))
        h = int(rng.integers(4, int(14 + 14 * severity * profile.dropout_scale)))
        draw.rectangle(
            (x0, y0, min(working.width, x0 + w), min(working.height, y0 + h)), fill=0
        )

    if rng.random() < min(0.98, 0.35 * profile.dropout_scale):
        working = _apply_stroke_loss(working, rng, severity, profile)

    if rng.random() < min(0.94, (0.22 if split == "train" else 0.52) * profile.crop_scale):
        margin = int(rng.integers(3, int(12 + 14 * severity * profile.crop_scale)))
        side = int(rng.integers(0, 4))
        if side == 0:
            draw.rectangle((0, 0, margin, working.height), fill=0)
        elif side == 1:
            draw.rectangle(
                (working.width - margin, 0, working.width, working.height), fill=0
            )
        elif side == 2:
            draw.rectangle((0, 0, working.width, margin), fill=0)
        else:
            draw.rectangle(
                (0, working.height - margin, working.width, working.height), fill=0
            )

    if rng.random() < min(0.95, 0.30 * profile.damage_scale):
        overlay = Image.new("L", working.size, color=0)
        overlay_draw = ImageDraw.Draw(overlay)
        patch_count = int(
            rng.integers(1, 3 + int(np.ceil(4.0 * severity * profile.damage_scale)))
        )
        for _ in range(patch_count):
            x0 = int(rng.integers(0, working.width))
            y0 = int(rng.integers(0, working.height))
            x1 = int(
                np.clip(
                    x0 + rng.integers(-12, int(18 + 18 * profile.damage_scale)),
                    0,
                    working.width,
                )
            )
            y1 = int(
                np.clip(
                    y0 + rng.integers(-12, int(18 + 18 * profile.damage_scale)),
                    0,
                    working.height,
                )
            )
            overlay_draw.ellipse(
                (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                fill=int(rng.integers(140, 256)),
            )
        overlay = overlay.filter(
            ImageFilter.GaussianBlur(
                radius=float(rng.uniform(0.8, 2.2) * profile.damage_scale)
            )
        )
        working = _subtract_overlay(working, overlay)

    return working


def _render_stains(
    size: tuple[int, int],
    rng: np.random.Generator,
    severity: float,
    profile: GenerationProfile,
) -> np.ndarray:
    overlay = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(overlay)
    stain_count = int(
        rng.integers(1, 3 + int(np.ceil(4.0 * severity * profile.damage_scale)))
    )
    for _ in range(stain_count):
        radius_x = int(
            rng.integers(8, 26 + int(22 * severity * profile.damage_scale))
        )
        radius_y = int(
            rng.integers(8, 26 + int(22 * severity * profile.damage_scale))
        )
        center_x = int(rng.integers(0, size[0]))
        center_y = int(rng.integers(0, size[1]))
        strength = int(rng.integers(10, 68))
        draw.ellipse(
            (
                center_x - radius_x,
                center_y - radius_y,
                center_x + radius_x,
                center_y + radius_y,
            ),
            fill=strength,
        )

    overlay = overlay.filter(
        ImageFilter.GaussianBlur(
            radius=float(rng.uniform(3.0, 9.5) * severity * profile.damage_scale)
        )
    )
    return np.asarray(overlay, dtype=np.float32)


def _apply_illumination_and_shadow(
    image_array: np.ndarray,
    size: tuple[int, int],
    rng: np.random.Generator,
    severity: float,
    profile: GenerationProfile,
) -> np.ndarray:
    width, height = size
    lighting_strength = severity * profile.lighting_scale
    shadow_strength = severity * profile.shadow_scale

    illumination = _sample_noise_map(size, rng, 3, 7) * float(
        rng.uniform(18.0, 52.0)
    ) * lighting_strength
    illumination += _sample_noise_map(size, rng, 8, 14) * float(
        rng.uniform(6.0, 18.0)
    ) * lighting_strength

    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    axis_angle = float(rng.uniform(0.0, np.pi))
    axis_gradient = (
        np.cos(axis_angle) * (x_grid - (width / 2.0))
        + np.sin(axis_angle) * (y_grid - (height / 2.0))
    ) / max(width, height)
    illumination += axis_gradient * float(rng.uniform(18.0, 48.0)) * lighting_strength
    image_array += illumination

    if rng.random() < min(0.96, 0.36 + 0.26 * profile.shadow_scale):
        shadow = np.zeros((height, width), dtype=np.float32)
        shadow_count = int(
            rng.integers(1, 2 + int(np.ceil(3.0 * severity * profile.shadow_scale)))
        )
        for _ in range(shadow_count):
            radius_x = int(
                rng.integers(24, 54 + int(40 * severity * profile.shadow_scale))
            )
            radius_y = int(
                rng.integers(24, 54 + int(40 * severity * profile.shadow_scale))
            )
            center_x = int(rng.integers(-radius_x // 3, width + radius_x // 3))
            center_y = int(rng.integers(-radius_y // 3, height + radius_y // 3))
            blob = Image.new("L", size, color=0)
            blob_draw = ImageDraw.Draw(blob)
            blob_draw.ellipse(
                (
                    center_x - radius_x,
                    center_y - radius_y,
                    center_x + radius_x,
                    center_y + radius_y,
                ),
                fill=int(rng.integers(48, 150)),
            )
            blob = blob.filter(
                ImageFilter.GaussianBlur(
                    radius=float(rng.uniform(8.0, 18.0) * profile.shadow_scale)
                )
            )
            shadow += np.asarray(blob, dtype=np.float32)

        if rng.random() < 0.7:
            edge_side = int(rng.integers(0, 4))
            edge_overlay = Image.new("L", size, color=0)
            edge_draw = ImageDraw.Draw(edge_overlay)
            thickness = int(
                rng.integers(10, int(26 + 18 * severity * profile.shadow_scale))
            )
            strength = int(rng.integers(36, 110))
            if edge_side == 0:
                edge_draw.rectangle((0, 0, thickness, height), fill=strength)
            elif edge_side == 1:
                edge_draw.rectangle((width - thickness, 0, width, height), fill=strength)
            elif edge_side == 2:
                edge_draw.rectangle((0, 0, width, thickness), fill=strength)
            else:
                edge_draw.rectangle((0, height - thickness, width, height), fill=strength)
            edge_overlay = edge_overlay.filter(
                ImageFilter.GaussianBlur(
                    radius=float(rng.uniform(4.0, 10.0) * profile.shadow_scale)
                )
            )
            shadow += np.asarray(edge_overlay, dtype=np.float32)

        image_array -= shadow * float(rng.uniform(0.28, 0.75)) * shadow_strength

    return image_array


def _apply_surface_damage(
    image: Image.Image,
    rng: np.random.Generator,
    severity: float,
    profile: GenerationProfile,
) -> Image.Image:
    if rng.random() >= min(0.97, 0.26 + 0.24 * profile.damage_scale):
        return image

    overlay = Image.new("L", image.size, color=0)
    draw = ImageDraw.Draw(overlay)
    count = int(rng.integers(1, 3 + int(np.ceil(5.0 * severity * profile.damage_scale))))
    for _ in range(count):
        if rng.random() < 0.55:
            x0 = int(rng.integers(0, image.width))
            y0 = int(rng.integers(0, image.height))
            x1 = int(np.clip(x0 + rng.integers(-18, 19), 0, image.width - 1))
            y1 = int(
                np.clip(
                    y0 + rng.integers(10, int(34 + 24 * profile.damage_scale)),
                    0,
                    image.height - 1,
                )
            )
            draw.line(
                (x0, y0, x1, y1),
                fill=int(rng.integers(28, 96)),
                width=int(rng.integers(1, 3)),
            )
        else:
            radius_x = int(
                rng.integers(10, int(24 + 24 * severity * profile.damage_scale))
            )
            radius_y = int(
                rng.integers(10, int(24 + 24 * severity * profile.damage_scale))
            )
            center_x = int(rng.integers(0, image.width))
            center_y = int(rng.integers(0, image.height))
            draw.ellipse(
                (
                    center_x - radius_x,
                    center_y - radius_y,
                    center_x + radius_x,
                    center_y + radius_y,
                ),
                fill=int(rng.integers(18, 72)),
            )

    overlay = overlay.filter(
        ImageFilter.GaussianBlur(
            radius=float(rng.uniform(1.4, 4.6) * profile.damage_scale)
        )
    )
    damaged = np.asarray(image, dtype=np.float32)
    damaged += np.asarray(overlay, dtype=np.float32) * float(rng.uniform(-0.6, 1.1))
    return _image_from_grayscale_array(np.clip(damaged, 0.0, 255.0))


def _apply_hard_crop(
    image: Image.Image,
    rng: np.random.Generator,
    severity: float,
    profile: GenerationProfile,
) -> Image.Image:
    if rng.random() >= min(0.98, 0.12 + 0.34 * profile.crop_scale):
        return image

    max_trim = max(2, int(round((5 + 10 * severity) * profile.crop_scale)))
    left = int(rng.integers(0, max_trim + 1)) if rng.random() < 0.7 else 0
    right = int(rng.integers(0, max_trim + 1)) if rng.random() < 0.7 else 0
    top = int(rng.integers(0, max_trim + 1)) if rng.random() < 0.7 else 0
    bottom = int(rng.integers(0, max_trim + 1)) if rng.random() < 0.7 else 0
    if left + right + top + bottom == 0:
        side = int(rng.integers(0, 4))
        trim = max(1, int(rng.integers(1, max_trim + 1)))
        if side == 0:
            left = trim
        elif side == 1:
            right = trim
        elif side == 2:
            top = trim
        else:
            bottom = trim

    cropped = image.crop((left, top, image.width - right, image.height - bottom))
    return cropped.resize(image.size, resample=RESAMPLE_BILINEAR)


def _build_preview_sheet(images: list[Image.Image], tile_size: int) -> Image.Image:
    columns = 4
    rows = int(np.ceil(len(images) / columns))
    gutter = 4
    canvas = Image.new(
        "L",
        (
            columns * tile_size + (columns + 1) * gutter,
            rows * tile_size + (rows + 1) * gutter,
        ),
        color=248,
    )
    for index, image in enumerate(images):
        x = index % columns
        y = index // columns
        tile = image.resize((tile_size, tile_size), resample=RESAMPLE_LANCZOS)
        canvas.paste(
            tile,
            (
                gutter + x * (tile_size + gutter),
                gutter + y * (tile_size + gutter),
            ),
        )
    return canvas


def _save_preview_sheets(
    output_dir: Path,
    preview_buckets: dict[tuple[str, str], list[Image.Image]],
    tile_size: int,
) -> dict[str, dict[str, str]]:
    preview_dir = ensure_dir(output_dir / "previews")
    preview_paths: dict[str, dict[str, str]] = {}
    for (split, profile_name), images in sorted(preview_buckets.items()):
        if not images:
            continue
        preview_path = preview_dir / f"{split}_{profile_name}.png"
        _build_preview_sheet(images, tile_size).save(preview_path)
        preview_paths.setdefault(split, {})[profile_name] = str(
            preview_path.relative_to(output_dir).as_posix()
        )
    return preview_paths


def _compose_scan(
    glyph_mask: Image.Image,
    config: DataGenConfig,
    rng: np.random.Generator,
    severity: float,
    split: str,
    profile: GenerationProfile,
) -> Image.Image:
    paper = _paper_background(glyph_mask.size, rng, severity, profile)
    glyph = np.asarray(glyph_mask, dtype=np.float32) / 255.0
    ink_strength = float(rng.uniform(92.0, 178.0)) * (0.90 + 0.18 * severity)
    ink_texture = 1.0 + (
        _sample_noise_map(glyph_mask.size, rng, 8, 18)
        * float(rng.uniform(0.08, 0.28))
        * severity
        * profile.contrast_scale
    )
    composed = paper - (glyph * ink_strength * ink_texture)

    if rng.random() < min(0.98, (0.30 if split == "train" else 0.62) * profile.bleed_scale):
        ghost = glyph_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if rng.random() < 0.65:
            ghost = ghost.rotate(
                float(rng.uniform(-6.0, 6.0) * profile.bleed_scale),
                resample=RESAMPLE_BILINEAR,
                fillcolor=0,
            )
        ghost = ImageChops.offset(
            ghost,
            int(rng.integers(-10, 11)),
            int(rng.integers(-10, 11)),
        )
        ghost = ghost.filter(
            ImageFilter.GaussianBlur(
                radius=float(rng.uniform(0.8, 2.6) * profile.bleed_scale)
            )
        )
        ghost_array = np.asarray(ghost, dtype=np.float32) / 255.0
        composed -= (
            ghost_array
            * float(rng.uniform(10.0, 38.0))
            * severity
            * profile.bleed_scale
        )

    stains = _render_stains(glyph_mask.size, rng, severity, profile)
    if rng.random() < 0.5:
        composed += stains * float(rng.uniform(0.25, 0.70))
    else:
        composed -= stains * float(rng.uniform(0.15, 0.55))

    row_noise = rng.normal(
        0.0, float(rng.uniform(0.6, 2.8)) * severity, size=(glyph_mask.height, 1)
    )
    col_noise = rng.normal(
        0.0, float(rng.uniform(0.6, 2.2)) * severity, size=(1, glyph_mask.width)
    )
    composed += row_noise + col_noise

    speckles = rng.normal(
        0.0,
        float(rng.uniform(1.2, 5.8)) * severity,
        size=(glyph_mask.height, glyph_mask.width),
    )
    composed += speckles
    composed = _apply_illumination_and_shadow(
        composed, glyph_mask.size, rng, severity, profile
    )

    composed = np.clip(composed, 0.0, 255.0)
    image = _image_from_grayscale_array(composed)

    if rng.random() < min(0.98, (0.40 if split == "train" else 0.72) * profile.blur_scale):
        downscale_factor = float(
            rng.uniform(
                max(0.24, 0.62 - 0.22 * severity * profile.blur_scale),
                0.88,
            )
        )
        down_w = max(18, int(round(image.width * downscale_factor)))
        down_h = max(18, int(round(image.height * downscale_factor)))
        image = image.resize((down_w, down_h), resample=RESAMPLE_BILINEAR)
        image = image.resize(
            (config.canvas_size, config.canvas_size), resample=RESAMPLE_BILINEAR
        )

    if rng.random() < min(0.98, (0.34 if split == "train" else 0.64) * profile.blur_scale):
        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=float(rng.uniform(0.25, 2.25) * severity * profile.blur_scale)
            )
        )

    image = _apply_surface_damage(image, rng, severity, profile)
    image = _apply_hard_crop(image, rng, severity, profile)

    contrast = float(
        rng.uniform(
            max(0.28, 1.04 - 0.58 * severity * profile.contrast_scale),
            1.08 + 0.10 * severity * profile.contrast_scale,
        )
    )
    brightness = float(
        rng.uniform(
            max(0.58, 1.00 - 0.28 * severity * profile.lighting_scale),
            1.08 + 0.08 * severity * profile.lighting_scale,
        )
    )
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    return image


def render_symbol(
    symbol_char: str,
    font_path: Path,
    config: DataGenConfig,
    rng: np.random.Generator,
    split: str,
    profile: GenerationProfile,
) -> Image.Image:
    severity = (
        _severity_value(config.train_hardness, config.val_hardness, split)
        * profile.severity_scale
    )
    glyph_mask = _draw_glyph_mask(symbol_char, font_path, config, rng, severity)

    if rng.random() < 0.90:
        glyph_mask = _random_affine(glyph_mask, rng, severity)
    if rng.random() < 0.90:
        angle = float(rng.uniform(-(9.0 + 5.0 * severity), 9.0 + 5.0 * severity))
        glyph_mask = glyph_mask.rotate(angle, resample=RESAMPLE_BILINEAR, fillcolor=0)
    if rng.random() < min(0.98, (0.34 if split == "train" else 0.72) * profile.warp_scale):
        glyph_mask = _random_quad(glyph_mask, rng, severity)

    glyph_mask = _center_by_mass(glyph_mask)
    glyph_mask = _apply_mask_damage(glyph_mask, rng, severity, split, profile)
    glyph_mask = _center_by_mass(glyph_mask)

    image = _compose_scan(glyph_mask, config, rng, severity, split, profile)
    image = image.resize(
        (config.output_size, config.output_size), resample=RESAMPLE_LANCZOS
    )
    return image


def _split_fonts_for_domains(
    font_paths: list[Path],
    rng: np.random.Generator,
    holdout_fraction: float,
) -> dict[str, list[Path]]:
    if len(font_paths) < 2:
        return {"train": list(font_paths), "val": list(font_paths)}

    shuffled = list(font_paths)
    rng.shuffle(shuffled)
    holdout_fraction = min(max(float(holdout_fraction), 0.0), 0.9)
    holdout_count = max(1, int(round(len(shuffled) * holdout_fraction)))
    holdout_count = min(holdout_count, len(shuffled) - 1)
    val_fonts = shuffled[-holdout_count:]
    train_fonts = shuffled[:-holdout_count]
    return {"train": train_fonts, "val": val_fonts}


def build_dataset(
    config: DataGenConfig,
) -> dict:
    seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)
    font_paths = discover_fonts(config.explicit_fonts, config.extra_font_dirs)
    split_fonts = _split_fonts_for_domains(
        font_paths, rng, config.holdout_font_fraction
    )
    split_profiles = {
        "train": _resolve_profile_names("train", config.train_profiles),
        "val": _resolve_profile_names("val", config.val_profiles),
    }
    preview_buckets = {
        (split, profile_name): []
        for split, profile_names in split_profiles.items()
        for profile_name in profile_names
    }
    sample_manifest: list[dict[str, str | int]] = []
    profile_histogram = {
        split: {profile_name: 0 for profile_name in profile_names}
        for split, profile_names in split_profiles.items()
    }

    total_per_class = config.train_per_class + config.val_per_class
    total_images = total_per_class * len(IMPERIAL_ARAMAIC_SYMBOLS)
    for split in ("train", "val"):
        for label_dir in LABEL_DIRS.values():
            ensure_dir(config.output_dir / split / label_dir)

    total_symbols = len(IMPERIAL_ARAMAIC_SYMBOLS)
    overall_progress = create_progress(
        command="gen",
        scope="render",
        color="green",
        leave=True,
        total=total_images,
    )
    overall_progress.set_postfix_str(f"ready 0/{total_symbols}")
    overall_progress.refresh()
    for symbol_idx, symbol in enumerate(IMPERIAL_ARAMAIC_SYMBOLS, start=1):
        label_dir = LABEL_DIRS[symbol["index"]]
        for sample_idx in range(total_per_class):
            split = "train" if sample_idx < config.train_per_class else "val"
            split_idx = (
                sample_idx if split == "train" else sample_idx - config.train_per_class
            )
            split_font_paths = split_fonts[split] or font_paths
            profile = _choose_profile(rng, split, split_profiles[split])
            font_path = split_font_paths[int(rng.integers(0, len(split_font_paths)))]
            image = render_symbol(
                symbol_char=symbol["char"],
                font_path=font_path,
                config=config,
                rng=rng,
                split=split,
                profile=profile,
            )
            output_path = (
                config.output_dir
                / split
                / label_dir
                / f"{label_dir}_{split_idx:04d}.png"
            )
            image.save(output_path)
            profile_histogram[split][profile.name] += 1
            if (
                len(preview_buckets[(split, profile.name)])
                < config.preview_samples_per_group
            ):
                preview_buckets[(split, profile.name)].append(image.copy())
            sample_manifest.append(
                {
                    "path": str(output_path.relative_to(config.output_dir).as_posix()),
                    "split": split,
                    "label": symbol["name"],
                    "profile": profile.name,
                    "font": font_path.name,
                }
            )
            overall_progress.update(1)
            overall_progress.set_postfix_str(
                (
                    f"ready {symbol_idx - 1}/{total_symbols} | current {label_dir} "
                    f"| split {split} | profile {profile.name}"
                ),
                refresh=False,
            )

        ready_line = (
            f"gen ready {label_dir} | letters {symbol_idx}/{total_symbols} | images {total_per_class}"
        )
        write_progress_line(overall_progress, ready_line)

    overall_progress.set_postfix_str(f"ready {total_symbols}/{total_symbols}")
    overall_progress.close()
    preview_sheets = _save_preview_sheets(
        config.output_dir, preview_buckets, config.output_size
    )

    metadata = {
        "unicode_range": ["U+10840", "U+10855"],
        "num_classes": len(IMPERIAL_ARAMAIC_SYMBOLS),
        "train_per_class": config.train_per_class,
        "val_per_class": config.val_per_class,
        "canvas_size": config.canvas_size,
        "output_size": config.output_size,
        "seed": config.seed,
        "fonts": [str(path) for path in font_paths],
        "split_fonts": {
            split: [str(path) for path in split_font_paths]
            for split, split_font_paths in split_fonts.items()
        },
        "train_hardness": config.train_hardness,
        "val_hardness": config.val_hardness,
        "holdout_font_fraction": config.holdout_font_fraction,
        "generation_profiles": {
            name: profile.__dict__ for name, profile in GENERATION_PROFILES.items()
        },
        "split_profiles": {
            split: {
                "available": list(profile_names),
                "weights": {
                    name: weight
                    for name, weight in zip(
                        profile_names, _profile_weights(split, profile_names)
                    )
                },
            }
            for split, profile_names in split_profiles.items()
        },
        "profile_histogram": profile_histogram,
        "preview_sheets": preview_sheets,
        "domain_shift": {
            "train": (
                "mixed synthetic render domain using lighter clean/scan/damaged "
                "profiles with broad augmentation"
            ),
            "val": (
                "heavier scan-like validation domain using scan/damaged/harsh "
                "profiles with stronger corruption and optional held-out fonts"
            ),
        },
        "classes": IMPERIAL_ARAMAIC_SYMBOLS,
        "total_images": total_images,
        "sample_manifest": sample_manifest,
    }
    save_json(config.output_dir / "metadata.json", metadata)
    return metadata
