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

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_LANCZOS = Image.LANCZOS


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
    noise_image = Image.fromarray((noise * 255.0).astype(np.uint8), mode="L")
    resized = noise_image.resize((width, height), resample=RESAMPLE_BILINEAR)
    return (np.asarray(resized, dtype=np.float32) / 127.5) - 1.0


def _paper_background(
    size: tuple[int, int],
    rng: np.random.Generator,
    severity: float,
) -> np.ndarray:
    width, height = size
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    paper_level = float(rng.uniform(184.0, 242.0))
    coarse = (
        _sample_noise_map(size, rng, min_grid=4, max_grid=9)
        * float(rng.uniform(14.0, 34.0))
        * severity
    )
    medium = (
        _sample_noise_map(size, rng, min_grid=10, max_grid=18)
        * float(rng.uniform(6.0, 18.0))
        * severity
    )
    fine = rng.normal(
        0.0, float(rng.uniform(1.5, 5.0)) * severity, size=(height, width)
    )

    angle = float(rng.uniform(0.0, np.pi))
    gradient = (
        np.cos(angle) * (x_grid - (width / 2.0))
        + np.sin(angle) * (y_grid - (height / 2.0))
    ) / max(width, height)
    gradient *= float(rng.uniform(24.0, 54.0)) * severity

    paper = paper_level + coarse + medium + fine + gradient
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
        Image.AFFINE,
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
        Image.QUAD,
        quad,
        resample=RESAMPLE_BILINEAR,
        fillcolor=0,
    )


def _apply_mask_damage(
    image: Image.Image, rng: np.random.Generator, severity: float, split: str
) -> Image.Image:
    working = image

    if rng.random() < (0.45 if split == "train" else 0.70):
        filter_size = 3 if rng.random() < 0.6 else 5
        working = working.filter(ImageFilter.MaxFilter(size=filter_size))
    if rng.random() < (0.35 if split == "train" else 0.65):
        filter_size = 3 if rng.random() < 0.7 else 5
        working = working.filter(ImageFilter.MinFilter(size=filter_size))
    if rng.random() < (0.30 if split == "train" else 0.55):
        working = working.filter(
            ImageFilter.GaussianBlur(radius=float(rng.uniform(0.35, 1.20) * severity))
        )

    draw = ImageDraw.Draw(working)
    dropout_count = int(rng.integers(0, 2 + int(np.ceil(2.0 * severity))))
    for _ in range(dropout_count):
        if rng.random() > (0.30 + 0.25 * severity):
            continue
        x0 = int(rng.integers(0, max(1, working.width - 8)))
        y0 = int(rng.integers(0, max(1, working.height - 8)))
        w = int(rng.integers(4, int(14 + 10 * severity)))
        h = int(rng.integers(4, int(14 + 10 * severity)))
        draw.rectangle(
            (x0, y0, min(working.width, x0 + w), min(working.height, y0 + h)), fill=0
        )

    if rng.random() < (0.25 if split == "train" else 0.60):
        margin = int(rng.integers(3, int(12 + 8 * severity)))
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

    return working


def _render_stains(
    size: tuple[int, int],
    rng: np.random.Generator,
    severity: float,
) -> np.ndarray:
    overlay = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(overlay)
    stain_count = int(rng.integers(1, 3 + int(np.ceil(3.0 * severity))))
    for _ in range(stain_count):
        radius_x = int(rng.integers(8, 26 + int(18 * severity)))
        radius_y = int(rng.integers(8, 26 + int(18 * severity)))
        center_x = int(rng.integers(0, size[0]))
        center_y = int(rng.integers(0, size[1]))
        strength = int(rng.integers(10, 52))
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
        ImageFilter.GaussianBlur(radius=float(rng.uniform(3.0, 8.0) * severity))
    )
    return np.asarray(overlay, dtype=np.float32)


def _compose_scan(
    glyph_mask: Image.Image,
    config: DataGenConfig,
    rng: np.random.Generator,
    severity: float,
    split: str,
) -> Image.Image:
    paper = _paper_background(glyph_mask.size, rng, severity)
    glyph = np.asarray(glyph_mask, dtype=np.float32) / 255.0
    ink_strength = float(rng.uniform(92.0, 178.0)) * (0.90 + 0.18 * severity)
    ink_texture = 1.0 + (
        _sample_noise_map(glyph_mask.size, rng, 8, 18)
        * float(rng.uniform(0.08, 0.28))
        * severity
    )
    composed = paper - (glyph * ink_strength * ink_texture)

    if rng.random() < (0.40 if split == "train" else 0.75):
        ghost = glyph_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        ghost = ImageChops.offset(
            ghost,
            int(rng.integers(-8, 9)),
            int(rng.integers(-8, 9)),
        )
        ghost_array = np.asarray(ghost, dtype=np.float32) / 255.0
        composed -= ghost_array * float(rng.uniform(8.0, 28.0)) * severity

    stains = _render_stains(glyph_mask.size, rng, severity)
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

    composed = np.clip(composed, 0.0, 255.0)
    image = Image.fromarray(composed.astype(np.uint8), mode="L")

    if rng.random() < (0.45 if split == "train" else 0.75):
        downscale_factor = float(rng.uniform(max(0.28, 0.62 - 0.18 * severity), 0.88))
        down_w = max(18, int(round(image.width * downscale_factor)))
        down_h = max(18, int(round(image.height * downscale_factor)))
        image = image.resize((down_w, down_h), resample=RESAMPLE_BILINEAR)
        image = image.resize(
            (config.canvas_size, config.canvas_size), resample=RESAMPLE_BILINEAR
        )

    if rng.random() < (0.40 if split == "train" else 0.72):
        image = image.filter(
            ImageFilter.GaussianBlur(radius=float(rng.uniform(0.25, 1.70) * severity))
        )

    contrast = float(rng.uniform(max(0.45, 1.02 - 0.40 * severity), 1.05))
    brightness = float(
        rng.uniform(max(0.70, 1.00 - 0.18 * severity), 1.06 + 0.06 * severity)
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
) -> Image.Image:
    severity = _severity_value(config.train_hardness, config.val_hardness, split)
    glyph_mask = _draw_glyph_mask(symbol_char, font_path, config, rng, severity)

    if rng.random() < 0.90:
        glyph_mask = _random_affine(glyph_mask, rng, severity)
    if rng.random() < 0.90:
        angle = float(rng.uniform(-(9.0 + 5.0 * severity), 9.0 + 5.0 * severity))
        glyph_mask = glyph_mask.rotate(angle, resample=RESAMPLE_BILINEAR, fillcolor=0)
    if rng.random() < (0.40 if split == "train" else 0.80):
        glyph_mask = _random_quad(glyph_mask, rng, severity)

    glyph_mask = _center_by_mass(glyph_mask)
    glyph_mask = _apply_mask_damage(glyph_mask, rng, severity, split)
    glyph_mask = _center_by_mass(glyph_mask)

    image = _compose_scan(glyph_mask, config, rng, severity, split)
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

    total_per_class = config.train_per_class + config.val_per_class
    total_images = total_per_class * len(IMPERIAL_ARAMAIC_SYMBOLS)
    for split in ("train", "val"):
        for label_dir in LABEL_DIRS.values():
            ensure_dir(config.output_dir / split / label_dir)

    total_symbols = len(IMPERIAL_ARAMAIC_SYMBOLS)
    overall_progress = create_progress(
        command="data",
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
            image = render_symbol(
                symbol_char=symbol["char"],
                font_path=split_font_paths[int(rng.integers(0, len(split_font_paths)))],
                config=config,
                rng=rng,
                split=split,
            )
            output_path = (
                config.output_dir
                / split
                / label_dir
                / f"{label_dir}_{split_idx:04d}.png"
            )
            image.save(output_path)
            overall_progress.update(1)
            overall_progress.set_postfix_str(
                f"ready {symbol_idx - 1}/{total_symbols} | current {label_dir} | split {split}",
                refresh=False,
            )

        ready_line = (
            f"data ready {label_dir} | letters {symbol_idx}/{total_symbols} | images {total_per_class}"
        )
        write_progress_line(overall_progress, ready_line)

    overall_progress.set_postfix_str(f"ready {total_symbols}/{total_symbols}")
    overall_progress.close()

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
        "domain_shift": {
            "train": "mixed synthetic scan artifacts with broad augmentation",
            "val": "harder scan-like render domain with stronger corruption and optional held-out fonts",
        },
        "classes": IMPERIAL_ARAMAIC_SYMBOLS,
        "total_images": total_images,
    }
    save_json(config.output_dir / "metadata.json", metadata)
    return metadata
