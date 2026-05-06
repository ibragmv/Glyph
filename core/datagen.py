from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from core.console import create_progress, write_progress_line
from core.source import build_source_manifest, get_class_assets, get_texture_paths, split_real_paths
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


def _severity_value(train_hardness: float, val_hardness: float, split: str) -> float:
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
    rng: np.random.Generator, split: str, profile_names: tuple[str, ...]
) -> GenerationProfile:
    weights = _profile_weights(split, profile_names)
    selected = str(rng.choice(np.asarray(profile_names, dtype=object), p=weights))
    return GENERATION_PROFILES[selected]


def _open_grayscale(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if "A" in image.getbands():
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image.convert("RGBA"))
        return image.convert("L")


def _threshold_mask(image: Image.Image) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    border = np.concatenate(
        [
            array[0, :],
            array[-1, :],
            array[:, 0],
            array[:, -1],
        ]
    )
    background = float(np.median(border)) if border.size else 255.0
    normalized = array.copy()
    if background < 128:
        normalized = 255.0 - normalized
        background = 255.0 - background

    threshold = min(background - 12.0, 230.0)
    threshold = max(threshold, 32.0)
    darkness = np.clip(background - normalized, 0.0, 255.0)
    mask = np.where(normalized <= threshold, darkness, 0.0)
    if mask.max() <= 0.0:
        mask = np.clip(255.0 - normalized, 0.0, 255.0)
    return Image.fromarray(mask.astype(np.uint8))


def _crop_mask(mask: Image.Image) -> Image.Image:
    bbox = mask.getbbox()
    if bbox is None:
        return mask
    return mask.crop(bbox)


def _fit_mask(mask: Image.Image, canvas_size: int, scale: float) -> Image.Image:
    target = Image.new("L", (canvas_size, canvas_size), 0)
    cropped = _crop_mask(mask)
    width, height = cropped.size
    if width == 0 or height == 0:
        return target

    max_dim = max(width, height)
    fit_ratio = (canvas_size * scale) / float(max_dim)
    resized = cropped.resize(
        (
            max(4, int(round(width * fit_ratio))),
            max(4, int(round(height * fit_ratio))),
        ),
        RESAMPLE_LANCZOS,
    )
    offset = (
        (canvas_size - resized.width) // 2,
        (canvas_size - resized.height) // 2,
    )
    target.paste(resized, offset)
    return target


def _apply_affine(
    mask: Image.Image,
    rng: np.random.Generator,
    severity: float,
    profile: GenerationProfile,
) -> Image.Image:
    angle = float(rng.uniform(-20.0, 20.0) * severity * profile.warp_scale)
    scale = float(rng.uniform(0.84, 1.14))
    shear_x = float(rng.uniform(-0.20, 0.20) * severity * profile.warp_scale)
    shear_y = float(rng.uniform(-0.10, 0.10) * severity * profile.warp_scale)
    translate_x = float(rng.uniform(-7.0, 7.0) * severity)
    translate_y = float(rng.uniform(-7.0, 7.0) * severity)

    rotated = mask.rotate(angle, resample=RESAMPLE_BILINEAR, fillcolor=0)
    matrix = (
        scale,
        shear_x,
        translate_x,
        shear_y,
        scale,
        translate_y,
    )
    return rotated.transform(
        rotated.size,
        TRANSFORM_AFFINE,
        matrix,
        resample=RESAMPLE_BILINEAR,
        fillcolor=0,
    )


def _apply_perspective(
    mask: Image.Image,
    rng: np.random.Generator,
    severity: float,
    profile: GenerationProfile,
) -> Image.Image:
    width, height = mask.size
    margin = int(round((4 + 10 * severity) * profile.warp_scale))
    src = [
        (rng.integers(0, margin + 1), rng.integers(0, margin + 1)),
        (width - rng.integers(0, margin + 1), rng.integers(0, margin + 1)),
        (width - rng.integers(0, margin + 1), height - rng.integers(0, margin + 1)),
        (rng.integers(0, margin + 1), height - rng.integers(0, margin + 1)),
    ]
    quad = tuple(float(value) for point in src for value in point)
    return mask.transform(
        mask.size,
        TRANSFORM_QUAD,
        quad,
        resample=RESAMPLE_BILINEAR,
        fillcolor=0,
    )


def _apply_damage(
    mask: Image.Image,
    rng: np.random.Generator,
    severity: float,
    split: str,
    profile: GenerationProfile,
) -> Image.Image:
    damaged = mask.copy()
    draw = ImageDraw.Draw(damaged)
    width, height = damaged.size
    holes = int(round(rng.uniform(1, 4) * severity * profile.damage_scale))
    for _ in range(max(1, holes)):
        if rng.random() > (0.30 if split == "train" else 0.55):
            continue
        hole_w = int(round(rng.uniform(4, 16) * severity * profile.damage_scale))
        hole_h = int(round(rng.uniform(4, 18) * severity * profile.damage_scale))
        x0 = int(rng.integers(0, max(1, width - hole_w)))
        y0 = int(rng.integers(0, max(1, height - hole_h)))
        draw.rectangle((x0, y0, x0 + hole_w, y0 + hole_h), fill=0)

    if rng.random() < min(0.95, 0.30 * severity * profile.damage_scale):
        damaged = damaged.filter(ImageFilter.MaxFilter(size=3))
    if rng.random() < min(0.95, 0.40 * severity * profile.damage_scale):
        damaged = damaged.filter(ImageFilter.MinFilter(size=3))

    if rng.random() < min(0.98, 0.36 * severity * profile.crop_scale):
        crop = int(round(rng.uniform(2, 8) * severity * profile.crop_scale))
        cropped = Image.new("L", damaged.size, 0)
        offset_x = int(rng.integers(-crop, crop + 1))
        offset_y = int(rng.integers(-crop, crop + 1))
        cropped.paste(damaged, (offset_x, offset_y))
        damaged = cropped

    return damaged


def _pick_background(
    texture_paths: tuple[Path, ...],
    rng: np.random.Generator,
    size: int,
    severity: float,
    profile: GenerationProfile,
) -> Image.Image:
    if texture_paths and rng.random() < min(0.98, 0.72 * profile.texture_scale):
        texture_path = texture_paths[int(rng.integers(0, len(texture_paths)))]
        texture = _open_grayscale(texture_path).resize((size, size), RESAMPLE_LANCZOS)
        texture = ImageOps.autocontrast(texture)
        texture = ImageEnhance.Contrast(texture).enhance(
            0.72 + (severity * 0.22 * profile.texture_scale)
        )
        texture = ImageEnhance.Brightness(texture).enhance(
            1.05 + (rng.uniform(-0.08, 0.10))
        )
        return texture

    base = np.full((size, size), 236.0, dtype=np.float32)
    base += rng.normal(0.0, 7.5 * severity * profile.texture_scale, size=(size, size))
    gradient_x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    gradient_y = np.linspace(-1.0, 1.0, size, dtype=np.float32)[:, None]
    base += gradient_x * rng.uniform(-18.0, 18.0) * severity * profile.lighting_scale
    base += gradient_y * rng.uniform(-18.0, 18.0) * severity * profile.lighting_scale
    array = np.clip(base, 140.0, 255.0).astype(np.uint8)
    return Image.fromarray(array).filter(ImageFilter.GaussianBlur(radius=1.4))


def _compose_scan(
    mask: Image.Image,
    texture_paths: tuple[Path, ...],
    rng: np.random.Generator,
    severity: float,
    split: str,
    profile: GenerationProfile,
    output_size: int,
) -> Image.Image:
    background = _pick_background(
        texture_paths=texture_paths,
        rng=rng,
        size=mask.size[0],
        severity=severity,
        profile=profile,
    )
    glyph_strength = np.asarray(mask, dtype=np.float32) / 255.0
    background_arr = np.asarray(background, dtype=np.float32)
    ink_level = float(20.0 + rng.uniform(0.0, 45.0))
    composite = background_arr - (glyph_strength * (background_arr - ink_level))

    if rng.random() < min(0.98, 0.42 * severity * profile.lighting_scale):
        shadow = np.linspace(
            rng.uniform(0.94, 1.04),
            rng.uniform(0.78, 1.08),
            composite.shape[1],
            dtype=np.float32,
        )
        composite *= shadow[None, :]

    if rng.random() < min(0.98, 0.42 * severity * profile.texture_scale):
        column_noise = rng.normal(
            0.0,
            4.0 * severity * profile.texture_scale,
            size=(1, composite.shape[1]),
        )
        composite += column_noise

    if rng.random() < min(0.98, 0.42 * severity * profile.texture_scale):
        row_noise = rng.normal(
            0.0,
            4.0 * severity * profile.texture_scale,
            size=(composite.shape[0], 1),
        )
        composite += row_noise

    composite = np.clip(composite, 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(composite)

    blur_limit = (0.55 if split == "train" else 0.85) * severity * profile.blur_scale
    if rng.random() < min(0.98, 0.42 * profile.blur_scale):
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.5, 2.4) * blur_limit)))
    if rng.random() < min(0.98, 0.28 * profile.blur_scale):
        image = image.filter(ImageFilter.MedianFilter(size=3))

    image = image.resize((output_size, output_size), RESAMPLE_LANCZOS)
    if rng.random() < min(0.98, 0.38 * severity * profile.texture_scale):
        down = max(28, int(round(output_size * rng.uniform(0.52, 0.88))))
        image = image.resize((down, down), RESAMPLE_BILINEAR).resize(
            (output_size, output_size), RESAMPLE_LANCZOS
        )

    return ImageOps.autocontrast(image)


def _choose_source_path(
    *,
    alphabet_path: Path,
    exemplar_paths: tuple[Path, ...],
    real_paths: tuple[Path, ...],
    rng: np.random.Generator,
) -> Path:
    weighted: list[Path] = []
    weighted.extend(exemplar_paths)
    weighted.extend(exemplar_paths)
    weighted.extend(real_paths)
    if alphabet_path.is_file():
        weighted.append(alphabet_path)
    if not weighted:
        raise FileNotFoundError("No source image is available for the requested class.")
    return weighted[int(rng.integers(0, len(weighted)))]


def _prepare_source_mask(
    source_path: Path,
    *,
    canvas_size: int,
    rng: np.random.Generator,
) -> Image.Image:
    grayscale = _open_grayscale(source_path)
    mask = _threshold_mask(grayscale)
    scale = float(rng.uniform(0.62, 0.84))
    return _fit_mask(mask, canvas_size=canvas_size, scale=scale)


def render_symbol(
    *,
    alphabet_path: Path,
    exemplar_paths: tuple[Path, ...],
    real_paths: tuple[Path, ...],
    texture_paths: tuple[Path, ...],
    config: DataGenConfig,
    rng: np.random.Generator,
    split: str,
    profile: GenerationProfile,
) -> Image.Image:
    severity = _severity_value(config.train_hardness, config.val_hardness, split)
    source_path = _choose_source_path(
        alphabet_path=alphabet_path,
        exemplar_paths=exemplar_paths,
        real_paths=real_paths,
        rng=rng,
    )
    mask = _prepare_source_mask(
        source_path,
        canvas_size=config.canvas_size,
        rng=rng,
    )
    mask = _apply_affine(mask, rng, severity, profile)
    if rng.random() < min(0.98, 0.36 * severity * profile.warp_scale):
        mask = _apply_perspective(mask, rng, severity, profile)
    mask = _apply_damage(mask, rng, severity, split, profile)
    return _compose_scan(
        mask=mask,
        texture_paths=texture_paths,
        rng=rng,
        severity=severity,
        split=split,
        profile=profile,
        output_size=config.output_size,
    )


def _prepare_real_image(source_path: Path, output_size: int) -> Image.Image:
    mask = _threshold_mask(_open_grayscale(source_path))
    fitted = _fit_mask(mask, canvas_size=max(output_size * 2, 128), scale=0.80)
    return _compose_scan(
        mask=fitted,
        texture_paths=(),
        rng=np.random.default_rng(0),
        severity=0.60,
        split="val",
        profile=GENERATION_PROFILES["clean"],
        output_size=output_size,
    )


def _make_preview_sheet(images: list[Image.Image]) -> Image.Image:
    tile_size = images[0].size[0]
    columns = min(4, len(images))
    rows = int(np.ceil(len(images) / columns))
    sheet = Image.new("L", (columns * tile_size, rows * tile_size), 245)
    for index, image in enumerate(images):
        x = (index % columns) * tile_size
        y = (index // columns) * tile_size
        sheet.paste(image, (x, y))
    return sheet


def _save_preview_sheets(output_dir: Path, preview_buckets: dict) -> dict:
    preview_dir = ensure_dir(output_dir / "preview")
    preview_paths: dict[str, dict[str, str]] = {}
    for (split, profile_name), images in sorted(preview_buckets.items()):
        if not images:
            continue
        preview_path = preview_dir / f"{split}_{profile_name}.png"
        _make_preview_sheet(images).save(preview_path)
        preview_paths.setdefault(split, {})[profile_name] = str(preview_path)
    return preview_paths


def build_dataset(config: DataGenConfig) -> dict:
    seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)
    ensure_dir(config.output_dir)

    class_assets = get_class_assets()
    texture_paths = get_texture_paths()
    if not class_assets:
        raise FileNotFoundError("No source assets were found in ./source.")

    split_profiles = {
        "train": _resolve_profile_names("train", config.train_profiles),
        "val": _resolve_profile_names("val", config.val_profiles),
    }
    preview_buckets = {
        (split, profile_name): []
        for split, profile_names in split_profiles.items()
        for profile_name in profile_names
    }
    profile_histogram = {
        split: {profile_name: 0 for profile_name in profile_names}
        for split, profile_names in split_profiles.items()
    }

    for split in ("train", "val", "realtrain", "realval"):
        for assets in class_assets:
            ensure_dir(config.output_dir / split / assets.label_dir)

    total_synthetic = (config.train_per_class + config.val_per_class) * len(class_assets)
    progress = create_progress(
        range(total_synthetic),
        command="gen",
        scope="build",
        color="green",
        leave=False,
        total=total_synthetic,
    )

    sample_index = 0
    for assets in class_assets:
        real_split = split_real_paths(
            assets.real_paths,
            seed=config.seed + assets.index,
            val_fraction=config.real_val_fraction,
        )

        for split_name, paths in real_split.items():
            for index, source_path in enumerate(paths):
                image = _prepare_real_image(source_path, config.output_size)
                out_path = (
                    config.output_dir
                    / split_name
                    / assets.label_dir
                    / f"{assets.label_dir}_{index:04d}.png"
                )
                image.save(out_path)

        for split_name, count in (("train", config.train_per_class), ("val", config.val_per_class)):
            for split_index in range(count):
                profile = _choose_profile(rng, split_name, split_profiles[split_name])
                image = render_symbol(
                    alphabet_path=assets.alphabet_path,
                    exemplar_paths=assets.exemplar_paths,
                    real_paths=assets.real_paths,
                    texture_paths=texture_paths,
                    config=config,
                    rng=rng,
                    split=split_name,
                    profile=profile,
                )
                out_path = (
                    config.output_dir
                    / split_name
                    / assets.label_dir
                    / f"{assets.label_dir}_{split_index:04d}.png"
                )
                image.save(out_path)
                profile_histogram[split_name][profile.name] += 1
                if (
                    len(preview_buckets[(split_name, profile.name)])
                    < config.preview_samples_per_group
                ):
                    preview_buckets[(split_name, profile.name)].append(image.copy())
                progress.update(1)
                progress.set_postfix_str(
                    (
                        f"class {assets.name} | "
                        f"split {split_name} | "
                        f"profile {profile.name}"
                    ),
                    refresh=False,
                )
                sample_index += 1

        write_progress_line(
            progress,
            (
                f"gen ready {assets.label_dir} | "
                f"synthetic {config.train_per_class + config.val_per_class} | "
                f"real {len(real_split['realtrain']) + len(real_split['realval'])}"
            ),
        )

    progress.close()
    preview_sheets = _save_preview_sheets(config.output_dir, preview_buckets)
    source_manifest = build_source_manifest(
        seed=config.seed,
        real_val_fraction=config.real_val_fraction,
    )

    realtrain_total = sum(
        len(list((config.output_dir / "realtrain" / assets.label_dir).glob("*.png")))
        for assets in class_assets
    )
    realval_total = sum(
        len(list((config.output_dir / "realval" / assets.label_dir).glob("*.png")))
        for assets in class_assets
    )
    metadata = {
        "version": 2,
        "num_classes": len(class_assets),
        "class_names": [assets.name for assets in class_assets],
        "class_titles": [assets.title for assets in class_assets],
        "train_per_class": config.train_per_class,
        "val_per_class": config.val_per_class,
        "synthetic_total_images": total_synthetic,
        "real_total_images": realtrain_total + realval_total,
        "total_images": total_synthetic + realtrain_total + realval_total,
        "primary_validation_split": "realval" if realval_total > 0 else "val",
        "real_val_fraction": config.real_val_fraction,
        "train_profiles": list(split_profiles["train"]),
        "val_profiles": list(split_profiles["val"]),
        "profile_histogram": profile_histogram,
        "preview_sheets": preview_sheets,
        "source": source_manifest,
        "notes": [
            "Synthetic images are generated from canonical exemplars and real crops.",
            "Real images are copied into dataset/realtrain and dataset/realval.",
            "Primary validation should use realval when available.",
        ],
    }
    save_json(config.output_dir / "metadata.json", metadata)
    return metadata
