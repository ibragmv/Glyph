from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from core.console import create_progress, write_progress_line
from core.constants import R_TRAIN_SPLIT, R_VAL_SPLIT, TRAIN_SPLIT, VAL_SPLIT
from core.datagen_profiles import (
    ClassBuildResult,
    DataGenConfig,
    GenerationProfile,
    choose_profile,
    resolve_profile_names,
    severity_value,
)
from core.runtime import resolve_num_workers
from core.source import (
    ClassAssets,
    build_source_manifest,
    get_class_assets,
    get_texture_paths,
    split_real_paths,
)
from core.utils import ensure_dir, save_json, seed_everything

RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
TRANSFORM_AFFINE = Image.Transform.AFFINE
TRANSFORM_QUAD = Image.Transform.QUAD
RGB_WHITE = (248, 245, 239)
RGBA_TRANSPARENT = (255, 255, 255, 0)


def _open_rgba(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGBA")


def _trim_to_content(image: Image.Image) -> Image.Image:
    alpha_bbox = image.getchannel("A").getbbox()
    if alpha_bbox is not None:
        return image.crop(alpha_bbox)

    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, background)
    bbox = diff.convert("L").point(lambda value: 255 if value > 12 else 0).getbbox()
    if bbox is None:
        return image
    return image.crop(bbox)


def _derive_alpha_from_lightness(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getbbox() is not None:
        return rgba

    rgb = rgba.convert("RGB")
    luma = np.asarray(rgb.convert("L"), dtype=np.float32)
    alpha_values = np.clip((255.0 - luma) * 2.8, 0.0, 255.0).astype(np.uint8)
    rgba.putalpha(Image.fromarray(alpha_values))
    return rgba


def _fit_overlay(
    overlay: Image.Image,
    *,
    canvas_size: int,
    scale: float,
) -> Image.Image:
    target = Image.new("RGBA", (canvas_size, canvas_size), RGBA_TRANSPARENT)
    overlay = _trim_to_content(_derive_alpha_from_lightness(overlay))
    width, height = overlay.size
    if width == 0 or height == 0:
        return target

    fit_ratio = (canvas_size * scale) / float(max(width, height))
    resized = overlay.resize(
        (
            max(8, int(round(width * fit_ratio))),
            max(8, int(round(height * fit_ratio))),
        ),
        RESAMPLE_LANCZOS,
    )
    offset = (
        (canvas_size - resized.width) // 2,
        (canvas_size - resized.height) // 2,
    )
    target.alpha_composite(resized, offset)
    return target


def _apply_affine(
    overlay: Image.Image,
    rng: np.random.Generator,
    severity: float,
    profile: GenerationProfile,
) -> Image.Image:
    angle = float(rng.uniform(-18.0, 18.0) * severity * profile.warp_scale)
    scale = float(rng.uniform(0.86, 1.10))
    shear_x = float(rng.uniform(-0.16, 0.16) * severity * profile.warp_scale)
    shear_y = float(rng.uniform(-0.08, 0.08) * severity * profile.warp_scale)
    translate_x = float(rng.uniform(-8.0, 8.0) * severity)
    translate_y = float(rng.uniform(-8.0, 8.0) * severity)

    rotated = overlay.rotate(
        angle, resample=RESAMPLE_BILINEAR, fillcolor=RGBA_TRANSPARENT
    )
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
        fillcolor=RGBA_TRANSPARENT,
    )


def _apply_perspective(
    overlay: Image.Image,
    rng: np.random.Generator,
    severity: float,
    profile: GenerationProfile,
) -> Image.Image:
    width, height = overlay.size
    margin = int(round((4 + 10 * severity) * profile.warp_scale))
    src = [
        (rng.integers(0, margin + 1), rng.integers(0, margin + 1)),
        (width - rng.integers(0, margin + 1), rng.integers(0, margin + 1)),
        (width - rng.integers(0, margin + 1), height - rng.integers(0, margin + 1)),
        (rng.integers(0, margin + 1), height - rng.integers(0, margin + 1)),
    ]
    quad = tuple(float(value) for point in src for value in point)
    return overlay.transform(
        overlay.size,
        TRANSFORM_QUAD,
        quad,
        resample=RESAMPLE_BILINEAR,
        fillcolor=RGBA_TRANSPARENT,
    )


def _apply_overlay_damage(
    overlay: Image.Image,
    rng: np.random.Generator,
    severity: float,
    profile: GenerationProfile,
) -> Image.Image:
    damaged = overlay.copy()
    draw = ImageDraw.Draw(damaged)
    width, height = damaged.size
    patches = int(round(rng.uniform(1, 4) * severity * profile.damage_scale))
    for _ in range(max(1, patches)):
        if rng.random() > min(0.96, 0.42 * profile.damage_scale):
            continue
        patch_w = int(round(rng.uniform(6, 20) * severity * profile.damage_scale))
        patch_h = int(round(rng.uniform(6, 20) * severity * profile.damage_scale))
        x0 = int(rng.integers(0, max(1, width - patch_w)))
        y0 = int(rng.integers(0, max(1, height - patch_h)))
        draw.rectangle((x0, y0, x0 + patch_w, y0 + patch_h), fill=RGBA_TRANSPARENT)
    return damaged


def _tint_overlay(
    overlay: Image.Image,
    rng: np.random.Generator,
    severity: float,
    profile: GenerationProfile,
) -> Image.Image:
    rgb = overlay.convert("RGB")
    rgb = ImageEnhance.Color(rgb).enhance(
        0.80 + rng.uniform(-0.10, 0.22) * profile.color_scale
    )
    rgb = ImageEnhance.Contrast(rgb).enhance(
        1.00 + rng.uniform(-0.10, 0.20) * profile.color_scale
    )
    rgb = ImageEnhance.Brightness(rgb).enhance(
        0.98 + rng.uniform(-0.08, 0.12) * severity * profile.color_scale
    )
    tinted = rgb.convert("RGBA")
    tinted.putalpha(overlay.getchannel("A"))
    return tinted


def _pick_background(
    texture_paths: tuple[Path, ...],
    rng: np.random.Generator,
    size: int,
    severity: float,
    profile: GenerationProfile,
) -> Image.Image:
    if texture_paths and rng.random() < min(0.98, 0.76 * profile.texture_scale):
        texture_path = texture_paths[int(rng.integers(0, len(texture_paths)))]
        texture = (
            _open_rgba(texture_path)
            .convert("RGB")
            .resize((size, size), RESAMPLE_LANCZOS)
        )
        texture = ImageOps.autocontrast(texture)
        texture = ImageEnhance.Color(texture).enhance(
            0.88 + rng.uniform(-0.08, 0.18) * profile.color_scale
        )
        texture = ImageEnhance.Contrast(texture).enhance(
            0.85 + rng.uniform(0.00, 0.24) * severity * profile.texture_scale
        )
        return texture

    base = np.full((size, size, 3), RGB_WHITE, dtype=np.float32)
    base += rng.normal(0.0, 6.5 * severity * profile.texture_scale, size=base.shape)
    gradient_x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    gradient_y = np.linspace(-1.0, 1.0, size, dtype=np.float32)[:, None]
    base[..., 0] += gradient_x[None, :] * rng.uniform(-14.0, 14.0)
    base[..., 1] += gradient_y * rng.uniform(-12.0, 12.0)
    base[..., 2] += gradient_x[None, :] * rng.uniform(-10.0, 10.0)
    return Image.fromarray(np.clip(base, 120.0, 255.0).astype(np.uint8))


def _compose_image(
    overlay: Image.Image,
    *,
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
        size=overlay.size[0],
        severity=severity,
        profile=profile,
    )
    composite = background.convert("RGBA")
    composite.alpha_composite(overlay)
    image = composite.convert("RGB")

    if rng.random() < min(0.96, 0.45 * severity * profile.lighting_scale):
        width, height = image.size
        band = np.linspace(
            rng.uniform(0.94, 1.06),
            rng.uniform(0.78, 1.08),
            width,
            dtype=np.float32,
        )
        shaded = np.asarray(image, dtype=np.float32)
        shaded *= band[None, :, None]
        image = Image.fromarray(np.clip(shaded, 0.0, 255.0).astype(np.uint8))

    if rng.random() < min(0.94, 0.38 * severity * profile.texture_scale):
        noisy = np.asarray(image, dtype=np.float32)
        noisy += rng.normal(
            0.0, 4.0 * severity * profile.texture_scale, size=noisy.shape
        )
        image = Image.fromarray(np.clip(noisy, 0.0, 255.0).astype(np.uint8))

    blur_limit = (0.55 if split == "train" else 0.85) * severity * profile.blur_scale
    if rng.random() < min(0.96, 0.46 * profile.blur_scale):
        image = image.filter(
            ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 2.0) * blur_limit))
        )
    if rng.random() < min(0.94, 0.20 * profile.blur_scale):
        image = image.filter(ImageFilter.MedianFilter(size=3))

    image = image.resize((output_size, output_size), RESAMPLE_LANCZOS)
    if rng.random() < min(0.92, 0.34 * severity * profile.texture_scale):
        down = max(28, int(round(output_size * rng.uniform(0.56, 0.88))))
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
    severity = severity_value(config.train_hardness, config.val_hardness, split)
    source_path = _choose_source_path(
        alphabet_path=alphabet_path,
        exemplar_paths=exemplar_paths,
        real_paths=real_paths,
        rng=rng,
    )
    overlay = _fit_overlay(
        _open_rgba(source_path),
        canvas_size=config.canvas_size,
        scale=float(rng.uniform(0.60, 0.84)),
    )
    overlay = _apply_affine(overlay, rng, severity, profile)
    if rng.random() < min(0.96, 0.34 * severity * profile.warp_scale):
        overlay = _apply_perspective(overlay, rng, severity, profile)
    overlay = _apply_overlay_damage(overlay, rng, severity, profile)
    overlay = _tint_overlay(overlay, rng, severity, profile)
    return _compose_image(
        overlay,
        texture_paths=texture_paths,
        rng=rng,
        severity=severity,
        split=split,
        profile=profile,
        output_size=config.output_size,
    )


def _fit_rgb_square(image: Image.Image, output_size: int) -> Image.Image:
    target = Image.new("RGB", (output_size, output_size), RGB_WHITE)
    image = ImageOps.exif_transpose(image.convert("RGB"))
    image = ImageOps.autocontrast(image)
    fit = ImageOps.contain(image, (output_size, output_size), RESAMPLE_LANCZOS)
    offset = (
        (output_size - fit.width) // 2,
        (output_size - fit.height) // 2,
    )
    target.paste(fit, offset)
    return target


def _prepare_real_image(source_path: Path, output_size: int) -> Image.Image:
    with Image.open(source_path) as image:
        prepared = _fit_rgb_square(image, output_size)
    prepared = ImageEnhance.Color(prepared).enhance(1.02)
    prepared = ImageEnhance.Contrast(prepared).enhance(1.04)
    return prepared


def _make_preview_sheet(images: list[Image.Image]) -> Image.Image:
    tile_size = images[0].size[0]
    columns = min(4, len(images))
    rows = int(np.ceil(len(images) / columns))
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), RGB_WHITE)
    for index, image in enumerate(images):
        x = (index % columns) * tile_size
        y = (index // columns) * tile_size
        sheet.paste(image, (x, y))
    return sheet


def _preview_image_arrays(images: list[Image.Image]) -> list[np.ndarray]:
    return [np.asarray(image, dtype=np.uint8) for image in images]


def _image_from_array(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8))


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


def _reset_output_dir(output_dir: Path, class_assets: list[ClassAssets]) -> None:
    managed_paths = [output_dir / "metadata.json", output_dir / "preview"]
    for split in (TRAIN_SPLIT, VAL_SPLIT, R_TRAIN_SPLIT, R_VAL_SPLIT):
        for assets in class_assets:
            managed_paths.append(output_dir / split / assets.label_dir)

    for path in managed_paths:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _merge_preview_buckets(
    preview_buckets: dict[tuple[str, str], list[Image.Image]],
    preview_payload: dict[str, dict[str, list[np.ndarray]]],
    limit: int,
) -> None:
    for split_name, split_payload in preview_payload.items():
        for profile_name, arrays in split_payload.items():
            bucket = preview_buckets[(split_name, profile_name)]
            if len(bucket) >= limit:
                continue
            remaining = limit - len(bucket)
            bucket.extend(_image_from_array(array) for array in arrays[:remaining])


def _build_class_dataset(
    assets: ClassAssets,
    config: DataGenConfig,
    texture_paths: tuple[Path, ...],
    split_profiles: dict[str, tuple[str, ...]],
) -> ClassBuildResult:
    rng = np.random.default_rng(config.seed + assets.index)
    preview_buckets = {
        split: {profile_name: [] for profile_name in profile_names}
        for split, profile_names in split_profiles.items()
    }
    profile_histogram = {
        split: {profile_name: 0 for profile_name in profile_names}
        for split, profile_names in split_profiles.items()
    }

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

    for split_name, count in (
        ("train", config.train_per_class),
        ("val", config.val_per_class),
    ):
        for split_index in range(count):
            profile = choose_profile(rng, split_name, split_profiles[split_name])
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
                len(preview_buckets[split_name][profile.name])
                < config.preview_samples_per_group
            ):
                preview_buckets[split_name][profile.name].append(image.copy())

    return ClassBuildResult(
        index=assets.index,
        label_dir=assets.label_dir,
        name=assets.name,
        synthetic_count=config.train_per_class + config.val_per_class,
        real_count=len(real_split[R_TRAIN_SPLIT]) + len(real_split[R_VAL_SPLIT]),
        profile_histogram=profile_histogram,
        previews={
            split_name: {
                profile_name: _preview_image_arrays(images)
                for profile_name, images in split_payload.items()
                if images
            }
            for split_name, split_payload in preview_buckets.items()
        },
    )


def _resolve_gen_jobs(total_classes: int) -> int:
    cpu_jobs = resolve_num_workers(None, min_auto_workers=2, max_auto_workers=8)
    if total_classes <= 1:
        return 1
    return max(1, min(total_classes, cpu_jobs))


def build_dataset(config: DataGenConfig) -> dict:
    seed_everything(config.seed)
    ensure_dir(config.output_dir)
    if not 0.0 <= config.real_val_fraction <= 1.0:
        raise ValueError("real_val_fraction must be in [0, 1]")

    class_assets = get_class_assets()
    texture_paths = get_texture_paths()
    if not class_assets:
        raise FileNotFoundError("No source assets were found in ./source.")
    _reset_output_dir(config.output_dir, class_assets)

    split_profiles = {
        TRAIN_SPLIT: resolve_profile_names(TRAIN_SPLIT, config.train_profiles),
        VAL_SPLIT: resolve_profile_names(VAL_SPLIT, config.val_profiles),
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

    for split in (TRAIN_SPLIT, VAL_SPLIT, R_TRAIN_SPLIT, R_VAL_SPLIT):
        for assets in class_assets:
            ensure_dir(config.output_dir / split / assets.label_dir)

    total_synthetic = (config.train_per_class + config.val_per_class) * len(
        class_assets
    )
    progress = create_progress(
        range(total_synthetic),
        command="gen",
        scope="run",
        color="green",
        leave=False,
        total=total_synthetic,
    )

    jobs = _resolve_gen_jobs(len(class_assets))
    progress.set_postfix_str(f"classes 0/{len(class_assets)} | jobs {jobs}")
    progress.refresh()

    completed_classes = 0
    next_result_index = 0
    pending_results: dict[int, ClassBuildResult] = {}

    def handle_result(result: ClassBuildResult) -> None:
        nonlocal completed_classes, next_result_index
        completed_classes += 1
        progress.update(result.synthetic_count)
        for split_name, split_histogram in result.profile_histogram.items():
            for profile_name, count in split_histogram.items():
                profile_histogram[split_name][profile_name] += count
        _merge_preview_buckets(
            preview_buckets, result.previews, config.preview_samples_per_group
        )
        progress.set_postfix_str(
            f"class {result.name} | classes {completed_classes}/{len(class_assets)} | jobs {jobs}",
            refresh=False,
        )
        pending_results[result.index] = result
        while next_result_index in pending_results:
            ready_result = pending_results.pop(next_result_index)
            write_progress_line(
                progress,
                f"gen ready {ready_result.label_dir} | synthetic {ready_result.synthetic_count} | real {ready_result.real_count}",
            )
            next_result_index += 1

    if jobs == 1:
        for assets in class_assets:
            handle_result(
                _build_class_dataset(assets, config, texture_paths, split_profiles)
            )
    else:
        try:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = [
                    executor.submit(
                        _build_class_dataset,
                        assets,
                        config,
                        texture_paths,
                        split_profiles,
                    )
                    for assets in class_assets
                ]
                for completed_index, future in enumerate(
                    as_completed(futures), start=1
                ):
                    result = future.result()
                    progress.set_postfix_str(
                        f"classes {completed_index}/{len(class_assets)} | jobs {jobs}",
                        refresh=False,
                    )
                    handle_result(result)
        except (OSError, PermissionError):
            _reset_output_dir(config.output_dir, class_assets)
            for split in (TRAIN_SPLIT, VAL_SPLIT, R_TRAIN_SPLIT, R_VAL_SPLIT):
                for assets in class_assets:
                    ensure_dir(config.output_dir / split / assets.label_dir)
            jobs = 1
            completed_classes = 0
            next_result_index = 0
            pending_results = {}
            preview_buckets = {
                (split, profile_name): []
                for split, profile_names in split_profiles.items()
                for profile_name in profile_names
            }
            profile_histogram = {
                split: {profile_name: 0 for profile_name in profile_names}
                for split, profile_names in split_profiles.items()
            }
            progress.set_postfix_str(
                f"classes 0/{len(class_assets)} | jobs {jobs}", refresh=False
            )
            for assets in class_assets:
                handle_result(
                    _build_class_dataset(assets, config, texture_paths, split_profiles)
                )

    progress.close()
    preview_sheets = _save_preview_sheets(config.output_dir, preview_buckets)
    source_manifest = build_source_manifest(
        seed=config.seed,
        real_val_fraction=config.real_val_fraction,
    )

    r_train_total = sum(
        len(list((config.output_dir / R_TRAIN_SPLIT / assets.label_dir).glob("*.png")))
        for assets in class_assets
    )
    r_val_total = sum(
        len(list((config.output_dir / R_VAL_SPLIT / assets.label_dir).glob("*.png")))
        for assets in class_assets
    )
    metadata = {
        "version": 3,
        "jobs": jobs,
        "num_classes": len(class_assets),
        "class_names": [assets.name for assets in class_assets],
        "class_titles": [assets.title for assets in class_assets],
        "train_per_class": config.train_per_class,
        "val_per_class": config.val_per_class,
        "image_size": config.output_size,
        "color_mode": "rgb",
        "synthetic_total_images": total_synthetic,
        "real_total_images": r_train_total + r_val_total,
        "total_images": total_synthetic + r_train_total + r_val_total,
        "primary_validation_split": R_VAL_SPLIT if r_val_total > 0 else VAL_SPLIT,
        "real_val_fraction": config.real_val_fraction,
        "train_profiles": list(split_profiles[TRAIN_SPLIT]),
        "val_profiles": list(split_profiles[VAL_SPLIT]),
        "profile_histogram": profile_histogram,
        "preview_sheets": preview_sheets,
        "source": source_manifest,
        "notes": [
            "Synthetic images are composited from exemplar and real source images.",
            "Real images are copied into dataset/r_train and dataset/r_val with light RGB normalization only.",
            "Primary validation should use r_val when available.",
        ],
    }
    save_json(config.output_dir / "metadata.json", metadata)
    return metadata
