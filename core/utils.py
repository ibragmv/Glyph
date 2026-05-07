from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def list_image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for extension in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        for path in root.rglob(extension)
    )


def compute_image_mean_std(
    image_paths: Iterable[Path],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in image_paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        flat = array.reshape(-1, 3)
        pixel_sum += flat.sum(axis=0)
        pixel_sq_sum += np.square(flat).sum(axis=0)
        pixel_count += flat.shape[0]

    if pixel_count == 0:
        raise ValueError("No images found to compute mean/std.")

    mean = pixel_sum / pixel_count
    variance = np.maximum((pixel_sq_sum / pixel_count) - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


def format_float(value: float, digits: int = 6) -> float:
    return float(f"{value:.{digits}f}")


def format_channels(
    values: tuple[float, float, float], digits: int = 6
) -> tuple[float, float, float]:
    return tuple(format_float(value, digits=digits) for value in values)


def format_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"
