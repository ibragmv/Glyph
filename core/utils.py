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


def compute_image_mean_std(image_paths: Iterable[Path]) -> tuple[float, float]:
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    for path in image_paths:
        image = Image.open(path).convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
        pixel_sum += float(array.sum())
        pixel_sq_sum += float(np.square(array).sum())
        pixel_count += array.size

    if pixel_count == 0:
        raise ValueError("No images found to compute mean/std.")

    mean = pixel_sum / pixel_count
    variance = max((pixel_sq_sum / pixel_count) - (mean * mean), 1e-12)
    std = float(np.sqrt(variance))
    return float(mean), std


def format_float(value: float, digits: int = 6) -> float:
    return float(f"{value:.{digits}f}")


def format_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"
