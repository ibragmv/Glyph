from __future__ import annotations

import hashlib
from typing import Iterable


RUNTIME_DEPENDENCIES: tuple[str, ...] = (
    "numpy>=1.26",
    "Pillow>=10.3",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "scikit-learn>=1.5",
    "tqdm>=4.66",
    "opencv-python-headless>=4.10",
    "albumentations>=1.4",
    "torch>=2.3",
    "torchvision>=0.18",
)

DEV_DEPENDENCIES: tuple[str, ...] = (
    "pytest>=8.3",
    "ruff>=0.6",
)


def bootstrap_dependencies() -> tuple[str, ...]:
    return RUNTIME_DEPENDENCIES + DEV_DEPENDENCIES


def dependency_manifest_hash(lines: Iterable[str] | None = None) -> str:
    manifest_lines = tuple(bootstrap_dependencies() if lines is None else lines)
    payload = "\n".join(manifest_lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
