from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from core.constants import (
    ALPHABET,
    ALPHABET_ROOT,
    EXEMPLAR_ROOT,
    REAL_ROOT,
    TEXTURE_ROOT,
)
from core.utils import list_image_files


@dataclass(frozen=True)
class ClassAssets:
    index: int
    name: str
    title: str
    label_dir: str
    alphabet_path: Path
    exemplar_paths: tuple[Path, ...]
    real_paths: tuple[Path, ...]


def _sort_paths(paths: list[Path]) -> tuple[Path, ...]:
    return tuple(sorted(paths, key=lambda path: path.name.lower()))


def get_class_assets() -> list[ClassAssets]:
    assets: list[ClassAssets] = []
    for item in ALPHABET:
        label_dir = item["label_dir"]
        alphabet_path = ALPHABET_ROOT / f"{label_dir}.png"
        exemplar_paths = _sort_paths(list_image_files(EXEMPLAR_ROOT / label_dir))
        real_paths = _sort_paths(list_image_files(REAL_ROOT / label_dir))
        assets.append(
            ClassAssets(
                index=item["index"],
                name=item["name"],
                title=item["title"],
                label_dir=label_dir,
                alphabet_path=alphabet_path,
                exemplar_paths=exemplar_paths,
                real_paths=real_paths,
            )
        )
    return assets


def get_texture_paths() -> tuple[Path, ...]:
    return _sort_paths(list_image_files(TEXTURE_ROOT))


def split_real_paths(
    paths: tuple[Path, ...],
    *,
    seed: int,
    val_fraction: float,
) -> dict[str, tuple[Path, ...]]:
    if not paths:
        return {"realtrain": (), "realval": ()}
    if val_fraction <= 0.0:
        return {"realtrain": paths, "realval": ()}

    ordered = list(paths)
    # Deterministic shuffle without depending on Python's randomized hash seed.
    keyed = sorted(
        ordered,
        key=lambda path: hashlib.sha1(
            f"{seed}:{path.name.lower()}:{path.as_posix()}".encode("utf-8")
        ).hexdigest(),
    )
    if val_fraction >= 1.0:
        val_count = len(keyed)
    else:
        val_count = max(1, int(round(len(keyed) * val_fraction)))
    if 0.0 < val_fraction < 1.0 and len(keyed) > 1:
        val_count = min(val_count, len(keyed) - 1)
    val_paths = tuple(keyed[:val_count])
    train_paths = tuple(keyed[val_count:])
    return {"realtrain": train_paths, "realval": val_paths}


def build_source_manifest(*, seed: int, real_val_fraction: float) -> dict:
    classes = []
    for assets in get_class_assets():
        split = split_real_paths(
            assets.real_paths,
            seed=seed,
            val_fraction=real_val_fraction,
        )
        classes.append(
            {
                "index": assets.index,
                "name": assets.name,
                "title": assets.title,
                "label_dir": assets.label_dir,
                "alphabet_path": str(assets.alphabet_path),
                "exemplars": [str(path) for path in assets.exemplar_paths],
                "real_paths": [str(path) for path in assets.real_paths],
                "realtrain_paths": [str(path) for path in split["realtrain"]],
                "realval_paths": [str(path) for path in split["realval"]],
            }
        )
    return {
        "source_root": "source",
        "textures": [str(path) for path in get_texture_paths()],
        "classes": classes,
    }
