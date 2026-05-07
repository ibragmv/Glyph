from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from core.constants import ALPHABET, LABEL_DIRS
from core.utils import list_image_files


def read_dataset_metadata(root: Path) -> dict | None:
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        return None
    with metadata_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def choose_validation_split(root: Path) -> str:
    metadata = read_dataset_metadata(root) or {}
    preferred = metadata.get("primary_validation_split")
    if isinstance(preferred, str) and (root / preferred).is_dir():
        return preferred
    if (root / "realval").is_dir():
        return "realval"
    return "val"


def choose_training_splits(root: Path) -> tuple[str, ...]:
    splits = []
    if (root / "train").is_dir():
        splits.append("train")
    if (root / "realtrain").is_dir():
        splits.append("realtrain")
    if not splits:
        raise FileNotFoundError(f"No training splits were found in {root}")
    return tuple(splits)


class ImperialAramaicDataset(Dataset):
    def __init__(
        self, root: Path, split: str, transform=None, return_paths: bool = False
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_paths = return_paths
        self.samples: list[tuple[Path, int]] = []

        split_root = self.root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        for item in ALPHABET:
            label_idx = item["index"]
            label_dir = split_root / LABEL_DIRS[label_idx]
            if not label_dir.exists():
                continue
            for image_path in sorted(label_dir.glob("*.png")):
                self.samples.append((image_path, label_idx))

        if not self.samples:
            raise FileNotFoundError(f"No PNG samples were found in {split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("L"))

        if self.transform is not None:
            image_tensor = self.transform(image=array)["image"]
        else:
            image_tensor = array

        if self.return_paths:
            return image_tensor, label, str(image_path)
        return image_tensor, label


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples = list_image_files(self.root)

        if not self.root.exists():
            raise FileNotFoundError(f"Image directory not found: {self.root}")
        if not self.samples:
            raise FileNotFoundError(f"No supported image files were found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("L").resize((64, 64)))

        if self.transform is not None:
            image_tensor = self.transform(image=array)["image"]
        else:
            image_tensor = array

        return image_tensor, str(image_path)
