from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from core.constants import IMPERIAL_ARAMAIC_SYMBOLS, LABEL_DIRS
from core.utils import list_image_files


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

        for symbol in IMPERIAL_ARAMAIC_SYMBOLS:
            label_idx = symbol["index"]
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
