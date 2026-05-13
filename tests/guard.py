from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from core.checkpoints import load_model_checkpoint
from core.constants import CLASS_NAMES
from core.models import build_model


@dataclass
class UnsafePayload:
    label: str


def _checkpoint_metadata() -> dict:
    return {
        "model": {
            "backbone_name": "mobilenet_v3_small",
            "num_classes": len(CLASS_NAMES),
            "dropout": 0.0,
            "small_image_stem": False,
        },
        "dataset": {
            "class_names": CLASS_NAMES,
            "normalization": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "image_size": 64,
        },
        "calibration": {"temperature": 1.0},
    }


def _checkpoint_state_dict() -> dict[str, torch.Tensor]:
    model = build_model(
        backbone_name="mobilenet_v3_small",
        num_classes=len(CLASS_NAMES),
        dropout=0.0,
        pretrained=False,
        small_image_stem=False,
    )
    return dict(model.state_dict())


def test_load_model_checkpoint_rejects_non_mapping_payload(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "invalid.pt"
    torch.save(["not", "a", "checkpoint"], checkpoint_path)

    with pytest.raises(ValueError, match="payload must be a mapping"):
        load_model_checkpoint(checkpoint_path, torch.device("cpu"))


def test_load_model_checkpoint_rejects_unsafe_objects(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "unsafe.pt"
    torch.save(
        {
            "model_state_dict": _checkpoint_state_dict(),
            "metadata": _checkpoint_metadata(),
            "payload": UnsafePayload(label="unsafe"),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="could not be read safely"):
        load_model_checkpoint(checkpoint_path, torch.device("cpu"))
