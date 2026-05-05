from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from core.models import build_model


def get_checkpoint_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = checkpoint.get("metadata")
    if metadata is None:
        raise ValueError(
            "Checkpoint does not contain experiment metadata. Re-train or re-export "
            "the checkpoint with the current format."
        )
    return metadata


def get_model_config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = get_checkpoint_metadata(checkpoint)
    model_metadata = metadata["model"]
    return {
        "num_classes": model_metadata.get(
            "num_classes", len(metadata["dataset"]["class_names"])
        ),
        "dropout": model_metadata.get("dropout", 0.3),
        "pretrained": False,
        "small_image_stem": model_metadata.get("small_image_stem", True),
        "backbone_name": model_metadata["backbone_name"],
    }


def get_checkpoint_temperature(checkpoint: dict[str, Any]) -> float:
    metadata = get_checkpoint_metadata(checkpoint)
    calibration = metadata.get("calibration", {})
    return float(calibration.get("temperature", 1.0))


def get_checkpoint_class_names(checkpoint: dict[str, Any]) -> list[str]:
    metadata = get_checkpoint_metadata(checkpoint)
    return list(metadata["dataset"]["class_names"])


def get_checkpoint_normalization(checkpoint: dict[str, Any]) -> tuple[float, float]:
    metadata = get_checkpoint_metadata(checkpoint)
    normalization = metadata["dataset"]["normalization"]
    return float(normalization["mean"]), float(normalization["std"])


def build_model_from_checkpoint(
    checkpoint: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    model = build_model(**get_model_config_from_checkpoint(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_model_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    checkpoint["metadata"] = get_checkpoint_metadata(checkpoint)
    model = build_model_from_checkpoint(checkpoint, device)
    return model, checkpoint
