from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from core.models import build_model


def _validated_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint {name} must be a mapping.")
    return value


def _validate_model_state_dict(value: Any) -> Mapping[str, torch.Tensor]:
    state_dict = _validated_mapping(value, name="model_state_dict")
    if not state_dict:
        raise ValueError("Checkpoint model_state_dict is empty.")

    for key, tensor in state_dict.items():
        if not isinstance(key, str):
            raise ValueError("Checkpoint model_state_dict keys must be strings.")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                "Checkpoint model_state_dict values must be torch.Tensor instances."
            )
    return state_dict


def _validate_checkpoint_payload(payload: Any) -> dict[str, Any]:
    checkpoint = dict(_validated_mapping(payload, name="payload"))
    checkpoint["model_state_dict"] = _validate_model_state_dict(
        checkpoint.get("model_state_dict")
    )
    checkpoint["metadata"] = dict(
        _validated_mapping(checkpoint.get("metadata"), name="metadata")
    )

    model_metadata = dict(
        _validated_mapping(checkpoint["metadata"].get("model"), name="metadata.model")
    )
    dataset_metadata = dict(
        _validated_mapping(
            checkpoint["metadata"].get("dataset"),
            name="metadata.dataset",
        )
    )
    checkpoint["metadata"]["model"] = model_metadata
    checkpoint["metadata"]["dataset"] = dataset_metadata
    return checkpoint


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


def get_checkpoint_normalization(
    checkpoint: dict[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    metadata = get_checkpoint_metadata(checkpoint)
    normalization = metadata["dataset"]["normalization"]
    mean = tuple(float(value) for value in normalization["mean"])
    std = tuple(float(value) for value in normalization["std"])
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Checkpoint normalization must contain three RGB channels.")
    return mean, std


def get_checkpoint_image_size(checkpoint: dict[str, Any]) -> int:
    metadata = get_checkpoint_metadata(checkpoint)
    dataset_metadata = metadata.get("dataset", {})
    image_size = int(dataset_metadata.get("image_size", 64))
    if image_size < 1:
        raise ValueError(f"Checkpoint image_size must be >= 1, got {image_size}")
    return image_size


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
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(f"Checkpoint could not be read safely: {exc}") from exc

    checkpoint = _validate_checkpoint_payload(checkpoint)
    checkpoint["metadata"] = get_checkpoint_metadata(checkpoint)
    model = build_model_from_checkpoint(checkpoint, device)
    return model, checkpoint
