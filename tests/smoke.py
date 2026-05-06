from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from core.checkpoints import get_checkpoint_class_names, load_model_checkpoint
from core.constants import CLASS_NAMES
from core.inference import predict_image


pytestmark = pytest.mark.smoke


def test_cli_help_smoke(
    cli_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    result = cli_runner("--help")

    assert result.returncode == 0, result.stderr
    assert "glyph --help" in result.stdout
    assert "gen" in result.stdout
    assert "pred" in result.stdout
    assert "check" in result.stdout
    assert "qa" in result.stdout


def test_dataset_generation_smoke(generated_dataset_dir: Path) -> None:
    metadata = json.loads((generated_dataset_dir / "metadata.json").read_text())

    assert metadata["num_classes"] == len(CLASS_NAMES)
    assert metadata["train_per_class"] == 1
    assert metadata["val_per_class"] == 1
    assert metadata["synthetic_total_images"] == len(CLASS_NAMES) * 2
    assert metadata["real_total_images"] > 0
    assert metadata["total_images"] == metadata["synthetic_total_images"] + metadata["real_total_images"]
    assert metadata["primary_validation_split"] == "realval"
    assert len(list((generated_dataset_dir / "train").rglob("*.png"))) == len(CLASS_NAMES)
    assert len(list((generated_dataset_dir / "val").rglob("*.png"))) == len(CLASS_NAMES)
    assert len(list((generated_dataset_dir / "realval").rglob("*.png"))) > 0
    assert metadata["preview_sheets"]


def test_checkpoint_load_smoke(smoke_checkpoint_path: Path) -> None:
    model, checkpoint = load_model_checkpoint(smoke_checkpoint_path, torch.device("cpu"))

    assert model.training is False
    assert checkpoint["metadata"]["model"]["backbone_name"] == "mobilenet_v3_small"
    assert get_checkpoint_class_names(checkpoint) == CLASS_NAMES


def test_single_image_inference_smoke(
    smoke_checkpoint_path: Path,
    sample_image_path: Path,
) -> None:
    result = predict_image(
        checkpoint_path=smoke_checkpoint_path,
        image_path=sample_image_path,
        top_k=3,
    )

    assert result["checkpoint_path"] == str(smoke_checkpoint_path)
    assert result["image_path"] == str(sample_image_path)
    assert result["backbone_name"] == "mobilenet_v3_small"
    assert len(result["predictions"]) == 3
    assert result["predictions"][0]["label"] in CLASS_NAMES


def test_cli_pred_smoke(
    cli_runner: Callable[..., subprocess.CompletedProcess[str]],
    smoke_checkpoint_path: Path,
    sample_image_path: Path,
) -> None:
    result = cli_runner(
        "pred",
        "--pt",
        smoke_checkpoint_path,
        "--img",
        sample_image_path,
        "--top",
        "2",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Prediction Summary" in result.stdout
    assert "top_1" in result.stdout
    assert "pred prediction complete" in result.stdout
