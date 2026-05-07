from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from core.checkpoints import (
    get_checkpoint_class_names,
    get_checkpoint_image_size,
    load_model_checkpoint,
)
from core.constants import CLASS_NAMES
from core.inference import predict_folder, predict_image
from core.source import split_real_paths


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
    assert metadata["image_size"] == 64
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
    assert get_checkpoint_image_size(checkpoint) == 64


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


def test_dataset_regeneration_replaces_stale_samples(
    smoke_workspace: Path,
    cli_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    dataset_dir = smoke_workspace / "regen-dataset"

    first = cli_runner(
        "gen",
        "--out",
        dataset_dir,
        "--train",
        "2",
        "--val",
        "1",
        "--canvas",
        "96",
        "--size",
        "64",
        "--tmix",
        "clean",
        "--vmix",
        "clean",
        "--seed",
        "11",
    )
    assert first.returncode == 0, first.stdout + first.stderr

    second = cli_runner(
        "gen",
        "--out",
        dataset_dir,
        "--train",
        "1",
        "--val",
        "1",
        "--canvas",
        "96",
        "--size",
        "64",
        "--tmix",
        "clean",
        "--vmix",
        "clean",
        "--seed",
        "11",
    )
    assert second.returncode == 0, second.stdout + second.stderr

    assert len(list((dataset_dir / "train").rglob("*.png"))) == len(CLASS_NAMES)
    assert len(list((dataset_dir / "val").rglob("*.png"))) == len(CLASS_NAMES)


def test_split_real_paths_allows_zero_validation_fraction() -> None:
    paths = tuple(Path(f"sample_{index}.png") for index in range(3))

    split = split_real_paths(paths, seed=7, val_fraction=0.0)

    assert split["realtrain"] == paths
    assert split["realval"] == ()


def test_predict_image_uses_checkpoint_image_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.full((19, 27), 180, dtype=np.uint8)).save(image_path)

    seen_shapes: list[tuple[int, ...]] = []

    class FakeModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            seen_shapes.append(tuple(tensor.shape))
            return torch.zeros((tensor.shape[0], len(CLASS_NAMES)), dtype=torch.float32)

    fake_checkpoint = {
        "metadata": {
            "model": {"backbone_name": "fake"},
            "dataset": {
                "class_names": CLASS_NAMES,
                "normalization": {"mean": 0.5, "std": 0.5},
                "image_size": 32,
            },
            "calibration": {"temperature": 1.0},
        }
    }

    monkeypatch.setattr(
        "core.inference.load_model_checkpoint",
        lambda checkpoint_path, device: (FakeModel(), fake_checkpoint),
    )
    monkeypatch.setattr(
        "core.inference.optimize_model_for_device",
        lambda model, runtime_device: model,
    )

    result = predict_image(checkpoint_path=tmp_path / "fake.pt", image_path=image_path)

    assert result["predictions"][0]["label"] in CLASS_NAMES
    assert seen_shapes == [(1, 1, 32, 32)]


def test_predict_folder_uses_checkpoint_image_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.full((21, 17), 200, dtype=np.uint8)).save(image_dir / "one.png")

    seen_shapes: list[tuple[int, ...]] = []

    class FakeModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            seen_shapes.append(tuple(tensor.shape))
            return torch.zeros((tensor.shape[0], len(CLASS_NAMES)), dtype=torch.float32)

    fake_checkpoint = {
        "metadata": {
            "model": {"backbone_name": "fake"},
            "dataset": {
                "class_names": CLASS_NAMES,
                "normalization": {"mean": 0.5, "std": 0.5},
                "image_size": 32,
            },
            "calibration": {"temperature": 1.0},
        }
    }

    monkeypatch.setattr(
        "core.inference.load_model_checkpoint",
        lambda checkpoint_path, device: (FakeModel(), fake_checkpoint),
    )
    monkeypatch.setattr(
        "core.inference.optimize_model_for_device",
        lambda model, runtime_device: model,
    )

    summary = predict_folder(
        checkpoint_path=tmp_path / "fake.pt",
        image_dir=image_dir,
        output_dir=tmp_path / "out",
        top_k=2,
        batch_size=1,
        num_workers=0,
    )

    assert summary["num_images"] == 1
    assert seen_shapes == [(1, 1, 32, 32)]
