from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from core.constants import CLASS_NAMES
from core.models import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = PROJECT_ROOT / "fonts" / "NotoSansImperialAramaic-Regular.ttf"


def _build_smoke_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(workspace / "pycache")
    env["MPLCONFIGDIR"] = str(workspace / "matplotlib")
    return env


@pytest.fixture(scope="session")
def smoke_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("smoke")


@pytest.fixture(scope="session")
def smoke_env(smoke_workspace: Path) -> dict[str, str]:
    return _build_smoke_env(smoke_workspace)


@pytest.fixture(scope="session")
def cli_runner(
    smoke_env: dict[str, str],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run_cli(*args: object) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, "-m", "core", *(str(arg) for arg in args)]
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=smoke_env,
            text=True,
            capture_output=True,
            check=False,
        )

    return run_cli


@pytest.fixture(scope="session")
def generated_dataset_dir(
    smoke_workspace: Path,
    cli_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> Path:
    dataset_dir = smoke_workspace / "dataset"
    result = cli_runner(
        "gen",
        "--out",
        dataset_dir,
        "--font",
        FONT_PATH,
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
        "--holdout",
        "0",
        "--seed",
        "7",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return dataset_dir


@pytest.fixture(scope="session")
def sample_image_path(generated_dataset_dir: Path) -> Path:
    return sorted((generated_dataset_dir / "val").rglob("*.png"))[0]


@pytest.fixture(scope="session")
def smoke_checkpoint_path(smoke_workspace: Path) -> Path:
    checkpoint_path = smoke_workspace / "artifacts" / "smoke_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)
    model = build_model(
        backbone_name="mobilenet_v3_small",
        num_classes=len(CLASS_NAMES),
        dropout=0.0,
        pretrained=False,
        small_image_stem=False,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "model": {
                    "backbone_name": "mobilenet_v3_small",
                    "num_classes": len(CLASS_NAMES),
                    "dropout": 0.0,
                    "small_image_stem": False,
                },
                "dataset": {
                    "class_names": CLASS_NAMES,
                    "normalization": {"mean": 0.5, "std": 0.5},
                },
                "calibration": {"temperature": 1.0},
            },
        },
        checkpoint_path,
    )
    return checkpoint_path
