from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from core.checkpoints import (
    get_checkpoint_class_names,
    get_checkpoint_temperature,
    load_model_checkpoint,
)
from core.constants import (
    CLASS_NAMES,
    LABEL_DIRS,
    R_TRAIN_SPLIT,
    R_VAL_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from core.datasets import choose_validation_split
from core.runtime import configure_runtime, prepare_matplotlib, resolve_runtime_device
from core.source import get_class_assets, get_texture_paths
from core.utils import list_image_files


CHECK_STATUS_ORDER = {"ok": 0, "warn": 1, "fail": 2}


def _merge_status(left: str, right: str) -> str:
    return left if CHECK_STATUS_ORDER[left] >= CHECK_STATUS_ORDER[right] else right


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _display_split_name(value: str) -> str:
    if value == f"real{TRAIN_SPLIT}":
        return R_TRAIN_SPLIT
    if value == f"real{VAL_SPLIT}":
        return R_VAL_SPLIT
    return value


def inspect_source() -> dict[str, Any]:
    class_assets = get_class_assets()
    texture_paths = get_texture_paths()
    if not class_assets:
        return {"status": "fail", "message": "source assets are missing"}

    status = "ok"
    notes: list[str] = []
    missing_alphabet = [
        assets.title for assets in class_assets if not assets.alphabet_path.is_file()
    ]
    missing_real = [assets.title for assets in class_assets if not assets.real_paths]
    missing_exemplars = [
        assets.title for assets in class_assets if not assets.exemplar_paths
    ]
    if missing_alphabet:
        status = _merge_status(status, "warn")
        notes.append(f"missing reference image for {', '.join(missing_alphabet[:6])}")
    if missing_real:
        status = _merge_status(status, "warn")
        notes.append(f"no real crops for {', '.join(missing_real[:6])}")
    if missing_exemplars:
        status = _merge_status(status, "warn")
        notes.append(f"no exemplars for {', '.join(missing_exemplars[:6])}")
    if not texture_paths:
        status = _merge_status(status, "warn")
        notes.append("no texture images found")

    total_real = sum(len(assets.real_paths) for assets in class_assets)
    total_exemplars = sum(len(assets.exemplar_paths) for assets in class_assets)
    message = (
        f"{len(class_assets)} classes | "
        f"{total_real} real crops | "
        f"{total_exemplars} exemplars | "
        f"{len(texture_paths)} textures"
    )
    if notes:
        message = f"{message} | {'; '.join(notes)}"
    return {
        "status": status,
        "message": message,
    }


def inspect_dataset(data_dir: Path) -> dict[str, Any]:
    if not data_dir.exists():
        return {
            "status": "fail",
            "message": f"dataset root not found: {data_dir}",
        }

    split_stats: dict[str, Any] = {}
    status = "ok"
    required_splits = (TRAIN_SPLIT, VAL_SPLIT, R_TRAIN_SPLIT, R_VAL_SPLIT)
    missing_splits = [
        split for split in required_splits if not (data_dir / split).is_dir()
    ]
    if missing_splits:
        status = _merge_status(status, "warn")

    for split in required_splits:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        expected_dirs = [split_dir / name for name in LABEL_DIRS.values()]
        present_dirs = [path for path in expected_dirs if path.is_dir()]
        image_count = len(list_image_files(split_dir))
        split_stats[split] = {
            "dir": str(split_dir),
            "class_dirs": len(present_dirs),
            "images": image_count,
            "missing_class_dirs": [
                path.name for path in expected_dirs if not path.is_dir()
            ],
        }
        if len(present_dirs) != len(CLASS_NAMES):
            status = _merge_status(status, "warn")

    metadata_path = data_dir / "metadata.json"
    metadata_payload = None
    metadata_status = "ok"
    metadata_message = "metadata.json ready"
    if metadata_path.is_file():
        metadata_payload = _read_json(metadata_path)
        if metadata_payload.get("class_names") != CLASS_NAMES:
            metadata_status = "fail"
            metadata_message = "metadata class names do not match project classes"
        elif metadata_payload.get(
            "primary_validation_split"
        ) != choose_validation_split(data_dir):
            metadata_status = "warn"
            metadata_message = "metadata validation split differs from detected split"
    else:
        metadata_status = "warn"
        metadata_message = "metadata.json is missing"

    status = _merge_status(status, metadata_status)
    total_images = sum(item["images"] for item in split_stats.values())
    return {
        "status": status,
        "message": (
            f"train {split_stats.get(TRAIN_SPLIT, {}).get('images', 0)} | "
            f"val {split_stats.get(VAL_SPLIT, {}).get('images', 0)} | "
            f"r_train {split_stats.get(R_TRAIN_SPLIT, {}).get('images', 0)} | "
            f"r_val {split_stats.get(R_VAL_SPLIT, {}).get('images', 0)} | "
            f"total {total_images}"
        ),
        "metadata_message": metadata_message,
        "splits": split_stats,
        "metadata": metadata_payload,
    }


def inspect_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        return {
            "status": "fail",
            "message": f"checkpoint not found: {checkpoint_path}",
        }

    device = torch.device("cpu")
    try:
        model, checkpoint = load_model_checkpoint(checkpoint_path, device)
    except Exception as exc:
        return {
            "status": "fail",
            "message": f"checkpoint could not be loaded: {exc}",
        }

    del model
    metadata = checkpoint["metadata"]
    class_names = get_checkpoint_class_names(checkpoint)
    model_classes = int(metadata["model"].get("num_classes", len(class_names)))
    status = "ok"
    issues: list[str] = []
    if class_names != CLASS_NAMES:
        status = _merge_status(status, "fail")
        issues.append("class names do not match current canonical alphabet")
    if model_classes != len(CLASS_NAMES):
        status = _merge_status(status, "fail")
        issues.append(f"model metadata says {model_classes} classes")

    dataset_split = _display_split_name(
        metadata.get("dataset", {}).get("val_split", "unknown")
    )
    backbone = metadata["model"]["backbone_name"]
    message = (
        f"{backbone} | {len(class_names)} classes | "
        f"temperature {get_checkpoint_temperature(checkpoint):.4f} | "
        f"val_split {dataset_split}"
    )
    if issues:
        message = f"{message} | {'; '.join(issues)}"

    return {
        "status": status,
        "message": message,
        "backbone_name": backbone,
        "temperature": get_checkpoint_temperature(checkpoint),
        "class_names": class_names,
        "metadata": metadata,
    }


def inspect_runtime() -> dict[str, Any]:
    configure_runtime()
    try:
        import albumentations  # noqa: F401
        import matplotlib  # noqa: F401
        import PIL  # noqa: F401
        import sklearn  # noqa: F401
        import torchvision  # noqa: F401

        prepare_matplotlib()
    except Exception as exc:
        return {
            "status": "fail",
            "message": f"runtime import failed: {exc}",
        }

    runtime_device = resolve_runtime_device()
    return {
        "status": "ok",
        "message": (
            f"python {sys.version.split()[0]} | "
            f"torch {torch.__version__} | "
            f"device {runtime_device.label} | {runtime_device.reason}"
        ),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": runtime_device.label,
            "device_type": runtime_device.type,
            "device_reason": runtime_device.reason,
            "cuda_built": runtime_device.cuda_built,
            "cuda_available": runtime_device.cuda_available,
            "cuda_version": runtime_device.cuda_version,
            "cuda_device_count": runtime_device.cuda_device_count,
            "cuda_name": runtime_device.cuda_name,
            "mps_built": runtime_device.mps_built,
            "mps_available": runtime_device.mps_available,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    }


def inspect_project(
    *,
    data_dir: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    runtime = inspect_runtime()
    source = inspect_source()
    dataset = inspect_dataset(data_dir)
    checkpoint = inspect_checkpoint(checkpoint_path)

    status = "ok"
    for item in (runtime, source, dataset, checkpoint):
        status = _merge_status(status, item["status"])

    dataset_metadata = dataset.get("metadata") or {}
    dataset_class_names = dataset_metadata.get("class_names", [])
    checkpoint_class_names = checkpoint.get("class_names", [])
    consistency_status = "ok"
    consistency_notes: list[str] = []
    if dataset_class_names and dataset_class_names != CLASS_NAMES:
        consistency_status = _merge_status(consistency_status, "fail")
        consistency_notes.append("dataset classes disagree with project classes")
    if checkpoint_class_names and checkpoint_class_names != CLASS_NAMES:
        consistency_status = _merge_status(consistency_status, "fail")
        consistency_notes.append("checkpoint classes disagree with project classes")
    if not consistency_notes:
        consistency_notes.append(
            f"class count matches canonical alphabet ({len(CLASS_NAMES)})"
        )

    status = _merge_status(status, consistency_status)
    checks = [
        {"name": "runtime", **runtime},
        {"name": "source", **source},
        {"name": "dataset", **dataset},
        {"name": "checkpoint", **checkpoint},
        {
            "name": "classes",
            "status": consistency_status,
            "message": " | ".join(consistency_notes),
        },
    ]
    return {
        "status": status,
        "checks": checks,
    }
