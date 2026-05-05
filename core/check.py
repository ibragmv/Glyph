from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from core.checkpoints import get_checkpoint_class_names, load_model_checkpoint
from core.constants import CLASS_NAMES, LABEL_DIRS
from core.datagen import discover_fonts
from core.runtime import configure_runtime, prepare_matplotlib
from core.utils import list_image_files


CHECK_STATUS_ORDER = {"ok": 0, "warn": 1, "fail": 2}


def _merge_status(left: str, right: str) -> str:
    return left if CHECK_STATUS_ORDER[left] >= CHECK_STATUS_ORDER[right] else right


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def inspect_dataset(data_dir: Path) -> dict[str, Any]:
    if not data_dir.exists():
        return {
            "status": "fail",
            "message": f"dataset root not found: {data_dir}",
        }

    split_stats: dict[str, Any] = {}
    status = "ok"
    missing_splits: list[str] = []
    for split in ("train", "val"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            missing_splits.append(split)
            continue

        expected_dirs = [split_dir / name for name in LABEL_DIRS.values()]
        present_dirs = [path for path in expected_dirs if path.is_dir()]
        image_count = len(list_image_files(split_dir))
        split_stats[split] = {
            "dir": str(split_dir),
            "class_dirs": len(present_dirs),
            "images": image_count,
            "missing_class_dirs": [path.name for path in expected_dirs if not path.is_dir()],
        }
        if len(present_dirs) != len(CLASS_NAMES) or image_count == 0:
            status = _merge_status(status, "warn")

    if missing_splits:
        return {
            "status": "fail",
            "message": f"missing dataset splits: {', '.join(missing_splits)}",
        }

    metadata_path = data_dir / "metadata.json"
    metadata_status = "ok"
    metadata_message = "metadata.json ready"
    metadata_payload = None
    if metadata_path.is_file():
        metadata_payload = _read_json(metadata_path)
        metadata_class_count = int(metadata_payload.get("num_classes", -1))
        if metadata_class_count != len(CLASS_NAMES):
            metadata_status = "fail"
            metadata_message = (
                f"metadata class count is {metadata_class_count}, expected {len(CLASS_NAMES)}"
            )
        class_names = [item.get("name") for item in metadata_payload.get("classes", [])]
        if class_names and class_names != CLASS_NAMES:
            metadata_status = "fail"
            metadata_message = "metadata class names do not match project classes"
    else:
        metadata_status = "warn"
        metadata_message = "metadata.json is missing"

    status = _merge_status(status, metadata_status)
    total_images = sum(item["images"] for item in split_stats.values())
    return {
        "status": status,
        "message": (
            f"train {split_stats['train']['images']} images | "
            f"val {split_stats['val']['images']} images | "
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
    if len(class_names) != len(CLASS_NAMES):
        status = _merge_status(status, "fail")
        issues.append(
            f"class count is {len(class_names)}, expected {len(CLASS_NAMES)}"
        )
    if class_names != CLASS_NAMES:
        status = _merge_status(status, "fail")
        issues.append("class names do not match project order")
    if model_classes != len(class_names):
        status = _merge_status(status, "fail")
        issues.append(
            f"model metadata says {model_classes} classes, checkpoint stores {len(class_names)}"
        )

    backbone = metadata["model"]["backbone_name"]
    message = f"{backbone} | {len(class_names)} classes"
    if issues:
        message = f"{message} | {'; '.join(issues)}"

    return {
        "status": status,
        "message": message,
        "backbone_name": backbone,
        "temperature": float(metadata.get("calibration", {}).get("temperature", 1.0)),
        "class_names": class_names,
        "metadata": metadata,
    }


def inspect_fonts(extra_font_dirs: tuple[Path, ...] = ()) -> dict[str, Any]:
    try:
        fonts = discover_fonts(explicit_fonts=(), extra_font_dirs=extra_font_dirs)
    except FileNotFoundError as exc:
        return {"status": "fail", "message": str(exc), "fonts": []}
    except Exception as exc:
        return {"status": "fail", "message": f"font discovery failed: {exc}", "fonts": []}

    return {
        "status": "ok",
        "message": f"{len(fonts)} compatible font(s) detected",
        "fonts": [str(path) for path in fonts],
    }


def inspect_runtime() -> dict[str, Any]:
    configure_runtime()
    try:
        import albumentations  # noqa: F401
        import fontTools  # noqa: F401
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "ok",
        "message": (
            f"python {sys.version.split()[0]} | "
            f"torch {torch.__version__} | "
            f"device {device}"
        ),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    }


def inspect_project(
    *,
    data_dir: Path,
    checkpoint_path: Path,
    extra_font_dirs: tuple[Path, ...] = (),
) -> dict[str, Any]:
    runtime = inspect_runtime()
    dataset = inspect_dataset(data_dir)
    checkpoint = inspect_checkpoint(checkpoint_path)
    fonts = inspect_fonts(extra_font_dirs=extra_font_dirs)

    status = "ok"
    for item in (runtime, dataset, checkpoint, fonts):
        status = _merge_status(status, item["status"])

    dataset_metadata = dataset.get("metadata") or {}
    dataset_class_count = int(dataset_metadata.get("num_classes", len(CLASS_NAMES)))
    checkpoint_class_count = len(checkpoint.get("class_names", []))
    consistency_status = "ok"
    consistency_notes: list[str] = []
    if dataset_class_count != len(CLASS_NAMES):
        consistency_status = _merge_status(consistency_status, "fail")
        consistency_notes.append(
            f"dataset metadata says {dataset_class_count} classes"
        )
    if checkpoint["status"] == "fail":
        consistency_status = _merge_status(consistency_status, "warn")
        consistency_notes.append("checkpoint class count could not be verified")
    elif checkpoint_class_count not in (0, len(CLASS_NAMES)):
        consistency_status = _merge_status(consistency_status, "fail")
        consistency_notes.append(
            f"checkpoint says {checkpoint_class_count} classes"
        )
    if (
        checkpoint["status"] != "fail"
        and checkpoint_class_count != 0
        and dataset_class_count == len(CLASS_NAMES)
        and checkpoint_class_count != dataset_class_count
    ):
        consistency_status = _merge_status(consistency_status, "fail")
        consistency_notes.append("dataset/checkpoint class counts disagree")
    if not consistency_notes:
        consistency_notes.append(f"class count matches project constant ({len(CLASS_NAMES)})")

    status = _merge_status(status, consistency_status)
    checks = [
        {"name": "runtime", **runtime},
        {"name": "dataset", **dataset},
        {"name": "checkpoint", **checkpoint},
        {"name": "fonts", **fonts},
        {
            "name": "classes",
            "status": consistency_status,
            "message": " | ".join(consistency_notes),
        },
    ]

    return {
        "status": status,
        "checks": checks,
        "runtime": runtime,
        "dataset": dataset,
        "checkpoint": checkpoint,
        "fonts": fonts,
    }
