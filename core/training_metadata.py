from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from core.constants import CLASS_NAMES
from core.runtime import RuntimeDevice
from core.training_history import build_overfitting_signals
from core.utils import ensure_dir, format_float


def build_training_metadata(
    *,
    run_id: str,
    started_at: str,
    completed_at: str,
    duration_seconds: float,
    data_dir: Path,
    output_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_splits: tuple[str, ...],
    val_split: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    seed: int,
    model_config: dict[str, Any],
    loss_config: dict[str, Any],
    calibration_config: dict[str, Any],
    device: RuntimeDevice,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    history: dict[str, Any],
    best_path: Path,
    last_path: Path,
    image_size: int,
) -> dict[str, Any]:
    best_epoch = history["best_epoch"]
    best_val_acc = history["best_val_acc"]
    best_record = (
        history["epochs"][best_epoch - 1]
        if best_epoch is not None and best_epoch > 0
        else None
    )
    overfitting_signals = build_overfitting_signals(history)

    return {
        "run": {
            "run_id": run_id,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "duration_seconds": format_float(duration_seconds, digits=3),
        },
        "model": model_config,
        "loss": loss_config,
        "calibration": calibration_config,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "scheduler": "OneCycleLR",
            "num_workers": num_workers,
            "seed": seed,
        },
        "dataset": {
            "root": str(data_dir),
            "train_splits": list(train_splits),
            "val_split": val_split,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "class_names": CLASS_NAMES,
            "normalization": {
                "mean": mean,
                "std": std,
            },
            "image_size": image_size,
        },
        "environment": {
            "device": device.label,
            "device_type": device.type,
            "device_reason": device.reason,
            "cuda_built": device.cuda_built,
            "cuda_available": device.cuda_available,
            "cuda_version": device.cuda_version,
            "cuda_device_count": device.cuda_device_count,
            "cuda_name": device.cuda_name,
            "mps_built": device.mps_built,
            "mps_available": device.mps_available,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "amp_enabled": device.amp_enabled,
        },
        "summary": {
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_loss": None if best_record is None else best_record["val_loss"],
            "final_train_acc": (
                history["epochs"][-1]["train_acc"] if history["epochs"] else None
            ),
            "final_val_acc": (
                history["epochs"][-1]["val_acc"] if history["epochs"] else None
            ),
            "final_train_val_acc_gap": (
                history["epochs"][-1]["train_val_acc_gap"]
                if history["epochs"]
                else None
            ),
            "final_train_val_loss_gap": (
                history["epochs"][-1]["train_val_loss_gap"]
                if history["epochs"]
                else None
            ),
            "overfitting_signals": overfitting_signals,
        },
        "artifacts": {
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "history_json": str(output_dir / "history.json"),
            "training_curves": str(output_dir / "training_curves.png"),
            "training_diagnostics": str(output_dir / "training_diagnostics.png"),
            "epoch_metrics_csv": str(output_dir / "epoch_metrics.csv"),
            "training_metadata_json": str(output_dir / "training_metadata.json"),
        },
        "history": history,
    }


def save_checkpoint(
    path: Path,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    scheduler_state_dict: dict[str, Any],
    epoch: int,
    metadata: dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
            "metadata": metadata,
        },
        path,
    )
