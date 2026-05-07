from __future__ import annotations

import csv
from copy import deepcopy
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader

from core.augmentations import build_eval_transforms, build_train_transforms
from core.console import (
    create_progress,
    display_path,
    print_log,
    print_summary,
    write_progress_line,
)
from core.constants import CLASS_NAMES
from core.datasets import (
    ImperialAramaicDataset,
    choose_training_splits,
    choose_validation_split,
)
from core.models import DEFAULT_BACKBONE, build_model
from core.runtime import (
    RuntimeDevice,
    configure_runtime,
    optimize_model_for_device,
    prepare_image_batch,
    resolve_num_workers,
    resolve_runtime_device,
)
from core.utils import (
    compute_image_mean_std,
    ensure_dir,
    format_float,
    format_percent,
    list_image_files,
    save_json,
    seed_everything,
)


def current_learning_rate(optimizer: Adam) -> float:
    return float(optimizer.param_groups[0]["lr"])


def format_signed_delta(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def format_signed_percentage_points(value: float, digits: int = 2) -> str:
    return f"{value * 100:+.{digits}f}pp"


def format_optional_signed_delta(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return format_signed_delta(value, digits=digits)


def format_optional_signed_percentage_points(
    value: Optional[float], digits: int = 2
) -> str:
    if value is None:
        return "n/a"
    return format_signed_percentage_points(value, digits=digits)


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int | None,
    mean: float,
    std: float,
    *,
    runtime_device: RuntimeDevice,
) -> tuple[DataLoader, DataLoader, tuple[str, ...], str]:
    train_splits = choose_training_splits(data_dir)
    val_split = choose_validation_split(data_dir)

    train_datasets = [
        ImperialAramaicDataset(
            root=data_dir,
            split=split,
            transform=build_train_transforms(mean, std),
        )
        for split in train_splits
    ]
    train_dataset = (
        train_datasets[0]
        if len(train_datasets) == 1
        else ConcatDataset(train_datasets)
    )
    val_dataset = ImperialAramaicDataset(
        root=data_dir,
        split=val_split,
        transform=build_eval_transforms(mean, std),
    )

    resolved_num_workers = resolve_num_workers(num_workers)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": resolved_num_workers,
        "pin_memory": runtime_device.pin_memory,
        "drop_last": False,
    }
    if resolved_num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, train_splits, val_split


class FocalLoss(nn.Module):
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: float | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        num_classes = logits.shape[1]

        target_dist = F.one_hot(targets, num_classes=num_classes).to(logits.dtype)
        if self.label_smoothing > 0.0:
            smoothing = self.label_smoothing / num_classes
            target_dist = target_dist * (1.0 - self.label_smoothing) + smoothing

        ce_per_sample = -(target_dist * log_probs).sum(dim=1)
        pt = (target_dist * probs).sum(dim=1).clamp_min(1e-8)
        focal_weight = (1.0 - pt).pow(self.gamma)
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha
        return (focal_weight * ce_per_sample).mean()


def build_criterion(
    *,
    loss_name: str,
    label_smoothing: float,
    focal_gamma: float,
    focal_alpha: float | None,
) -> nn.Module:
    normalized_loss_name = loss_name.strip().lower()
    if normalized_loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if normalized_loss_name == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


def apply_temperature(
    logits: torch.Tensor, temperature: float | torch.Tensor
) -> torch.Tensor:
    if isinstance(temperature, torch.Tensor):
        safe_temperature = temperature.clamp_min(1e-6)
    else:
        safe_temperature = max(float(temperature), 1e-6)
    return logits / safe_temperature


def collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: RuntimeDevice,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            images = prepare_image_batch(images, device)
            logits = model(images)
            logits_chunks.append(logits.detach())
            label_chunks.append(labels.to(device.device, non_blocking=device.pin_memory))
    return torch.cat(logits_chunks, dim=0), torch.cat(label_chunks, dim=0)


def fit_temperature_scaling(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: RuntimeDevice,
    max_iter: int,
) -> dict[str, Any]:
    logits, labels = collect_logits_and_labels(model, loader, device)
    before_nll = float(F.cross_entropy(logits, labels).item())
    before_acc = float((logits.argmax(dim=1) == labels).float().mean().item())

    temperature = torch.ones(1, device=device.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [temperature],
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(apply_temperature(logits, temperature), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    final_temperature = max(float(temperature.detach().item()), 1e-3)
    calibrated_logits = apply_temperature(logits, final_temperature)
    after_nll = float(F.cross_entropy(calibrated_logits, labels).item())
    after_acc = float(
        (calibrated_logits.argmax(dim=1) == labels).float().mean().item()
    )
    return {
        "enabled": True,
        "method": "temperature_scaling",
        "temperature": format_float(final_temperature),
        "optimizer": "LBFGS",
        "max_iter": max_iter,
        "metrics": {
            "nll_before": format_float(before_nll),
            "nll_after": format_float(after_nll),
            "accuracy_before": format_float(before_acc),
            "accuracy_after": format_float(after_acc),
        },
    }


def create_disabled_calibration() -> dict[str, Any]:
    return {
        "enabled": False,
        "method": "temperature_scaling",
        "temperature": 1.0,
        "optimizer": None,
        "max_iter": None,
        "metrics": {},
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: RuntimeDevice,
    optimizer: Optional[Adam] = None,
    scheduler: Optional[OneCycleLR] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    overall_progress=None,
    epoch_progress=None,
    total_epochs: int = 1,
    epoch_index: int = 1,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    running_correct = 0
    sample_count = 0

    for images, labels in loader:
        images = prepare_image_batch(images, device)
        labels = labels.to(device.device, non_blocking=device.pin_memory)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if is_train:
            assert optimizer is not None
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        running_correct += int((predictions == labels).sum().item())
        sample_count += batch_size

        if overall_progress is not None:
            stage = "train" if is_train else "val"
            overall_progress.update(1)
            overall_progress.set_postfix_str(
                (
                    f"epoch {epoch_index}/{total_epochs} | "
                    f"stage {stage} | "
                    f"loss {running_loss / max(sample_count, 1):.4f} | "
                    f"acc {format_percent(running_correct / max(sample_count, 1))}"
                ),
                refresh=False,
            )
        if epoch_progress is not None:
            stage = "train" if is_train else "val"
            epoch_progress.update(1)
            epoch_progress.set_postfix_str(
                (
                    f"stage {stage} | "
                    f"loss {running_loss / max(sample_count, 1):.4f} | "
                    f"acc {format_percent(running_correct / max(sample_count, 1))}"
                ),
                refresh=False,
            )

    return running_loss / sample_count, running_correct / sample_count


def create_history() -> dict[str, Any]:
    return {
        "epochs": [],
        "best_epoch": None,
        "best_val_acc": None,
    }


def build_epoch_record(
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
    learning_rate: float,
    best_epoch: int,
    best_val_acc: float,
    prev_val_loss: Optional[float],
    prev_val_acc: Optional[float],
    is_best_epoch: bool,
) -> dict[str, Any]:
    val_loss_delta = None if prev_val_loss is None else val_loss - prev_val_loss
    val_acc_delta = None if prev_val_acc is None else val_acc - prev_val_acc
    loss_gap = val_loss - train_loss
    acc_gap = train_acc - val_acc
    return {
        "epoch": epoch,
        "train_loss": format_float(train_loss),
        "val_loss": format_float(val_loss),
        "train_acc": format_float(train_acc),
        "val_acc": format_float(val_acc),
        "learning_rate": format_float(learning_rate),
        "train_val_loss_gap": format_float(loss_gap),
        "train_val_acc_gap": format_float(acc_gap),
        "val_loss_delta": None if val_loss_delta is None else format_float(val_loss_delta),
        "val_acc_delta": None if val_acc_delta is None else format_float(val_acc_delta),
        "best_epoch_so_far": best_epoch,
        "best_val_acc_so_far": format_float(best_val_acc),
        "is_best_epoch": is_best_epoch,
    }


def append_epoch_history(history: dict[str, Any], epoch_record: dict[str, Any]) -> None:
    history["epochs"].append(epoch_record)
    history["best_epoch"] = epoch_record["best_epoch_so_far"]
    history["best_val_acc"] = epoch_record["best_val_acc_so_far"]


def history_series(history: dict[str, Any], key: str) -> list[Any]:
    return [epoch[key] for epoch in history["epochs"]]


def plot_history(history: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    train_loss = history_series(history, "train_loss")
    val_loss = history_series(history, "val_loss")
    train_acc = history_series(history, "train_acc")
    val_acc = history_series(history, "val_acc")
    learning_rate = history_series(history, "learning_rate")
    train_val_loss_gap = history_series(history, "train_val_loss_gap")
    train_val_acc_gap = history_series(history, "train_val_acc_gap")
    epochs = range(1, len(train_loss) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    axes[2].plot(epochs, train_val_loss_gap, label="val_loss - train_loss")
    axes[2].plot(epochs, train_val_acc_gap, label="train_acc - val_acc")
    axes[2].axhline(0.0, color="black", linewidth=1, alpha=0.35)
    axes[2].set_title("Generalization Gap")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.2)

    axes[3].plot(epochs, learning_rate, color="#6f5eff")
    axes[3].set_title("Learning Rate")
    axes[3].set_xlabel("Epoch")
    axes[3].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_generalization_diagnostics(history: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    epochs = list(range(1, len(history["epochs"]) + 1))
    val_acc_delta = [
        item["val_acc_delta"] if item["val_acc_delta"] is not None else 0.0
        for item in history["epochs"]
    ]
    val_loss_delta = [
        item["val_loss_delta"] if item["val_loss_delta"] is not None else 0.0
        for item in history["epochs"]
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].bar(
        epochs,
        history_series(history, "train_val_acc_gap"),
        color="#ff8a5b",
        alpha=0.85,
    )
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.35)
    axes[0].set_title("Train/Val Accuracy Gap by Epoch")
    axes[0].set_ylabel("Gap")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(epochs, val_acc_delta, label="val_acc delta", color="#4cc9f0", alpha=0.8)
    axes[1].plot(epochs, val_loss_delta, label="val_loss delta", color="#f72585", linewidth=2)
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.35)
    axes[1].set_title("Validation Quality Change")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_epoch_metrics_csv(history: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "learning_rate",
        "train_val_loss_gap",
        "train_val_acc_gap",
        "val_loss_delta",
        "val_acc_delta",
        "best_epoch_so_far",
        "best_val_acc_so_far",
        "is_best_epoch",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history["epochs"])


def count_trailing_epochs_without_improvement(values: list[float]) -> int:
    if not values:
        return 0

    best_value = max(values)
    best_epoch_index = len(values) - 1 - values[::-1].index(best_value)
    return len(values) - best_epoch_index - 1


def longest_streak(values: list[float], predicate) -> int:
    longest = 0
    current = 0
    for value in values:
        if predicate(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def build_overfitting_signals(history: dict) -> dict[str, Any]:
    val_acc_series = history_series(history, "val_acc")
    val_loss_series = history_series(history, "val_loss")
    acc_gap_series = history_series(history, "train_val_acc_gap")
    loss_gap_series = history_series(history, "train_val_loss_gap")
    val_acc_deltas = [
        item["val_acc_delta"]
        for item in history["epochs"]
        if item["val_acc_delta"] is not None
    ]
    val_loss_deltas = [
        item["val_loss_delta"]
        for item in history["epochs"]
        if item["val_loss_delta"] is not None
    ]
    best_epoch = history["best_epoch"] or 0
    best_val_acc = history["best_val_acc"] or 0.0
    final_val_acc = val_acc_series[-1] if val_acc_series else 0.0
    final_val_loss = val_loss_series[-1] if val_loss_series else 0.0
    best_val_loss = (
        history["epochs"][best_epoch - 1]["val_loss"] if best_epoch > 0 else None
    )

    flags = []
    if acc_gap_series and acc_gap_series[-1] > 0.05:
        flags.append("final_train_val_acc_gap_gt_5pp")
    if val_acc_series and best_val_acc - final_val_acc > 0.03:
        flags.append("final_val_acc_drop_gt_3pp_from_best")
    if best_val_loss is not None and final_val_loss - best_val_loss > 0.05:
        flags.append("final_val_loss_above_best_gt_0_05")
    if longest_streak(val_acc_deltas, lambda value: value < 0) >= 3:
        flags.append("val_acc_declined_3_epochs_in_a_row")
    if longest_streak(val_loss_deltas, lambda value: value > 0) >= 3:
        flags.append("val_loss_increased_3_epochs_in_a_row")

    return {
        "best_epoch": best_epoch,
        "best_val_acc": format_float(best_val_acc),
        "best_val_loss": None if best_val_loss is None else format_float(best_val_loss),
        "final_val_acc": format_float(final_val_acc),
        "final_val_loss": format_float(final_val_loss),
        "final_val_acc_drop_from_best": format_float(best_val_acc - final_val_acc),
        "final_val_loss_increase_from_best": (
            None
            if best_val_loss is None
            else format_float(final_val_loss - best_val_loss)
        ),
        "max_train_val_acc_gap": (
            None if not acc_gap_series else format_float(max(acc_gap_series))
        ),
        "max_train_val_loss_gap": (
            None if not loss_gap_series else format_float(max(loss_gap_series))
        ),
        "epochs_since_best_val_acc": count_trailing_epochs_without_improvement(
            val_acc_series
        ),
        "longest_val_acc_decline_streak": longest_streak(
            val_acc_deltas, lambda value: value < 0
        ),
        "longest_val_loss_increase_streak": longest_streak(
            val_loss_deltas, lambda value: value > 0
        ),
        "flags": flags,
    }


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
    mean: float,
    std: float,
    history: dict,
    best_path: Path,
    last_path: Path,
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


def train_model(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_workers: int | None = None,
    seed: int = 42,
    pretrained: bool = False,
    small_image_stem: bool = True,
    backbone_name: str = DEFAULT_BACKBONE,
    dropout: float = 0.3,
    loss_name: str = "ce",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    temperature_scaling: bool = False,
    temperature_max_iter: int = 50,
) -> dict:
    configure_runtime()
    seed_everything(seed)
    ensure_dir(output_dir)

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if lr <= 0:
        raise ValueError("lr must be > 0")
    train_splits = choose_training_splits(data_dir)
    val_split = choose_validation_split(data_dir)
    training_image_paths: list[Path] = []
    for split in train_splits:
        training_image_paths.extend(list_image_files(data_dir / split))
    mean, std = compute_image_mean_std(training_image_paths)
    mean = format_float(mean)
    std = format_float(std)

    resolved_num_workers = resolve_num_workers(num_workers)
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if focal_gamma < 0.0:
        raise ValueError("focal_gamma must be >= 0")
    if temperature_max_iter < 1:
        raise ValueError("temperature_max_iter must be >= 1")

    runtime_device = resolve_runtime_device()
    train_loader, val_loader, train_splits, val_split = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=resolved_num_workers,
        mean=mean,
        std=std,
        runtime_device=runtime_device,
    )

    model = build_model(
        num_classes=len(CLASS_NAMES),
        dropout=dropout,
        pretrained=pretrained,
        small_image_stem=small_image_stem,
        backbone_name=backbone_name,
    ).to(runtime_device.device)
    model = optimize_model_for_device(model, runtime_device)

    resolved_small_image_stem = getattr(model, "small_image_stem", small_image_stem)
    model_config = {
        "backbone_name": getattr(model, "backbone_name", backbone_name),
        "family": getattr(model, "backbone_family", "unknown"),
        "dropout": dropout,
        "pretrained": pretrained,
        "small_image_stem": resolved_small_image_stem,
        "num_classes": len(CLASS_NAMES),
    }
    loss_config = {
        "name": "cross_entropy" if loss_name == "ce" else "focal",
        "label_smoothing": label_smoothing,
        "focal_gamma": focal_gamma if loss_name == "focal" else None,
        "focal_alpha": focal_alpha if loss_name == "focal" else None,
    }
    checkpoint_calibration = {
        "best": create_disabled_calibration(),
        "last": create_disabled_calibration(),
    }

    criterion = build_criterion(
        loss_name=loss_name,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler(
        runtime_device.type,
        enabled=runtime_device.amp_enabled,
    )

    history = create_history()
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state_dict = deepcopy(model.state_dict())
    best_optimizer_state_dict = deepcopy(optimizer.state_dict())
    best_scheduler_state_dict = deepcopy(scheduler.state_dict())
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    total_steps = epochs * (len(train_loader) + len(val_loader))
    run_started_at = datetime.now(timezone.utc)
    run_id = run_started_at.strftime("train-%Y%m%dT%H%M%SZ")

    print_summary(
        "Training Run",
        [
            ("device", runtime_device.label),
            ("device_reason", runtime_device.reason),
            ("workers", resolved_num_workers),
            ("train", len(train_loader.dataset)),
            ("val", len(val_loader.dataset)),
            ("train_splits", ",".join(train_splits)),
            ("val_split", val_split),
            ("backbone", model_config["backbone_name"]),
            ("pretrained", "on" if pretrained else "off"),
            ("loss", loss_config["name"]),
            ("label_smoothing", f"{label_smoothing:.3f}"),
            ("temp_scaling", "on" if temperature_scaling else "off"),
            ("output", display_path(output_dir)),
            ("mean", f"{mean:.4f}"),
            ("std", f"{std:.4f}"),
        ],
    )
    overall_progress = create_progress(
        command="train",
        scope="fit",
        color="green",
        leave=False,
        total=total_steps,
        position=0,
    )
    overall_progress.set_postfix_str(f"epoch 0/{epochs}")
    overall_progress.refresh()

    for epoch in range(1, epochs + 1):
        epoch_progress = create_progress(
            command="train",
            scope=f"epoch {epoch:02d}",
            color="green",
            leave=False,
            total=len(train_loader) + len(val_loader),
            position=1,
        )
        epoch_progress.set_postfix_str("stage train")
        epoch_progress.refresh()
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=runtime_device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            overall_progress=overall_progress,
            epoch_progress=epoch_progress,
            total_epochs=epochs,
            epoch_index=epoch,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=runtime_device,
            overall_progress=overall_progress,
            epoch_progress=epoch_progress,
            total_epochs=epochs,
            epoch_index=epoch,
        )

        lr_now = current_learning_rate(optimizer)
        prev_val_loss = history["epochs"][-1]["val_loss"] if history["epochs"] else None
        prev_val_acc = history["epochs"][-1]["val_acc"] if history["epochs"] else None
        is_best_epoch = val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss <= best_val_loss
        )
        if is_best_epoch:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state_dict = deepcopy(model.state_dict())
            best_optimizer_state_dict = deepcopy(optimizer.state_dict())
            best_scheduler_state_dict = deepcopy(scheduler.state_dict())

        epoch_record = build_epoch_record(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            learning_rate=lr_now,
            best_epoch=best_epoch,
            best_val_acc=best_val_acc,
            prev_val_loss=prev_val_loss,
            prev_val_acc=prev_val_acc,
            is_best_epoch=is_best_epoch,
        )
        append_epoch_history(history, epoch_record)

        print_log(
            "train",
            (
                f"epoch {epoch:02d}/{epochs} | "
                f"train_loss {train_loss:.4f} | train_acc {format_percent(train_acc)} | "
                f"val_loss {val_loss:.4f} ({format_optional_signed_delta(epoch_record['val_loss_delta'])}) | "
                f"val_acc {format_percent(val_acc)} ({format_optional_signed_percentage_points(epoch_record['val_acc_delta'])}) | "
                f"gap {format_signed_percentage_points(epoch_record['train_val_acc_gap'])} | "
                f"lr {epoch_record['learning_rate']:.6f} | "
                f"best ep {best_epoch:02d} {format_percent(best_val_acc)}"
            ),
        )
        write_progress_line(
            overall_progress,
            (
                f"train ready epoch {epoch:02d}/{epochs} | "
                f"train_acc {format_percent(train_acc)} | "
                f"val_acc {format_percent(val_acc)} | "
                f"gap {format_signed_percentage_points(epoch_record['train_val_acc_gap'])} | "
                f"best ep {best_epoch:02d}"
            ),
        )
        epoch_progress.set_postfix_str(
            (
                f"done | train_acc {format_percent(train_acc)} | "
                f"val_acc {format_percent(val_acc)} | "
                f"gap {format_signed_percentage_points(epoch_record['train_val_acc_gap'])}"
            )
        )
        epoch_progress.close()

        checkpoint_metadata = build_training_metadata(
            run_id=run_id,
            started_at=run_started_at.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=(datetime.now(timezone.utc) - run_started_at).total_seconds(),
            data_dir=data_dir,
            output_dir=output_dir,
            train_loader=train_loader,
            val_loader=val_loader,
            train_splits=train_splits,
            val_split=val_split,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            num_workers=resolved_num_workers,
            seed=seed,
            model_config=model_config,
            loss_config=loss_config,
            calibration_config=checkpoint_calibration["last"],
            device=runtime_device,
            mean=mean,
            std=std,
            history=history,
            best_path=best_path,
            last_path=last_path,
        )
        save_checkpoint(
            path=last_path,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            epoch=epoch,
            metadata=checkpoint_metadata,
        )

        if is_best_epoch:
            save_checkpoint(
                path=best_path,
                model_state_dict=best_model_state_dict,
                optimizer_state_dict=best_optimizer_state_dict,
                scheduler_state_dict=best_scheduler_state_dict,
                epoch=epoch,
                metadata=checkpoint_metadata,
            )

    overall_progress.set_postfix_str(f"epoch {epochs}/{epochs}")
    overall_progress.close()
    plot_history(history, output_dir / "training_curves.png")
    plot_generalization_diagnostics(history, output_dir / "training_diagnostics.png")
    save_epoch_metrics_csv(history, output_dir / "epoch_metrics.csv")

    if temperature_scaling:
        best_calibrated_model = build_model(
            num_classes=len(CLASS_NAMES),
            dropout=dropout,
            pretrained=False,
            small_image_stem=resolved_small_image_stem,
            backbone_name=model_config["backbone_name"],
        ).to(runtime_device.device)
        best_calibrated_model = optimize_model_for_device(
            best_calibrated_model,
            runtime_device,
        )
        best_calibrated_model.load_state_dict(best_model_state_dict)
        checkpoint_calibration["best"] = fit_temperature_scaling(
            model=best_calibrated_model,
            loader=val_loader,
            device=runtime_device,
            max_iter=temperature_max_iter,
        )
        checkpoint_calibration["last"] = fit_temperature_scaling(
            model=model,
            loader=val_loader,
            device=runtime_device,
            max_iter=temperature_max_iter,
        )
        print_log(
            "train",
            (
                f"temperature scaling best | T {checkpoint_calibration['best']['temperature']:.4f} | "
                f"nll {checkpoint_calibration['best']['metrics']['nll_before']:.4f} -> "
                f"{checkpoint_calibration['best']['metrics']['nll_after']:.4f}"
            ),
            tone="success",
        )
        print_log(
            "train",
            (
                f"temperature scaling last | T {checkpoint_calibration['last']['temperature']:.4f} | "
                f"nll {checkpoint_calibration['last']['metrics']['nll_before']:.4f} -> "
                f"{checkpoint_calibration['last']['metrics']['nll_after']:.4f}"
            ),
            tone="success",
        )

    run_completed_at = datetime.now(timezone.utc)
    best_metadata = build_training_metadata(
        run_id=run_id,
        started_at=run_started_at.isoformat(),
        completed_at=run_completed_at.isoformat(),
        duration_seconds=(run_completed_at - run_started_at).total_seconds(),
        data_dir=data_dir,
        output_dir=output_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        train_splits=train_splits,
        val_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=resolved_num_workers,
        seed=seed,
        model_config=model_config,
        loss_config=loss_config,
        calibration_config=checkpoint_calibration["best"],
        device=runtime_device,
        mean=mean,
        std=std,
        history=history,
        best_path=best_path,
        last_path=last_path,
    )
    last_metadata = build_training_metadata(
        run_id=run_id,
        started_at=run_started_at.isoformat(),
        completed_at=run_completed_at.isoformat(),
        duration_seconds=(run_completed_at - run_started_at).total_seconds(),
        data_dir=data_dir,
        output_dir=output_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        train_splits=train_splits,
        val_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=resolved_num_workers,
        seed=seed,
        model_config=model_config,
        loss_config=loss_config,
        calibration_config=checkpoint_calibration["last"],
        device=runtime_device,
        mean=mean,
        std=std,
        history=history,
        best_path=best_path,
        last_path=last_path,
    )
    save_json(
        output_dir / "history.json",
        history,
    )
    save_json(
        output_dir / "training_metadata.json",
        {
            "best_checkpoint": best_metadata,
            "last_checkpoint": last_metadata,
        },
    )

    save_checkpoint(
        path=last_path,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        epoch=epochs,
        metadata=last_metadata,
    )
    save_checkpoint(
        path=best_path,
        model_state_dict=best_model_state_dict,
        optimizer_state_dict=best_optimizer_state_dict,
        scheduler_state_dict=best_scheduler_state_dict,
        epoch=best_epoch,
        metadata=best_metadata,
    )

    return {
        "device": runtime_device.label,
        "device_reason": runtime_device.reason,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "train_splits": list(train_splits),
        "val_split": val_split,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_path": str(output_dir / "history.json"),
        "curves_path": str(output_dir / "training_curves.png"),
        "diagnostics_path": str(output_dir / "training_diagnostics.png"),
        "epoch_metrics_path": str(output_dir / "epoch_metrics.csv"),
        "metadata_path": str(output_dir / "training_metadata.json"),
    }
