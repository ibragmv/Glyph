from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from core.utils import ensure_dir, format_float


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
    prev_val_loss: float | None,
    prev_val_acc: float | None,
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
        "val_loss_delta": None
        if val_loss_delta is None
        else format_float(val_loss_delta),
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


def plot_history(history: dict[str, Any], output_path: Path) -> None:
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


def plot_generalization_diagnostics(
    history: dict[str, Any], output_path: Path
) -> None:
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

    axes[1].bar(
        epochs, val_acc_delta, label="val_acc delta", color="#4cc9f0", alpha=0.8
    )
    axes[1].plot(
        epochs, val_loss_delta, label="val_loss delta", color="#f72585", linewidth=2
    )
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.35)
    axes[1].set_title("Validation Quality Change")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_epoch_metrics_csv(history: dict[str, Any], output_path: Path) -> None:
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


def build_overfitting_signals(history: dict[str, Any]) -> dict[str, Any]:
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
