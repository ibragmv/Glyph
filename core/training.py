from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from core.augmentations import build_eval_transforms, build_train_transforms
from core.console import (
    create_progress,
    display_path,
    print_log,
    print_summary,
    write_progress_line,
)
from core.constants import CLASS_NAMES
from core.datasets import ImperialAramaicDataset
from core.models import build_model
from core.utils import (
    compute_image_mean_std,
    ensure_dir,
    format_float,
    format_percent,
    list_image_files,
    save_json,
    seed_everything,
)


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    mean: float,
    std: float,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = ImperialAramaicDataset(
        root=data_dir,
        split="train",
        transform=build_train_transforms(mean, std),
    )
    val_dataset = ImperialAramaicDataset(
        root=data_dir,
        split="val",
        transform=build_eval_transforms(mean, std),
    )

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
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
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
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
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
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


def plot_history(history: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Adam,
    scheduler: OneCycleLR,
    epoch: int,
    best_val_acc: float,
    mean: float,
    std: float,
    history: dict,
    small_image_stem: bool,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": CLASS_NAMES,
            "mean": mean,
            "std": std,
            "dropout": 0.3,
            "small_image_stem": small_image_stem,
            "history": history,
        },
        path,
    )


def train_model(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_workers: int = 0,
    seed: int = 42,
    pretrained: bool = False,
    small_image_stem: bool = True,
) -> dict:
    seed_everything(seed)
    ensure_dir(output_dir)

    mean, std = compute_image_mean_std(list_image_files(data_dir / "train"))
    mean = format_float(mean)
    std = format_float(std)

    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        mean=mean,
        std=std,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=len(CLASS_NAMES),
        dropout=0.3,
        pretrained=pretrained,
        small_image_stem=small_image_stem,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    total_steps = epochs * (len(train_loader) + len(val_loader))

    print_summary(
        "Training Run",
        [
            ("device", device.type),
            ("train", len(train_loader.dataset)),
            ("val", len(val_loader.dataset)),
            ("pretrained", "on" if pretrained else "off"),
            ("output", display_path(output_dir)),
            ("mean", f"{mean:.4f}"),
            ("std", f"{std:.4f}"),
        ],
    )
    overall_progress = create_progress(
        command="train",
        scope="fit",
        color="green",
        leave=True,
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
            device=device,
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
            device=device,
            overall_progress=overall_progress,
            epoch_progress=epoch_progress,
            total_epochs=epochs,
            epoch_index=epoch,
        )

        history["train_loss"].append(format_float(train_loss))
        history["val_loss"].append(format_float(val_loss))
        history["train_acc"].append(format_float(train_acc))
        history["val_acc"].append(format_float(val_acc))

        next_best_val_acc = max(best_val_acc, val_acc)
        print_log(
            "train",
            (
                f"epoch {epoch:02d}/{epochs} | "
                f"train_loss {train_loss:.4f} | train_acc {format_percent(train_acc)} | "
                f"val_loss {val_loss:.4f} | val_acc {format_percent(val_acc)} | "
                f"best {format_percent(next_best_val_acc)}"
            ),
        )
        write_progress_line(
            overall_progress,
            (
                f"train ready epoch {epoch:02d}/{epochs} | "
                f"train_acc {format_percent(train_acc)} | "
                f"val_acc {format_percent(val_acc)} | "
                f"best {format_percent(next_best_val_acc)}"
            ),
        )
        epoch_progress.set_postfix_str(
            (
                f"done | train_acc {format_percent(train_acc)} | "
                f"val_acc {format_percent(val_acc)}"
            )
        )
        epoch_progress.close()

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_acc=next_best_val_acc,
            mean=mean,
            std=std,
            history=history,
            small_image_stem=small_image_stem,
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_acc=best_val_acc,
                mean=mean,
                std=std,
                history=history,
                small_image_stem=small_image_stem,
            )

    overall_progress.set_postfix_str(f"epoch {epochs}/{epochs}")
    overall_progress.close()
    plot_history(history, output_dir / "training_curves.png")
    save_json(
        output_dir / "history.json",
        {
            "history": history,
            "best_val_acc": format_float(best_val_acc),
            "mean": mean,
            "std": std,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "pretrained": pretrained,
            "small_image_stem": small_image_stem,
        },
    )
    return {
        "device": device.type,
        "best_val_acc": best_val_acc,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_path": str(output_dir / "history.json"),
        "curves_path": str(output_dir / "training_curves.png"),
    }
