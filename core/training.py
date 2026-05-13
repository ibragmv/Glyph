from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
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
    read_dataset_metadata,
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
from core.training_calibration import (
    build_criterion,
    create_disabled_calibration,
    fit_temperature_scaling,
)
from core.training_history import (
    append_epoch_history,
    build_epoch_record,
    create_history,
    plot_generalization_diagnostics,
    plot_history,
    save_epoch_metrics_csv,
)
from core.training_metadata import build_training_metadata, save_checkpoint
from core.utils import (
    compute_image_mean_std,
    ensure_dir,
    format_channels,
    format_percent,
    list_image_files,
    save_json,
    seed_everything,
)


def current_learning_rate(optimizer: Adam) -> float:
    return float(optimizer.param_groups[0]["lr"])


def build_optimizer(model: nn.Module, lr: float, device: RuntimeDevice) -> Adam:
    if device.type == "cuda":
        try:
            return Adam(model.parameters(), lr=lr, fused=True)
        except (TypeError, RuntimeError):
            pass
    return Adam(model.parameters(), lr=lr)


def format_signed_percentage_points(value: float, digits: int = 2) -> str:
    return f"{value * 100:+.{digits}f}pp"


def format_optional_signed_percentage_points(
    value: Optional[float], digits: int = 2
) -> str:
    if value is None:
        return "n/a"
    return format_signed_percentage_points(value, digits=digits)


def format_rgb_triplet(values: tuple[float, float, float], digits: int = 4) -> str:
    return ", ".join(f"{value:.{digits}f}" for value in values)


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int | None,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
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
        train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
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
        loader_kwargs["prefetch_factor"] = 4

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
            stage = "tr" if is_train else "va"
            overall_progress.update(1)
            overall_progress.set_postfix_str(
                (
                    f"ep {epoch_index}/{total_epochs} | "
                    f"{stage} | "
                    f"loss {running_loss / max(sample_count, 1):.4f} | "
                    f"acc {format_percent(running_correct / max(sample_count, 1))}"
                ),
                refresh=False,
            )
        if epoch_progress is not None:
            stage = "tr" if is_train else "va"
            epoch_progress.update(1)
            epoch_progress.set_postfix_str(
                (
                    f"{stage} | "
                    f"loss {running_loss / max(sample_count, 1):.4f} | "
                    f"acc {format_percent(running_correct / max(sample_count, 1))}"
                ),
                refresh=False,
            )

    return running_loss / sample_count, running_correct / sample_count


def resolve_dataset_image_size(
    data_dir: Path,
    training_image_paths: list[Path],
) -> int:
    metadata = read_dataset_metadata(data_dir) or {}
    image_size = metadata.get("image_size")
    if isinstance(image_size, int) and image_size > 0:
        return image_size

    if not training_image_paths:
        raise ValueError("No training images were found to resolve image_size.")

    with Image.open(training_image_paths[0]) as image:
        width, height = image.size
    if width != height:
        raise ValueError(
            f"Training images must be square for checkpoint metadata, got {width}x{height}"
        )
    return int(width)


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
    mean = format_channels(mean)
    std = format_channels(std)
    image_size = resolve_dataset_image_size(data_dir, training_image_paths)

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
    optimizer = build_optimizer(model, lr, runtime_device)
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
            ("mean", format_rgb_triplet(mean)),
            ("std", format_rgb_triplet(std)),
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
    overall_progress.set_postfix_str(f"ep 0/{epochs}")
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
        epoch_progress.set_postfix_str("tr")
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
                f"ep {epoch:02d}/{epochs} | "
                f"tr {format_percent(train_acc)} {train_loss:.4f} | "
                f"va {format_percent(val_acc)} {val_loss:.4f} | "
                f"dva {format_optional_signed_percentage_points(epoch_record['val_acc_delta'])} | "
                f"gap {format_signed_percentage_points(epoch_record['train_val_acc_gap'])} | "
                f"lr {epoch_record['learning_rate']:.6f} | "
                f"best {best_epoch:02d} {format_percent(best_val_acc)}"
            ),
        )
        write_progress_line(
            overall_progress,
            (
                f"train ready ep {epoch:02d}/{epochs} | "
                f"tr {format_percent(train_acc)} | "
                f"va {format_percent(val_acc)} | "
                f"gap {format_signed_percentage_points(epoch_record['train_val_acc_gap'])} | "
                f"best {best_epoch:02d}"
            ),
        )
        epoch_progress.set_postfix_str(
            (
                f"done | tr {format_percent(train_acc)} | "
                f"va {format_percent(val_acc)} | "
                f"gap {format_signed_percentage_points(epoch_record['train_val_acc_gap'])}"
            )
        )
        epoch_progress.close()

        checkpoint_metadata = build_training_metadata(
            run_id=run_id,
            started_at=run_started_at.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=(
                datetime.now(timezone.utc) - run_started_at
            ).total_seconds(),
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
            image_size=image_size,
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

    overall_progress.set_postfix_str(f"ep {epochs}/{epochs}")
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
        image_size=image_size,
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
        image_size=image_size,
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
