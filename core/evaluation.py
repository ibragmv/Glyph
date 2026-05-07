from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from core.augmentations import build_eval_transforms
from core.checkpoints import (
    get_checkpoint_class_names,
    get_checkpoint_normalization,
    get_checkpoint_temperature,
    load_model_checkpoint,
)
from core.console import create_progress, display_path, print_log, print_summary
from core.datasets import ImperialAramaicDataset, choose_validation_split
from core.gradcam import GradCAM, overlay_heatmap
from core.runtime import (
    configure_runtime,
    optimize_model_for_device,
    prepare_image_batch,
    resolve_num_workers,
    resolve_runtime_device,
)
from core.utils import ensure_dir, format_percent, save_json


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    all_probs = []
    all_preds = []
    all_labels = []
    all_paths: list[str] = []

    with torch.no_grad():
        iterator = create_progress(
            loader,
            command="val",
            scope="predict",
            color="cyan",
            leave=False,
            total=len(loader),
        )
        for images, labels, paths in iterator:
            images = prepare_image_batch(images, device)
            logits = model(images) / temperature
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(paths)

    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        all_paths,
    )


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], output_path: Path
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_random_predictions(
    dataset: ImperialAramaicDataset,
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    class_names: list[str],
    mean: float,
    std: float,
    seed: int = 42,
    num_samples: int = 12,
) -> list[int]:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(dataset), size=min(num_samples, len(dataset)), replace=False
    )
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for ax, dataset_idx in zip(axes, indices):
        image_tensor, _, _ = dataset[dataset_idx]
        image = image_tensor.squeeze(0).numpy()
        image = np.clip((image * std) + mean, 0.0, 1.0)
        predicted_idx = int(preds[dataset_idx])
        true_idx = int(labels[dataset_idx])
        confidence = float(probs[dataset_idx, predicted_idx])
        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"P:{class_names[predicted_idx]}\nT:{class_names[true_idx]}\n{confidence:.2%}",
            fontsize=9,
        )
        ax.set_axis_off()

    for ax in axes[len(indices) :]:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return list(indices)


def plot_gradcam_examples(
    model: torch.nn.Module,
    dataset: ImperialAramaicDataset,
    device,
    output_path: Path,
    indices: list[int],
    class_names: list[str],
    mean: float,
    std: float,
) -> None:
    import matplotlib.pyplot as plt

    target_layer = getattr(model, "gradcam_layer", None)
    if target_layer is None:
        raise RuntimeError("Selected backbone does not expose a Grad-CAM target layer.")

    gradcam = GradCAM(model=model, target_layer=target_layer)
    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 3 * len(indices)))
    if len(indices) == 1:
        axes = np.array([axes])

    for row, dataset_idx in enumerate(indices):
        image_tensor, label, _ = dataset[dataset_idx]
        input_tensor = prepare_image_batch(image_tensor.unsqueeze(0), device)
        heatmap = gradcam.generate(input_tensor)
        grayscale = image_tensor.squeeze(0).numpy()
        grayscale = np.clip((grayscale * std) + mean, 0.0, 1.0)
        overlay = overlay_heatmap(grayscale, heatmap)

        with torch.no_grad():
            prediction = model(input_tensor).argmax(dim=1).item()

        axes[row, 0].imshow(grayscale, cmap="gray")
        axes[row, 0].set_title(f"Input: {class_names[label]}")
        axes[row, 0].set_axis_off()

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"Grad-CAM: {class_names[prediction]}")
        axes[row, 1].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    gradcam.close()


def evaluate_model(
    data_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int = 128,
    num_workers: int | None = None,
    seed: int = 42,
) -> dict:
    configure_runtime()
    ensure_dir(output_dir)
    runtime_device = resolve_runtime_device()
    resolved_num_workers = resolve_num_workers(num_workers)
    model, checkpoint = load_model_checkpoint(checkpoint_path, runtime_device.device)
    model = optimize_model_for_device(model, runtime_device)
    class_names = get_checkpoint_class_names(checkpoint)
    mean, std = get_checkpoint_normalization(checkpoint)
    temperature = get_checkpoint_temperature(checkpoint)
    val_split = choose_validation_split(data_dir)
    dataset = ImperialAramaicDataset(
        root=data_dir,
        split=val_split,
        transform=build_eval_transforms(mean, std),
        return_paths=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=resolved_num_workers,
        pin_memory=runtime_device.pin_memory,
    )

    print_summary(
        "Evaluation Run",
        [
            ("device", runtime_device.label),
            ("workers", resolved_num_workers),
            ("samples", len(dataset)),
            ("split", val_split),
            ("checkpoint", display_path(checkpoint_path)),
            ("output", display_path(output_dir)),
        ],
    )
    probs, preds, labels, paths = collect_predictions(
        model, loader, runtime_device, temperature
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
    selected_indices = plot_random_predictions(
        dataset=dataset,
        probs=probs,
        preds=preds,
        labels=labels,
        output_path=output_dir / "random_predictions.png",
        class_names=class_names,
        mean=mean,
        std=std,
        seed=seed,
    )
    plot_gradcam_examples(
        model=model,
        dataset=dataset,
        device=runtime_device,
        output_path=output_dir / "gradcam_examples.png",
        indices=selected_indices[: min(6, len(selected_indices))],
        class_names=class_names,
        mean=mean,
        std=std,
    )

    save_json(
        output_dir / "classification_report.json",
        {
            "classification_report": report,
            "class_names": class_names,
            "paths": paths,
        },
    )
    accuracy = float((preds == labels).mean())
    print_log(
        "val",
        f"accuracy {format_percent(accuracy)} | report {display_path(output_dir / 'classification_report.json')}",
        tone="success",
    )
    return {
        "device": runtime_device.label,
        "device_reason": runtime_device.reason,
        "accuracy": accuracy,
        "split": val_split,
        "checkpoint_path": str(checkpoint_path),
        "backbone_name": checkpoint["metadata"]["model"]["backbone_name"],
        "temperature": temperature,
        "output_dir": str(output_dir),
        "report_path": str(output_dir / "classification_report.json"),
        "confusion_matrix_path": str(output_dir / "confusion_matrix.png"),
        "random_predictions_path": str(output_dir / "random_predictions.png"),
        "gradcam_path": str(output_dir / "gradcam_examples.png"),
    }
