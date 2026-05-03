from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from core.augmentations import build_eval_transforms
from core.checkpoints import load_model_checkpoint
from core.console import create_progress, display_path, print_log, print_summary
from core.constants import CLASS_NAMES
from core.datasets import ImperialAramaicDataset
from core.gradcam import GradCAM, overlay_heatmap
from core.utils import ensure_dir, format_percent, save_json


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    all_probs = []
    all_preds = []
    all_labels = []
    all_paths: list[str] = []

    with torch.no_grad():
        iterator = create_progress(
            loader,
            command="eval",
            scope="predict",
            color="cyan",
            leave=False,
            total=len(loader),
        )
        for images, labels, paths in iterator:
            images = images.to(device, non_blocking=True)
            logits = model(images)
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
            f"P:{CLASS_NAMES[predicted_idx]}\nT:{CLASS_NAMES[true_idx]}\n{confidence:.2%}",
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
    device: torch.device,
    output_path: Path,
    indices: list[int],
    mean: float,
    std: float,
) -> None:
    import matplotlib.pyplot as plt

    gradcam = GradCAM(model=model, target_layer=model.layer4[-1].conv2)
    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 3 * len(indices)))
    if len(indices) == 1:
        axes = np.array([axes])

    for row, dataset_idx in enumerate(indices):
        image_tensor, label, _ = dataset[dataset_idx]
        input_tensor = image_tensor.unsqueeze(0).to(device)
        heatmap = gradcam.generate(input_tensor)
        grayscale = image_tensor.squeeze(0).numpy()
        grayscale = np.clip((grayscale * std) + mean, 0.0, 1.0)
        overlay = overlay_heatmap(grayscale, heatmap)

        with torch.no_grad():
            prediction = model(input_tensor).argmax(dim=1).item()

        axes[row, 0].imshow(grayscale, cmap="gray")
        axes[row, 0].set_title(f"Input: {CLASS_NAMES[label]}")
        axes[row, 0].set_axis_off()

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"Grad-CAM: {CLASS_NAMES[prediction]}")
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
    num_workers: int = 0,
    seed: int = 42,
) -> dict:
    ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device)

    mean = checkpoint["mean"]
    std = checkpoint["std"]
    dataset = ImperialAramaicDataset(
        root=data_dir,
        split="val",
        transform=build_eval_transforms(mean, std),
        return_paths=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print_summary(
        "Evaluation Run",
        [
            ("device", device.type),
            ("samples", len(dataset)),
            ("checkpoint", display_path(checkpoint_path)),
            ("output", display_path(output_dir)),
        ],
    )
    probs, preds, labels, paths = collect_predictions(model, loader, device)
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        labels,
        preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    plot_confusion_matrix(cm, CLASS_NAMES, output_dir / "confusion_matrix.png")
    selected_indices = plot_random_predictions(
        dataset=dataset,
        probs=probs,
        preds=preds,
        labels=labels,
        output_path=output_dir / "random_predictions.png",
        mean=mean,
        std=std,
        seed=seed,
    )
    plot_gradcam_examples(
        model=model,
        dataset=dataset,
        device=device,
        output_path=output_dir / "gradcam_examples.png",
        indices=selected_indices[: min(6, len(selected_indices))],
        mean=mean,
        std=std,
    )

    save_json(
        output_dir / "classification_report.json",
        {
            "classification_report": report,
            "class_names": CLASS_NAMES,
            "paths": paths,
        },
    )
    accuracy = float((preds == labels).mean())
    print_log(
        "eval",
        f"accuracy {format_percent(accuracy)} | report {display_path(output_dir / 'classification_report.json')}",
        tone="success",
    )
    return {
        "device": device.type,
        "accuracy": accuracy,
        "output_dir": str(output_dir),
        "report_path": str(output_dir / "classification_report.json"),
    }
