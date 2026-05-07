from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from core.augmentations import build_eval_transforms
from core.checkpoints import (
    get_checkpoint_class_names,
    get_checkpoint_image_size,
    get_checkpoint_normalization,
    get_checkpoint_temperature,
    load_model_checkpoint,
)
from core.console import create_progress, display_path, print_log, print_summary
from core.datasets import ImageFolderDataset
from core.runtime import (
    optimize_model_for_device,
    prepare_image_batch,
    resolve_num_workers,
    resolve_runtime_device,
)
from core.utils import ensure_dir, format_float, save_json


def make_run_dir(
    root_dir: Path,
    image_dir: Path,
    *,
    run_name: str | None = None,
) -> Path:
    slug_source = run_name or image_dir.name or "images"
    slug = re.sub(r"[^a-z0-9]+", "_", slug_source.strip().lower()).strip("_") or "run"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root_dir / f"{timestamp}_{slug}"


def _build_ranked_predictions(
    probs: torch.Tensor, class_names: list[str], top_k: int
) -> list[list[dict[str, float | int | str]]]:
    k = min(top_k, probs.shape[1])
    top_probs, top_indices = torch.topk(probs, k=k, dim=1)
    rows: list[list[dict[str, float | int | str]]] = []

    for sample_scores, sample_indices in zip(top_probs.tolist(), top_indices.tolist()):
        rows.append(
            [
                {
                    "rank": rank,
                    "label": class_names[class_idx],
                    "confidence": float(score),
                }
                for rank, (score, class_idx) in enumerate(
                    zip(sample_scores, sample_indices),
                    start=1,
                )
            ]
        )
    return rows


def _read_image_array(image_path: Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(rgb)


def _normalize_lookup_key(value: str) -> str:
    normalized = Path(value).as_posix()
    return normalized[2:] if normalized.startswith("./") else normalized


def _relative_image_key(image_path: Path, image_dir: Path) -> str:
    try:
        return image_path.resolve().relative_to(image_dir.resolve()).as_posix()
    except ValueError:
        return image_path.relative_to(image_dir).as_posix()


def _load_true_labels(
    labels_csv: Path,
    *,
    file_column: str,
    label_column: str,
    class_names: list[str],
) -> dict[str, str]:
    with labels_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"Labels CSV is empty: {labels_csv}")
        missing_columns = {
            column
            for column in (file_column, label_column)
            if column not in reader.fieldnames
        }
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Labels CSV {labels_csv} is missing required columns: {missing_text}"
            )

        label_map: dict[str, str] = {}
        valid_labels = set(class_names)
        for row_index, row in enumerate(reader, start=2):
            file_value = (row.get(file_column) or "").strip()
            label_value = (row.get(label_column) or "").strip()
            if not file_value or not label_value:
                continue
            if label_value not in valid_labels:
                raise ValueError(
                    f"Unknown true label {label_value!r} in {labels_csv} at line {row_index}"
                )
            label_map[_normalize_lookup_key(file_value)] = label_value

    return label_map


def _lookup_true_label(
    image_path: Path, image_dir: Path, label_map: dict[str, str]
) -> str | None:
    keys = (
        _normalize_lookup_key(str(image_path)),
        _normalize_lookup_key(str(image_path.resolve())),
        _normalize_lookup_key(_relative_image_key(image_path, image_dir)),
        _normalize_lookup_key(image_path.name),
    )
    for key in keys:
        if key in label_map:
            return label_map[key]
    return None


def _save_prediction_rows_csv(output_path: Path, rows: list[dict]) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["file", "true_label", "pred_label", "top_k", "confidence"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file": row["file"],
                    "true_label": row["true_label"] or "",
                    "pred_label": row["pred_label"],
                    "top_k": json.dumps(row["top_k"], ensure_ascii=False),
                    "confidence": row["confidence"],
                }
            )


def _build_metrics_summary(
    rows: list[dict], class_names: list[str], top_k: int
) -> dict | None:
    labeled_rows = [row for row in rows if row["true_label"] is not None]
    if not labeled_rows:
        return None

    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    true_indices = np.asarray(
        [label_to_idx[row["true_label"]] for row in labeled_rows], dtype=np.int64
    )
    pred_indices = np.asarray(
        [label_to_idx[row["pred_label"]] for row in labeled_rows], dtype=np.int64
    )
    topk_hits = np.asarray(
        [row["true_label"] in row["top_k"] for row in labeled_rows], dtype=bool
    )

    cm = confusion_matrix(
        true_indices, pred_indices, labels=list(range(len(class_names)))
    )
    report = classification_report(
        true_indices,
        pred_indices,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_recall = {}
    for class_idx, class_name in enumerate(class_names):
        support = int(cm[class_idx].sum())
        recall = None if support == 0 else float(cm[class_idx, class_idx] / support)
        per_class_recall[class_name] = {
            "recall": recall,
            "support": support,
        }

    return {
        "num_labeled_samples": len(labeled_rows),
        "accuracy": float((true_indices == pred_indices).mean()),
        "top_k": top_k,
        "top_k_accuracy": float(topk_hits.mean()),
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm.tolist(),
        },
        "classification_report": report,
        "per_class_recall": per_class_recall,
    }


def predict_image(
    checkpoint_path: Path,
    image_path: Path,
    top_k: int = 3,
) -> dict:
    runtime_device = resolve_runtime_device()
    model, checkpoint = load_model_checkpoint(checkpoint_path, runtime_device.device)
    model = optimize_model_for_device(model, runtime_device)
    metadata = checkpoint["metadata"]
    class_names = get_checkpoint_class_names(checkpoint)
    image_size = get_checkpoint_image_size(checkpoint)
    mean, std = get_checkpoint_normalization(checkpoint)
    temperature = get_checkpoint_temperature(checkpoint)

    array = _read_image_array(image_path, image_size)
    transform = build_eval_transforms(mean, std)
    tensor = prepare_image_batch(
        transform(image=array)["image"].unsqueeze(0),
        runtime_device,
    )

    with torch.inference_mode():
        probs = torch.softmax(model(tensor) / temperature, dim=1)
        predictions = _build_ranked_predictions(
            probs=probs,
            class_names=class_names,
            top_k=top_k,
        )[0]

    return {
        "device": runtime_device.label,
        "device_reason": runtime_device.reason,
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "backbone_name": metadata["model"]["backbone_name"],
        "temperature": temperature,
        "predictions": predictions,
    }


def predict_folder(
    checkpoint_path: Path,
    image_dir: Path,
    output_dir: Path,
    *,
    labels_csv: Path | None = None,
    top_k: int = 3,
    batch_size: int = 128,
    num_workers: int | None = None,
    file_column: str = "file",
    label_column: str = "true_label",
    command_name: str = "scan",
    summary_title: str = "Folder Scan",
    run_context: dict | None = None,
) -> dict:
    ensure_dir(output_dir)
    runtime_device = resolve_runtime_device()
    resolved_num_workers = resolve_num_workers(num_workers)
    model, checkpoint = load_model_checkpoint(checkpoint_path, runtime_device.device)
    model = optimize_model_for_device(model, runtime_device)
    class_names = get_checkpoint_class_names(checkpoint)
    image_size = get_checkpoint_image_size(checkpoint)
    mean, std = get_checkpoint_normalization(checkpoint)
    temperature = get_checkpoint_temperature(checkpoint)

    label_map = (
        _load_true_labels(
            labels_csv,
            file_column=file_column,
            label_column=label_column,
            class_names=class_names,
        )
        if labels_csv is not None
        else {}
    )

    dataset = ImageFolderDataset(
        root=image_dir,
        transform=build_eval_transforms(mean, std),
        image_size=image_size,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": resolved_num_workers,
        "pin_memory": runtime_device.pin_memory,
    }
    if resolved_num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    print_summary(
        summary_title,
        [
            ("device", runtime_device.label),
            ("workers", resolved_num_workers),
            ("images", len(dataset)),
            ("checkpoint", display_path(checkpoint_path)),
            ("image_dir", display_path(image_dir)),
            ("labels_csv", display_path(labels_csv) if labels_csv else "none"),
            ("output", display_path(output_dir)),
        ],
    )

    rows: list[dict] = []
    with torch.inference_mode():
        iterator = create_progress(
            loader,
            command=command_name,
            scope="run",
            color="green",
            leave=True,
            total=len(loader),
        )
        for images, paths in iterator:
            images = prepare_image_batch(images, runtime_device)
            probs = torch.softmax(model(images) / temperature, dim=1).cpu()
            batch_predictions = _build_ranked_predictions(
                probs=probs,
                class_names=class_names,
                top_k=top_k,
            )

            for path_str, predictions in zip(paths, batch_predictions):
                image_path = Path(path_str)
                relative_path = _relative_image_key(image_path, image_dir)
                true_label = _lookup_true_label(image_path, image_dir, label_map)
                rows.append(
                    {
                        "file": relative_path,
                        "true_label": true_label,
                        "pred_label": predictions[0]["label"],
                        "top_k": [item["label"] for item in predictions],
                        "confidence": format_float(float(predictions[0]["confidence"])),
                    }
                )

    predictions_csv_path = output_dir / "predictions.csv"
    predictions_json_path = output_dir / "predictions.json"
    _save_prediction_rows_csv(predictions_csv_path, rows)
    save_json(predictions_json_path, {"predictions": rows})
    manifest_path = output_dir / "run.json"
    save_json(
        manifest_path,
        {
            "command": command_name,
            "checkpoint_path": str(checkpoint_path),
            "image_dir": str(image_dir),
            "labels_csv": None if labels_csv is None else str(labels_csv),
            "output_dir": str(output_dir),
            "num_images": len(rows),
            "top_k": top_k,
            "batch_size": batch_size,
            "num_workers": resolved_num_workers,
            "context": run_context or {},
        },
    )

    summary = _build_metrics_summary(rows, class_names, top_k)
    summary_path = None
    confusion_matrix_path = None
    if summary is not None:
        summary_path = output_dir / "summary.json"
        save_json(summary_path, summary)
        confusion_matrix_path = output_dir / "confusion_matrix.png"
        from core.evaluation import plot_confusion_matrix

        plot_confusion_matrix(
            np.asarray(summary["confusion_matrix"]["matrix"], dtype=np.int64),
            class_names,
            confusion_matrix_path,
        )
        print_log(
            command_name,
            (
                f"accuracy {summary['accuracy']:.2%} | "
                f"top-{top_k} {summary['top_k_accuracy']:.2%} | "
                f"summary {display_path(summary_path)}"
            ),
            tone="success",
        )
    elif labels_csv is not None:
        print_log(
            command_name,
            "no true labels matched; summary metrics were skipped",
            tone="warn",
        )

    return {
        "device": runtime_device.label,
        "device_reason": runtime_device.reason,
        "checkpoint_path": str(checkpoint_path),
        "backbone_name": checkpoint["metadata"]["model"]["backbone_name"],
        "temperature": temperature,
        "output_dir": str(output_dir),
        "num_images": len(rows),
        "num_labeled": 0 if summary is None else summary["num_labeled_samples"],
        "predictions_csv": str(predictions_csv_path),
        "predictions_json": str(predictions_json_path),
        "run_json": str(manifest_path),
        "summary_json": None if summary_path is None else str(summary_path),
        "confusion_matrix_png": (
            None if confusion_matrix_path is None else str(confusion_matrix_path)
        ),
    }
