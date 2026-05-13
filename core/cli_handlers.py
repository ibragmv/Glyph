from __future__ import annotations

import argparse
from pathlib import Path

from core.console import (
    display_path,
    print_banner,
    print_log,
    print_rank,
    print_summary,
)
from core.runtime import prepare_matplotlib
from core.utils import format_percent


def _validate_external_eval_args(args: argparse.Namespace) -> None:
    if not args.pt.is_file():
        raise SystemExit(f"Checkpoint not found: {args.pt}")
    if not args.dir.is_dir():
        raise SystemExit(f"Image directory not found: {args.dir}")
    if getattr(args, "csv", None) is not None and not args.csv.is_file():
        raise SystemExit(f"Labels CSV not found: {args.csv}")
    if args.top < 1:
        raise SystemExit("--top must be at least 1")
    if args.batch < 1:
        raise SystemExit("--batch must be at least 1")
    if args.work is not None and args.work < 0:
        raise SystemExit("--work cannot be negative")


def _resolve_run_output_dir(
    output_dir: Path | None,
    *,
    root_dir: Path,
    image_dir: Path,
    run_name: str | None,
) -> Path:
    if output_dir is not None:
        return output_dir

    from core.inference import make_run_dir

    return make_run_dir(root_dir=root_dir, image_dir=image_dir, run_name=run_name)


def handle_gen(args: argparse.Namespace) -> None:
    from core.datagen import DataGenConfig, build_dataset

    train_profiles = tuple(
        profile.strip() for profile in args.tmix.split(",") if profile.strip()
    )
    val_profiles = tuple(
        profile.strip() for profile in args.vmix.split(",") if profile.strip()
    )

    print_banner("glyph gen", "dataset run")
    print_summary(
        "Dataset Run",
        [
            ("output", display_path(args.out)),
            ("train/class", args.train),
            ("val/class", args.val),
            ("r_val", f"{args.rval:.2f}"),
            ("train_hard", f"{args.thard:.2f}"),
            ("val_hard", f"{args.vhard:.2f}"),
            ("train_profiles", ",".join(train_profiles) or "default"),
            ("val_profiles", ",".join(val_profiles) or "default"),
            ("jobs", "auto"),
            ("seed", args.seed),
        ],
    )
    metadata = build_dataset(
        DataGenConfig(
            output_dir=args.out,
            train_per_class=args.train,
            val_per_class=args.val,
            canvas_size=args.canvas,
            output_size=args.size,
            seed=args.seed,
            train_hardness=args.thard,
            val_hardness=args.vhard,
            real_val_fraction=args.rval,
            train_profiles=train_profiles,
            val_profiles=val_profiles,
        ),
    )
    print_summary(
        "Dataset Ready",
        [
            ("output", display_path(args.out)),
            ("classes", metadata["num_classes"]),
            ("images", metadata["total_images"]),
            ("synthetic", metadata["synthetic_total_images"]),
            ("real", metadata["real_total_images"]),
            ("jobs", metadata["jobs"]),
            ("train/class", metadata["train_per_class"]),
            ("val/class", metadata["val_per_class"]),
            ("val_split", metadata["primary_validation_split"]),
            (
                "preview_groups",
                sum(len(item) for item in metadata["preview_sheets"].values()),
            ),
        ],
    )
    print_log("gen", "dataset complete", tone="success")


def handle_train(args: argparse.Namespace) -> None:
    if not args.dir.exists():
        raise SystemExit(f"Dataset directory not found: {args.dir}")

    print_banner("glyph train", "checkpoint training")
    prepare_matplotlib()
    from core.training import train_model

    summary = train_model(
        data_dir=args.dir,
        output_dir=args.out,
        epochs=args.ep,
        batch_size=args.batch,
        lr=args.lr,
        num_workers=args.work,
        seed=args.seed,
        pretrained=args.pre,
        small_image_stem=not args.nostem,
        backbone_name=args.backbone,
        dropout=args.dropout,
        loss_name=args.loss,
        label_smoothing=args.smooth,
        focal_gamma=args.gamma,
        focal_alpha=args.alpha,
        temperature_scaling=args.tscale,
        temperature_max_iter=args.titer,
    )
    print_summary(
        "Training Summary",
        [
            ("device", summary["device"]),
            ("device_reason", summary["device_reason"]),
            ("best_val_acc", format_percent(summary["best_val_acc"])),
            ("best_epoch", summary["best_epoch"]),
            ("val_split", summary["val_split"]),
            ("best_checkpoint", display_path(summary["best_checkpoint"])),
            ("last_checkpoint", display_path(summary["last_checkpoint"])),
            ("history_json", display_path(summary["history_path"])),
            ("curves_png", display_path(summary["curves_path"])),
            ("diagnostics_png", display_path(summary["diagnostics_path"])),
            ("metrics_csv", display_path(summary["epoch_metrics_path"])),
            ("metadata_json", display_path(summary["metadata_path"])),
        ],
    )
    print_log("train", "training complete", tone="success")


def handle_val(args: argparse.Namespace) -> None:
    if not args.dir.exists():
        raise SystemExit(f"Dataset directory not found: {args.dir}")
    if not args.pt.is_file():
        raise SystemExit(f"Checkpoint not found: {args.pt}")

    print_banner("glyph val", "report export")
    prepare_matplotlib()
    from core.evaluation import evaluate_model

    summary = evaluate_model(
        data_dir=args.dir,
        checkpoint_path=args.pt,
        output_dir=args.out,
        batch_size=args.batch,
        num_workers=args.work,
        seed=args.seed,
    )
    print_summary(
        "Validation Summary",
        [
            ("device", summary["device"]),
            ("device_reason", summary["device_reason"]),
            ("accuracy", format_percent(summary["accuracy"])),
            ("split", summary["split"]),
            ("backbone", summary["backbone_name"]),
            ("temperature", f"{summary['temperature']:.4f}"),
            ("checkpoint", display_path(summary["checkpoint_path"])),
            ("output", display_path(summary["output_dir"])),
            ("report_json", display_path(summary["report_path"])),
            ("confusion_png", display_path(summary["confusion_matrix_path"])),
            ("samples_png", display_path(summary["random_predictions_path"])),
            ("gradcam_png", display_path(summary["gradcam_path"])),
        ],
    )
    print_log("val", "validation complete", tone="success")


def handle_pred(args: argparse.Namespace) -> None:
    if not args.pt.is_file():
        raise SystemExit(f"Checkpoint not found: {args.pt}")
    if not args.img.is_file():
        raise SystemExit(f"Image not found: {args.img}")
    if args.top < 1:
        raise SystemExit("--top must be at least 1")

    print_banner("glyph pred", "single-image classification")
    from core.inference import predict_image

    result = predict_image(
        checkpoint_path=args.pt,
        image_path=args.img,
        top_k=args.top,
    )
    top_prediction = result["predictions"][0]
    print_summary(
        "Prediction Summary",
        [
            ("device", result["device"]),
            ("device_reason", result["device_reason"]),
            ("backbone", result["backbone_name"]),
            ("temperature", f"{result['temperature']:.4f}"),
            ("checkpoint", display_path(result["checkpoint_path"])),
            ("image", result["image_path"]),
            ("top_k", args.top),
            (
                "top_1",
                f"{top_prediction['label']} ({format_percent(top_prediction['confidence'])})",
            ),
        ],
    )
    for item in result["predictions"]:
        print_rank(item["rank"], item["label"], format_percent(item["confidence"]))
    print_log("pred", "prediction complete", tone="success")


def handle_scan(args: argparse.Namespace) -> None:
    _validate_external_eval_args(args)
    from core.inference import predict_folder

    output_dir = _resolve_run_output_dir(
        args.out,
        root_dir=Path("artifacts/scans"),
        image_dir=args.dir,
        run_name=args.name,
    )
    print_banner("glyph scan", "folder processing")
    summary = predict_folder(
        checkpoint_path=args.pt,
        image_dir=args.dir,
        output_dir=output_dir,
        top_k=args.top,
        batch_size=args.batch,
        num_workers=args.work,
        command_name="scan",
        summary_title="Scan Job",
        run_context={"mode": "scan", "name": args.name},
    )
    print_summary(
        "Scan Summary",
        [
            ("device", summary["device"]),
            ("device_reason", summary["device_reason"]),
            ("backbone", summary["backbone_name"]),
            ("temperature", f"{summary['temperature']:.4f}"),
            ("checkpoint", display_path(summary["checkpoint_path"])),
            ("output", display_path(summary["output_dir"])),
            ("images", summary["num_images"]),
            ("predictions_csv", display_path(summary["predictions_csv"])),
            ("predictions_json", display_path(summary["predictions_json"])),
            ("run_json", display_path(summary["run_json"])),
        ],
    )
    print_log("scan", "scan complete", tone="success")


def handle_bench(args: argparse.Namespace) -> None:
    _validate_external_eval_args(args)
    prepare_matplotlib()
    from core.inference import predict_folder

    output_dir = _resolve_run_output_dir(
        args.out,
        root_dir=Path("artifacts/benchmarks"),
        image_dir=args.dir,
        run_name=args.name,
    )
    print_banner("glyph bench", "benchmark execution")
    summary = predict_folder(
        checkpoint_path=args.pt,
        image_dir=args.dir,
        output_dir=output_dir,
        labels_csv=args.csv,
        top_k=args.top,
        batch_size=args.batch,
        num_workers=args.work,
        file_column=args.file,
        label_column=args.label,
        command_name="bench",
        summary_title="Benchmark Job",
        run_context={"mode": "bench", "name": args.name},
    )
    summary_rows = [
        ("device", summary["device"]),
        ("device_reason", summary["device_reason"]),
        ("backbone", summary["backbone_name"]),
        ("temperature", f"{summary['temperature']:.4f}"),
        ("checkpoint", display_path(summary["checkpoint_path"])),
        ("output", display_path(summary["output_dir"])),
        ("images", summary["num_images"]),
        ("labeled", summary["num_labeled"]),
        ("predictions_csv", display_path(summary["predictions_csv"])),
        ("predictions_json", display_path(summary["predictions_json"])),
        ("run_json", display_path(summary["run_json"])),
    ]
    if summary["summary_json"] is not None:
        summary_rows.append(("summary_json", display_path(summary["summary_json"])))
    if summary["confusion_matrix_png"] is not None:
        summary_rows.append(
            ("confusion_png", display_path(summary["confusion_matrix_png"]))
        )
    print_summary("Benchmark Summary", summary_rows)
    if summary["summary_json"] is None:
        print_log(
            "bench",
            "benchmark saved predictions only; provide --csv with matching labels to compute metrics",
            tone="warn",
        )
    else:
        print_log("bench", "benchmark complete", tone="success")


def handle_check(args: argparse.Namespace) -> None:
    print_banner("glyph check", "readiness verification")
    from core.check import inspect_project

    report = inspect_project(
        data_dir=args.dir,
        checkpoint_path=args.pt,
    )
    print_summary(
        "Check Summary",
        [
            ("status", report["status"]),
            ("dataset", display_path(args.dir)),
            ("checkpoint", display_path(args.pt)),
            ("source", display_path(Path("source"))),
        ],
    )
    for item in report["checks"]:
        tone = (
            "success"
            if item["status"] == "ok"
            else "warn"
            if item["status"] == "warn"
            else "error"
        )
        print_log("check", f"{item['name']}: {item['message']}", tone=tone, wrap=True)

    if report["status"] == "fail":
        raise SystemExit(1)
    if args.strict and any(item["status"] != "ok" for item in report["checks"]):
        raise SystemExit(1)
    print_log("check", "setup passed", tone="success")


def handle_lint(args: argparse.Namespace) -> None:
    del args
    print_banner("glyph lint", "ruff verification")
    from core.qa import run_lint

    run_lint()


def handle_syntax(args: argparse.Namespace) -> None:
    del args
    print_banner("glyph syntax", "module compilation")
    from core.qa import run_syntax

    run_syntax()


def handle_smoke(args: argparse.Namespace) -> None:
    del args
    print_banner("glyph smoke", "smoke execution")
    from core.qa import run_smoke

    run_smoke()


def handle_qa(args: argparse.Namespace) -> None:
    del args
    print_banner("glyph qa", "project verification")
    from core.qa import run_qa

    run_qa()
