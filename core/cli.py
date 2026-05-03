from __future__ import annotations

import argparse
from pathlib import Path

from core.runtime import configure_runtime, prepare_matplotlib
from core.utils import format_percent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glyph",
        description="Imperial Aramaic glyph classification toolkit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="Generate a synthetic dataset.")
    data_parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    data_parser.add_argument("--train-per-class", type=int, default=1200)
    data_parser.add_argument("--val-per-class", type=int, default=300)
    data_parser.add_argument("--canvas-size", type=int, default=128)
    data_parser.add_argument("--output-size", type=int, default=64)
    data_parser.add_argument("--seed", type=int, default=42)
    data_parser.add_argument("--font", type=Path, action="append", default=[])
    data_parser.add_argument("--font-dir", type=Path, action="append", default=[])
    data_parser.add_argument("--train-hardness", type=float, default=0.95)
    data_parser.add_argument("--val-hardness", type=float, default=1.35)
    data_parser.add_argument("--holdout-font-fraction", type=float, default=0.35)
    data_parser.add_argument("--progress", action="store_true")
    data_parser.set_defaults(handler=handle_data)

    train_parser = subparsers.add_parser("train", help="Train the classifier.")
    train_parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--pretrained", action="store_true")
    train_parser.add_argument("--no-small-image-stem", action="store_true")
    train_parser.add_argument("--progress", action="store_true")
    train_parser.set_defaults(handler=handle_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint.")
    eval_parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval"))
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--num-workers", type=int, default=0)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--progress", action="store_true")
    eval_parser.set_defaults(handler=handle_eval)

    infer_parser = subparsers.add_parser("infer", help="Run inference for one image.")
    infer_parser.add_argument("--checkpoint", type=Path, required=True)
    infer_parser.add_argument("--image", type=Path, required=True)
    infer_parser.add_argument("--top-k", type=int, default=3)
    infer_parser.set_defaults(handler=handle_infer)

    return parser


def handle_data(args: argparse.Namespace) -> None:
    from core.datagen import DataGenConfig, build_dataset

    metadata = build_dataset(
        DataGenConfig(
            output_dir=args.output_dir,
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
            canvas_size=args.canvas_size,
            output_size=args.output_size,
            seed=args.seed,
            explicit_fonts=tuple(args.font),
            extra_font_dirs=tuple(args.font_dir),
            train_hardness=args.train_hardness,
            val_hardness=args.val_hardness,
            holdout_font_fraction=args.holdout_font_fraction,
        ),
        show_progress=args.progress,
        logger=print,
    )
    print(
        f"[data] ready: {metadata['total_images']} images across {metadata['num_classes']} classes in {args.output_dir}"
    )


def handle_train(args: argparse.Namespace) -> None:
    if not args.data_dir.exists():
        raise SystemExit(f"Dataset directory not found: {args.data_dir}")

    prepare_matplotlib()
    from core.training import train_model

    summary = train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=args.pretrained,
        small_image_stem=not args.no_small_image_stem,
        show_progress=args.progress,
    )
    print(
        f"[train] complete: best val acc {format_percent(summary['best_val_acc'])} | checkpoint {summary['best_checkpoint']}"
    )


def handle_eval(args: argparse.Namespace) -> None:
    if not args.data_dir.exists():
        raise SystemExit(f"Dataset directory not found: {args.data_dir}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    prepare_matplotlib()
    from core.evaluation import evaluate_model

    summary = evaluate_model(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        show_progress=args.progress,
    )
    print(
        f"[eval] complete: accuracy {format_percent(summary['accuracy'])} | outputs {args.output_dir}"
    )


def handle_infer(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    from core.inference import predict_image

    result = predict_image(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        top_k=args.top_k,
    )
    print(f"[infer] device: {result['device']}")
    for item in result["predictions"]:
        print(f"{item['rank']}. {item['label']} ({format_percent(item['confidence'])})")


def main() -> None:
    configure_runtime()
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
