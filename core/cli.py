from __future__ import annotations

import argparse
from pathlib import Path

from core.console import (
    accent,
    accent_soft,
    display_path,
    dim,
    bright,
    info,
    print_error,
    print_banner,
    print_log,
    print_rank,
    print_summary,
)
from core.runtime import configure_runtime, prepare_matplotlib
from core.utils import format_percent


_GLYPH_HELP_ART = "\n".join(
    [
        " ██████╗ ██╗     ██╗   ██╗██████╗ ██╗  ██╗",
        "██╔════╝ ██║     ╚██╗ ██╔╝██╔══██╗██║  ██║",
        "██║  ███╗██║      ╚████╔╝ ██████╔╝███████║",
        "██║   ██║██║       ╚██╔╝  ██╔═══╝ ██╔══██║",
        "╚██████╔╝███████╗   ██║   ██║     ██║  ██║",
        " ╚═════╝ ╚══════╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝",
    ]
)


def _help_row(label: str, value: str, *, value_style=bright, note: str | None = None) -> str:
    line = f"{dim(label.ljust(11))} {value_style(value)}"
    if note:
        line = f"{line} {dim(note)}"
    return line


def _root_help_header(description: str | None) -> str:
    lines = [
        f"{accent('›')} {bright('glyph --help')}",
        "",
        str(info(_GLYPH_HELP_ART)),
        "",
        _help_row("Object:", "printed Imperial Aramaic letters"),
        _help_row("Pipeline:", "data -> train -> eval -> infer", value_style=accent_soft),
        _help_row("Dataset:", "dataset/", value_style=accent_soft),
        _help_row("Artifacts:", "artifacts/", value_style=accent_soft),
        _help_row("Launcher:", "./glyph", value_style=accent_soft),
    ]
    return "\n".join(lines)


def _command_help_header(prog: str, description: str | None) -> str:
    command_name = prog.split()[-1]
    lines = [f"{accent('›')} {bright(f'{prog} --help')}", ""]
    header_rows = {
        "data": [
            _help_row("Object:", "dataset root to generate", note="written to --output-dir"),
            _help_row("Output:", "dataset/", value_style=accent_soft),
            _help_row("Default:", "1200 train / 300 val per class"),
            _help_row("Render:", "128 canvas -> 64 output"),
        ],
        "train": [
            _help_row("Object:", "dataset/train + dataset/val"),
            _help_row("Output:", "artifacts/", value_style=accent_soft),
            _help_row("Default:", "30 epochs, batch 128, lr 1e-3"),
            _help_row("Model:", "ResNet-18 classifier"),
        ],
        "eval": [
            _help_row("Object:", "dataset/val"),
            _help_row("Checkpoint:", "--checkpoint FILE", value_style=accent_soft),
            _help_row("Output:", "artifacts/eval", value_style=accent_soft),
            _help_row("Default:", "batch 128"),
        ],
        "infer": [
            _help_row("Object:", "--image IMAGE", value_style=accent_soft, note="file path for prediction"),
            _help_row("Checkpoint:", "--checkpoint FILE", value_style=accent_soft),
            _help_row("Output:", "top-k ranked classes"),
            _help_row("Default:", "top_k = 3"),
        ],
    }
    lines.extend(header_rows.get(command_name, []))
    return "\n".join(lines)


class GlyphHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=30, width=100)

    def start_section(self, heading: str) -> None:
        heading_map = {
            "positional arguments": "Commands",
            "optional arguments": "Options",
            "options": "Options",
        }
        pretty_heading = heading_map.get(heading, heading.title())
        super().start_section(str(accent(pretty_heading)))

    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = f"{accent('Usage')} "
        return super().add_usage(usage, actions, groups, prefix)

    def _format_action_invocation(self, action):
        text = super()._format_action_invocation(action)
        if action.option_strings:
            return str(bright(text))
        return str(accent_soft(text))


class GlyphArgumentParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        text = super().format_help().rstrip()
        if self.description:
            text = text.replace(f"{self.description}\n\n", "", 1)
        header = (
            _root_help_header(self.description)
            if self.prog == "glyph"
            else _command_help_header(self.prog, self.description)
        )
        return "\n".join([header, "", text, ""])

    def error(self, message: str) -> None:
        self.print_usage()
        print_error(message)
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = GlyphArgumentParser(
        prog="glyph",
        description=(
            "Generate synthetic data, train a classifier, evaluate checkpoints, "
            "and infer printed Imperial Aramaic letters."
        ),
        formatter_class=GlyphHelpFormatter,
        epilog=(
            "Examples:\n"
            "  glyph data --output-dir dataset\n"
            "  glyph train --data-dir dataset --output-dir artifacts\n"
            "  glyph eval --data-dir dataset --checkpoint artifacts/best_model.pt\n"
            "  glyph infer --checkpoint artifacts/best_model.pt --image sample.png"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=GlyphArgumentParser
    )

    data_parser = subparsers.add_parser(
        "data",
        help="Generate a synthetic dataset.",
        description="Render a synthetic Imperial Aramaic dataset with split-specific corruption profiles.",
        formatter_class=GlyphHelpFormatter,
    )
    data_paths = data_parser.add_argument_group("Paths")
    data_paths.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="dataset output directory",
    )
    data_fonts = data_parser.add_argument_group("Fonts")
    data_fonts.add_argument(
        "--font",
        type=Path,
        action="append",
        default=[],
        help="explicit font path; repeatable",
    )
    data_fonts.add_argument(
        "--font-dir",
        type=Path,
        action="append",
        default=[],
        help="extra font directory; repeatable",
    )
    data_render = data_parser.add_argument_group("Generation")
    data_render.add_argument(
        "--train-per-class", type=int, default=1200, help="training images per class"
    )
    data_render.add_argument(
        "--val-per-class", type=int, default=300, help="validation images per class"
    )
    data_render.add_argument(
        "--canvas-size", type=int, default=128, help="internal render canvas size"
    )
    data_render.add_argument(
        "--output-size", type=int, default=64, help="final saved image size"
    )
    data_render.add_argument(
        "--train-hardness",
        type=float,
        default=0.95,
        help="training corruption severity",
    )
    data_render.add_argument(
        "--val-hardness",
        type=float,
        default=1.35,
        help="validation corruption severity",
    )
    data_render.add_argument(
        "--holdout-font-fraction",
        type=float,
        default=0.35,
        help="fraction of fonts reserved for validation",
    )
    data_runtime = data_parser.add_argument_group("Runtime")
    data_runtime.add_argument("--seed", type=int, default=42, help="random seed")
    data_parser.set_defaults(handler=handle_data)

    train_parser = subparsers.add_parser(
        "train",
        help="Train the classifier.",
        description="Train the Imperial Aramaic classifier on dataset/train and validate on dataset/val.",
        formatter_class=GlyphHelpFormatter,
    )
    train_paths = train_parser.add_argument_group("Paths")
    train_paths.add_argument(
        "--data-dir", type=Path, default=Path("dataset"), help="dataset root directory"
    )
    train_paths.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="checkpoint output directory",
    )
    train_opt = train_parser.add_argument_group("Optimization")
    train_opt.add_argument("--epochs", type=int, default=30, help="number of epochs")
    train_opt.add_argument("--batch-size", type=int, default=128, help="batch size")
    train_opt.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    train_model = train_parser.add_argument_group("Model")
    train_model.add_argument(
        "--pretrained",
        action="store_true",
        help="start from pretrained ResNet-18 weights",
    )
    train_model.add_argument(
        "--no-small-image-stem",
        action="store_true",
        help="disable the optimized small-image stem",
    )
    train_runtime = train_parser.add_argument_group("Runtime")
    train_runtime.add_argument(
        "--num-workers", type=int, default=0, help="dataloader worker count"
    )
    train_runtime.add_argument("--seed", type=int, default=42, help="random seed")
    train_parser.set_defaults(handler=handle_train)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a checkpoint.",
        description="Evaluate a trained checkpoint on the validation split and export reports.",
        formatter_class=GlyphHelpFormatter,
    )
    eval_paths = eval_parser.add_argument_group("Paths")
    eval_paths.add_argument(
        "--data-dir", type=Path, default=Path("dataset"), help="dataset root directory"
    )
    eval_paths.add_argument(
        "--checkpoint", type=Path, required=True, help="checkpoint file to evaluate"
    )
    eval_paths.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eval"),
        help="directory for evaluation outputs",
    )
    eval_runtime = eval_parser.add_argument_group("Runtime")
    eval_runtime.add_argument(
        "--batch-size", type=int, default=128, help="evaluation batch size"
    )
    eval_runtime.add_argument(
        "--num-workers", type=int, default=0, help="dataloader worker count"
    )
    eval_runtime.add_argument(
        "--seed", type=int, default=42, help="seed for sample visualizations"
    )
    eval_parser.set_defaults(handler=handle_eval)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference for one image.",
        description="Predict the most likely Imperial Aramaic classes for a single input image.",
        formatter_class=GlyphHelpFormatter,
    )
    infer_paths = infer_parser.add_argument_group("Paths")
    infer_paths.add_argument(
        "--checkpoint", type=Path, required=True, help="checkpoint file to load"
    )
    infer_paths.add_argument(
        "--image", type=Path, required=True, help="image file to classify"
    )
    infer_output = infer_parser.add_argument_group("Output")
    infer_output.add_argument(
        "--top-k", type=int, default=3, help="number of ranked predictions to show"
    )
    infer_parser.set_defaults(handler=handle_infer)

    return parser


def handle_data(args: argparse.Namespace) -> None:
    from core.datagen import DataGenConfig, build_dataset

    print_banner("glyph data", "synthetic dataset generation")
    print_summary(
        "Data Run",
        [
            ("output", display_path(args.output_dir)),
            ("train/class", args.train_per_class),
            ("val/class", args.val_per_class),
            ("train_hard", f"{args.train_hardness:.2f}"),
            ("val_hard", f"{args.val_hardness:.2f}"),
            ("seed", args.seed),
        ],
    )
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
    )
    print_summary(
        "Dataset Ready",
        [
            ("output", display_path(args.output_dir)),
            ("classes", metadata["num_classes"]),
            ("images", metadata["total_images"]),
            ("train/class", metadata["train_per_class"]),
            ("val/class", metadata["val_per_class"]),
            ("fonts", len(metadata["fonts"])),
        ],
    )
    print_log("data", "generation complete", tone="success")


def handle_train(args: argparse.Namespace) -> None:
    if not args.data_dir.exists():
        raise SystemExit(f"Dataset directory not found: {args.data_dir}")

    print_banner("glyph train", "model optimization")
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
    )
    print_summary(
        "Training Complete",
        [
            ("best_val_acc", format_percent(summary["best_val_acc"])),
            ("best_ckpt", display_path(summary["best_checkpoint"])),
            ("last_ckpt", display_path(summary["last_checkpoint"])),
            ("history", display_path(summary["history_path"])),
            ("curves", display_path(summary["curves_path"])),
        ],
    )
    print_log("train", "checkpoint export complete", tone="success")


def handle_eval(args: argparse.Namespace) -> None:
    if not args.data_dir.exists():
        raise SystemExit(f"Dataset directory not found: {args.data_dir}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    print_banner("glyph eval", "validation reports")
    prepare_matplotlib()
    from core.evaluation import evaluate_model

    summary = evaluate_model(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print_summary(
        "Evaluation Complete",
        [
            ("accuracy", format_percent(summary["accuracy"])),
            ("outputs", display_path(summary["output_dir"])),
            ("report", display_path(summary["report_path"])),
        ],
    )
    print_log("eval", "report export complete", tone="success")


def handle_infer(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    print_banner("glyph infer", "single-image prediction")
    from core.inference import predict_image

    result = predict_image(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        top_k=args.top_k,
    )
    print_summary(
        "Inference",
        [
            ("device", result["device"]),
            ("image", result["image_path"]),
            ("top_k", args.top_k),
        ],
    )
    for item in result["predictions"]:
        print_rank(item["rank"], item["label"], format_percent(item["confidence"]))


def main() -> None:
    configure_runtime()
    parser = build_parser()
    try:
        args = parser.parse_args()
        args.handler(args)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print_error(exc.code)
            raise SystemExit(1)
        raise
