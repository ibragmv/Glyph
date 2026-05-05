from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from core.models import get_available_backbones
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
    print_warning,
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


def _root_help_header() -> str:
    lines = [
        f"{accent('›')} {bright('glyph --help')}",
        "",
        str(info(_GLYPH_HELP_ART)),
        "",
        _help_row("Task:", "classify printed Imperial Aramaic letters"),
        _help_row("Flow:", "build -> gen -> train -> val -> pred / scan / bench / check", value_style=accent_soft),
        _help_row("Code:", "lint / syntax / smoke / qa", value_style=accent_soft),
        _help_row("Input:", "fonts/ and image files", value_style=accent_soft),
        _help_row("Output:", "dataset/ and artifacts/", value_style=accent_soft),
        _help_row("Entry:", "./glyph", value_style=accent_soft),
    ]
    return "\n".join(lines)


def _command_help_header(prog: str) -> str:
    command_name = prog.split()[-1]
    lines = [f"{accent('›')} {bright(f'{prog} --help')}", ""]
    header_rows = {
        "build": [
            _help_row("Action:", "build the glyph launcher"),
            _help_row("Output:", "./glyph", value_style=accent_soft),
            _help_row("Runtime:", ".glyph/", value_style=accent_soft),
        ],
        "gen": [
            _help_row("Action:", "generate training and validation data"),
            _help_row("Output:", "dataset/", value_style=accent_soft),
            _help_row("Fonts:", "--font FILE / --fontdir DIR", value_style=accent_soft),
            _help_row("Default:", "1200 train / 300 val per class"),
            _help_row("Render:", "128 canvas -> 64 px output"),
        ],
        "train": [
            _help_row("Action:", "train a classifier from data"),
            _help_row("Input:", "dataset/train and dataset/val", value_style=accent_soft),
            _help_row("Output:", "artifacts/", value_style=accent_soft),
            _help_row("Default:", "30 epochs, batch 128, lr 1e-3"),
            _help_row("Model:", "resnet18 and mobile backbones"),
        ],
        "val": [
            _help_row("Action:", "run validation and save reports"),
            _help_row("Dataset:", "--dir DIR", value_style=accent_soft, note="default dataset"),
            _help_row("Checkpoint:", "--pt FILE", value_style=accent_soft),
            _help_row("Output:", "--out DIR", value_style=accent_soft, note="default artifacts/val"),
            _help_row("Default:", "batch 128"),
        ],
        "pred": [
            _help_row("Action:", "predict one image"),
            _help_row("Image:", "--img FILE", value_style=accent_soft),
            _help_row("Checkpoint:", "--pt FILE", value_style=accent_soft),
            _help_row("Output:", "top-k ranked classes and confidence"),
            _help_row("Default:", "top = 3"),
        ],
        "scan": [
            _help_row("Action:", "scan a folder and save predictions"),
            _help_row("Input:", "DIR with images", value_style=accent_soft),
            _help_row("Checkpoint:", "--pt FILE", value_style=accent_soft),
            _help_row("Output:", "artifacts/scans/<stamp>_name", value_style=accent_soft),
            _help_row("Default:", "batch 128, top = 3"),
        ],
        "bench": [
            _help_row("Action:", "benchmark external images"),
            _help_row("Input:", "DIR with images", value_style=accent_soft),
            _help_row("Labels:", "--csv FILE", value_style=accent_soft, note="optional labels for metrics"),
            _help_row("Output:", "artifacts/benchmarks/<stamp>_name", value_style=accent_soft),
            _help_row("Default:", "batch 128, top = 3"),
        ],
        "check": [
            _help_row("Action:", "check setup readiness", value_style=accent_soft),
            _help_row("Dataset:", "--dir DIR", value_style=accent_soft, note="default dataset"),
            _help_row("Checkpoint:", "--pt FILE", value_style=accent_soft, note="default artifacts/best_model.pt"),
            _help_row("Checks:", "dataset, checkpoint, fonts, runtime"),
            _help_row("Default:", "warnings shown, broken state fails"),
        ],
        "lint": [
            _help_row("Action:", "run ruff on the project"),
            _help_row("Scope:", "core, scripts, tests", value_style=accent_soft),
            _help_row("Output:", "pass or fail", value_style=accent_soft),
        ],
        "syntax": [
            _help_row("Action:", "compile project modules"),
            _help_row("Scope:", "core, scripts, tests", value_style=accent_soft),
            _help_row("Output:", "pass or fail", value_style=accent_soft),
        ],
        "smoke": [
            _help_row("Action:", "run smoke suite"),
            _help_row("Scope:", "tests/smoke.py", value_style=accent_soft),
            _help_row("Output:", "pass or fail", value_style=accent_soft),
        ],
        "qa": [
            _help_row("Action:", "run full fast verification"),
            _help_row("Steps:", "lint, syntax, smoke", value_style=accent_soft),
            _help_row("Output:", "pass or fail", value_style=accent_soft),
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
        return super()._format_action_invocation(action)


class GlyphArgumentParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        text = super().format_help().rstrip()
        if self.description:
            text = text.replace(f"{self.description}\n\n", "", 1)
        header = (
            _root_help_header()
            if self.prog == "glyph"
            else _command_help_header(self.prog)
        )
        return "\n".join([header, "", text, ""])

    def error(self, message: str) -> None:
        self.print_usage()
        print_error(message)
        raise SystemExit(2)


def _configure_scan_parser(parser: argparse.ArgumentParser) -> None:
    scan_paths = parser.add_argument_group("Paths")
    scan_paths.add_argument("dir", type=Path, help="input directory with images")
    scan_paths.add_argument(
        "--pt", type=Path, required=True, help="checkpoint file to load"
    )
    scan_paths.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional output directory; default is a new run under artifacts/scans",
    )
    scan_paths.add_argument(
        "--name",
        type=str,
        default=None,
        help="optional label appended to the generated run directory",
    )
    scan_output = parser.add_argument_group("Output")
    scan_output.add_argument(
        "--top", type=int, default=3, help="number of ranked predictions to export"
    )
    scan_runtime = parser.add_argument_group("Runtime")
    scan_runtime.add_argument(
        "--batch", type=int, default=128, help="batch size for folder processing"
    )
    scan_runtime.add_argument(
        "--work", type=int, default=0, help="DataLoader worker count"
    )


def _configure_bench_parser(parser: argparse.ArgumentParser) -> None:
    bench_paths = parser.add_argument_group("Paths")
    bench_paths.add_argument("dir", type=Path, help="input directory with images")
    bench_paths.add_argument(
        "--pt", type=Path, required=True, help="checkpoint file to load"
    )
    bench_paths.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="optional CSV with file and true_label columns",
    )
    bench_paths.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional output directory; default is a new run under artifacts/benchmarks",
    )
    bench_paths.add_argument(
        "--name",
        type=str,
        default=None,
        help="optional label appended to the generated run directory",
    )
    bench_output = parser.add_argument_group("Output")
    bench_output.add_argument(
        "--top", type=int, default=3, help="number of ranked predictions to export"
    )
    bench_output.add_argument(
        "--file",
        type=str,
        default="file",
        help="CSV column containing image paths",
    )
    bench_output.add_argument(
        "--label",
        type=str,
        default="true_label",
        help="CSV column containing labels",
    )
    bench_runtime = parser.add_argument_group("Runtime")
    bench_runtime.add_argument(
        "--batch", type=int, default=128, help="batch size for benchmark processing"
    )
    bench_runtime.add_argument(
        "--work", type=int, default=0, help="DataLoader worker count"
    )


def _configure_check_parser(parser: argparse.ArgumentParser) -> None:
    check_paths = parser.add_argument_group("Paths")
    check_paths.add_argument(
        "--dir", type=Path, default=Path("dataset"), help="dataset root directory"
    )
    check_paths.add_argument(
        "--pt",
        type=Path,
        default=Path("artifacts/best_model.pt"),
        help="checkpoint file to inspect",
    )
    check_fonts = parser.add_argument_group("Fonts")
    check_fonts.add_argument(
        "--fontdir",
        type=Path,
        action="append",
        default=[],
        help="additional font directory; repeatable",
    )
    check_runtime = parser.add_argument_group("Runtime")
    check_runtime.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as a failing exit code",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = GlyphArgumentParser(
        prog="glyph",
        description=(
            "Build datasets, train checkpoints, export evaluation reports, and "
            "process printed Imperial Aramaic image inputs."
        ),
        formatter_class=GlyphHelpFormatter,
        epilog=(
            "Examples:\n"
            "  glyph build\n"
            "  glyph gen --out dataset\n"
            "  glyph train --dir dataset --out artifacts\n"
            "  glyph val --dir dataset --pt artifacts/best_model.pt\n"
            "  glyph pred --pt artifacts/best_model.pt --img sample.png\n"
            "  glyph scan --pt artifacts/best_model.pt external_scans\n"
            "  glyph bench --pt artifacts/best_model.pt --csv labels.csv external_scans\n"
            "  glyph check\n"
            "  glyph qa"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=GlyphArgumentParser
    )

    build_parser = subparsers.add_parser(
        "build",
        help="Build the glyph launcher.",
        description="Build the local glyph launcher and package the runtime.",
        formatter_class=GlyphHelpFormatter,
    )
    build_parser.set_defaults(handler=handle_build)

    gen_parser = subparsers.add_parser(
        "gen",
        help="Generate a dataset.",
        description="Generate an Imperial Aramaic dataset with split-specific profiles.",
        formatter_class=GlyphHelpFormatter,
    )
    gen_paths = gen_parser.add_argument_group("Paths")
    gen_paths.add_argument(
        "--out",
        type=Path,
        default=Path("dataset"),
        help="dataset output directory",
    )
    gen_fonts = gen_parser.add_argument_group("Fonts")
    gen_fonts.add_argument(
        "--font",
        type=Path,
        action="append",
        default=[],
        help="explicit font path; repeatable",
    )
    gen_fonts.add_argument(
        "--fontdir",
        type=Path,
        action="append",
        default=[],
        help="extra font directory; repeatable",
    )
    gen_render = gen_parser.add_argument_group("Generation")
    gen_render.add_argument(
        "--train", type=int, default=1200, help="training images per class"
    )
    gen_render.add_argument(
        "--val", type=int, default=300, help="validation images per class"
    )
    gen_render.add_argument(
        "--canvas", type=int, default=128, help="internal render canvas size"
    )
    gen_render.add_argument(
        "--size", type=int, default=64, help="final saved image size"
    )
    gen_render.add_argument(
        "--thard",
        type=float,
        default=0.95,
        help="train split render severity",
    )
    gen_render.add_argument(
        "--vhard",
        type=float,
        default=1.35,
        help="validation split render severity",
    )
    gen_render.add_argument(
        "--holdout",
        type=float,
        default=0.35,
        help="fraction of fonts reserved for validation-only use",
    )
    gen_render.add_argument(
        "--tmix",
        type=str,
        default="",
        help="comma-separated generation profiles for train split",
    )
    gen_render.add_argument(
        "--vmix",
        type=str,
        default="",
        help="comma-separated generation profiles for validation split",
    )
    gen_runtime = gen_parser.add_argument_group("Runtime")
    gen_runtime.add_argument("--seed", type=int, default=42, help="random seed")
    gen_parser.set_defaults(handler=handle_gen)

    train_parser = subparsers.add_parser(
        "train",
        help="Train a classifier.",
        description="Train the classifier on dataset/train and evaluate on dataset/val.",
        formatter_class=GlyphHelpFormatter,
    )
    train_paths = train_parser.add_argument_group("Paths")
    train_paths.add_argument(
        "--dir", type=Path, default=Path("dataset"), help="dataset root directory"
    )
    train_paths.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts"),
        help="output directory for checkpoints and reports",
    )
    train_opt = train_parser.add_argument_group("Optimization")
    train_opt.add_argument("--ep", type=int, default=30, help="number of epochs")
    train_opt.add_argument("--batch", type=int, default=128, help="batch size")
    train_opt.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    train_model = train_parser.add_argument_group("Model")
    train_model.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=get_available_backbones(),
        help="model backbone",
    )
    train_model.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="dropout before the classification head",
    )
    train_model.add_argument(
        "--pre",
        action="store_true",
        help="initialize from pretrained backbone weights",
    )
    train_model.add_argument(
        "--nostem",
        action="store_true",
        help="disable the optimized small-image stem for ResNet backbones",
    )
    train_loss = train_parser.add_argument_group("Loss")
    train_loss.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=("ce", "focal"),
        help="classification loss",
    )
    train_loss.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="label smoothing factor",
    )
    train_loss.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="gamma parameter for focal loss",
    )
    train_loss.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="optional alpha multiplier for focal loss",
    )
    train_calibration = train_parser.add_argument_group("Calibration")
    train_calibration.add_argument(
        "--tscale",
        action="store_true",
        help="fit temperature scaling on validation logits after training",
    )
    train_calibration.add_argument(
        "--titer",
        type=int,
        default=50,
        help="optimizer iterations for temperature scaling",
    )
    train_runtime = train_parser.add_argument_group("Runtime")
    train_runtime.add_argument(
        "--work", type=int, default=0, help="dataloader worker count"
    )
    train_runtime.add_argument("--seed", type=int, default=42, help="random seed")
    train_parser.set_defaults(handler=handle_train)

    val_parser = subparsers.add_parser(
        "val",
        help="Run validation.",
        description="Evaluate a checkpoint on the validation split and save reports.",
        formatter_class=GlyphHelpFormatter,
    )
    val_paths = val_parser.add_argument_group("Paths")
    val_paths.add_argument(
        "--dir", type=Path, default=Path("dataset"), help="dataset root directory"
    )
    val_paths.add_argument(
        "--pt", type=Path, required=True, help="checkpoint file to evaluate"
    )
    val_paths.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/val"),
        help="output directory for validation artifacts",
    )
    val_runtime = val_parser.add_argument_group("Runtime")
    val_runtime.add_argument(
        "--batch", type=int, default=128, help="evaluation batch size"
    )
    val_runtime.add_argument(
        "--work", type=int, default=0, help="dataloader worker count"
    )
    val_runtime.add_argument(
        "--seed", type=int, default=42, help="seed for sampled visualizations"
    )
    val_parser.set_defaults(handler=handle_val)

    pred_parser = subparsers.add_parser(
        "pred",
        help="Predict one image.",
        description="Predict the most likely class for a single input image.",
        formatter_class=GlyphHelpFormatter,
    )
    pred_paths = pred_parser.add_argument_group("Paths")
    pred_paths.add_argument(
        "--pt", type=Path, required=True, help="checkpoint file to load"
    )
    pred_paths.add_argument(
        "--img", type=Path, required=True, help="input image file"
    )
    pred_output = pred_parser.add_argument_group("Output")
    pred_output.add_argument(
        "--top", type=int, default=3, help="number of ranked predictions to display"
    )
    pred_parser.set_defaults(handler=handle_pred)

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan an image folder.",
        description=(
            "Scan a folder of images, save ranked predictions, and create "
            "a run directory under artifacts/scans by default."
        ),
        formatter_class=GlyphHelpFormatter,
    )
    _configure_scan_parser(scan_parser)
    scan_parser.set_defaults(handler=handle_scan)

    bench_parser = subparsers.add_parser(
        "bench",
        help="Run a benchmark.",
        description=(
            "Run evaluation on a folder of images, optionally join labels "
            "from CSV, compute metrics when labels match, and save the run under "
            "artifacts/benchmarks by default."
        ),
        formatter_class=GlyphHelpFormatter,
    )
    _configure_bench_parser(bench_parser)
    bench_parser.set_defaults(handler=handle_bench)

    check_parser = subparsers.add_parser(
        "check",
        help="Check setup readiness.",
        description=(
            "Inspect dataset, checkpoint, class-count consistency, font availability, "
            "and runtime readiness before generation, validation, or external runs."
        ),
        formatter_class=GlyphHelpFormatter,
    )
    _configure_check_parser(check_parser)
    check_parser.set_defaults(handler=handle_check)

    lint_parser = subparsers.add_parser(
        "lint",
        help="Run lint checks.",
        description="Run ruff on the project.",
        formatter_class=GlyphHelpFormatter,
    )
    lint_parser.set_defaults(handler=handle_lint)

    syntax_parser = subparsers.add_parser(
        "syntax",
        help="Run syntax checks.",
        description="Compile project modules and verify syntax.",
        formatter_class=GlyphHelpFormatter,
    )
    syntax_parser.set_defaults(handler=handle_syntax)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run smoke checks.",
        description="Run the smoke suite for the CLI and core flows.",
        formatter_class=GlyphHelpFormatter,
    )
    smoke_parser.set_defaults(handler=handle_smoke)

    qa_parser = subparsers.add_parser(
        "qa",
        help="Run fast checks.",
        description="Run lint, syntax, and smoke in one pass.",
        formatter_class=GlyphHelpFormatter,
    )
    qa_parser.set_defaults(handler=handle_qa)

    return parser


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
    if args.work < 0:
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


def _resolve_project_root() -> Path:
    env_root = os.environ.get("GLYPH_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if (root / "requirements.txt").is_file() and (root / "core").is_dir():
            return root

    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "requirements.txt").is_file()
            and (candidate / "core").is_dir()
            and (candidate / "scripts").is_dir()
        ):
            return candidate

    raise RuntimeError("Project root could not be resolved.")


def handle_gen(args: argparse.Namespace) -> None:
    from core.datagen import DataGenConfig, build_dataset

    train_profiles = tuple(
        profile.strip()
        for profile in args.tmix.split(",")
        if profile.strip()
    )
    val_profiles = tuple(
        profile.strip() for profile in args.vmix.split(",") if profile.strip()
    )

    print_banner("glyph gen", "dataset build")
    print_summary(
        "Dataset Build",
        [
            ("output", display_path(args.out)),
            ("train/class", args.train),
            ("val/class", args.val),
            ("train_hard", f"{args.thard:.2f}"),
            ("val_hard", f"{args.vhard:.2f}"),
            ("train_profiles", ",".join(train_profiles) or "default"),
            ("val_profiles", ",".join(val_profiles) or "default"),
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
            explicit_fonts=tuple(args.font),
            extra_font_dirs=tuple(args.fontdir),
            train_hardness=args.thard,
            val_hardness=args.vhard,
            holdout_font_fraction=args.holdout,
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
            ("train/class", metadata["train_per_class"]),
            ("val/class", metadata["val_per_class"]),
            ("fonts", len(metadata["fonts"])),
            ("preview_groups", sum(len(item) for item in metadata["preview_sheets"].values())),
        ],
    )
    print_log("gen", "dataset complete", tone="success")


def handle_build(args: argparse.Namespace) -> None:
    del args
    print_banner("glyph build", "launcher packaging")
    from core.builder import build_launcher

    root_dir = _resolve_project_root()
    built_path = build_launcher(
        output_path=root_dir / "glyph",
        python_executable=Path(sys.executable),
        project_root=root_dir,
    )
    print_log("build", f"complete {display_path(built_path)}", tone="success")


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
            ("best_val_acc", format_percent(summary["best_val_acc"])),
            ("best_epoch", summary["best_epoch"]),
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
            ("accuracy", format_percent(summary["accuracy"])),
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
        extra_font_dirs=tuple(args.fontdir),
    )
    print_summary(
        "Check Summary",
        [
            ("status", report["status"]),
            ("dataset", display_path(args.dir)),
            ("checkpoint", display_path(args.pt)),
            ("font_dirs", ", ".join(str(path) for path in args.fontdir) or "default"),
        ],
    )
    for item in report["checks"]:
        tone = "success" if item["status"] == "ok" else "warn" if item["status"] == "warn" else "error"
        print_log("check", f"{item['name']}: {item['message']}", tone=tone, wrap=True)

    if report["status"] == "fail":
        raise SystemExit(1)
    if args.strict and any(item["status"] != "ok" for item in report["checks"]):
        raise SystemExit(1)
    print_log("check", "setup check passed", tone="success")


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
    print_banner("glyph qa", "fast verification")
    from core.qa import run_qa

    run_qa()


def main(argv: list[str] | None = None) -> None:
    configure_runtime()
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parser.parse_args(raw_argv)
        args.handler(args)
    except KeyboardInterrupt:
        print_warning("Execution interrupted.")
        raise SystemExit(130)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print_error(exc.code)
            raise SystemExit(1)
        raise
    except Exception as exc:
        command = getattr(locals().get("args", None), "command", "glyph")
        print_error(f"{command} failed: {exc}")
        if os.environ.get("GLYPH_DEBUG") == "1":
            traceback.print_exc()
        else:
            print_log(
                command,
                "Set GLYPH_DEBUG=1 to print the full traceback.",
                tone="warn",
            )
        raise SystemExit(1)
