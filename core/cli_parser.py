from __future__ import annotations

import argparse
from pathlib import Path

from core.cli_handlers import (
    handle_bench,
    handle_check,
    handle_gen,
    handle_lint,
    handle_pred,
    handle_qa,
    handle_scan,
    handle_smoke,
    handle_syntax,
    handle_train,
    handle_val,
)
from core.console import accent, accent_soft, bright, cosmic_orange_block, dim
from core.models import get_available_backbones


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


def _help_row(
    label: str, value: str, *, value_style=bright, note: str | None = None
) -> str:
    line = f"{dim(label.ljust(11))} {value_style(value)}"
    if note:
        line = f"{line} {dim(note)}"
    return line


def _root_help_header() -> str:
    lines = [
        f"{accent('›')} {bright('glyph --help')}",
        "",
        cosmic_orange_block(_GLYPH_HELP_ART),
        "",
        _help_row(
            "Flow:",
            "gen | train | val | pred | scan | bench | check",
            value_style=accent_soft,
        ),
        _help_row("QA:", "lint | syntax | smoke | qa", value_style=accent_soft),
    ]
    return "\n".join(lines)


def _command_help_header(prog: str) -> str:
    command_name = prog.split()[-1]
    lines = [f"{accent('›')} {bright(f'{prog} --help')}", ""]
    summaries = {
        "gen": "generate rgb dataset splits from real and exemplar sources",
        "train": "train on rgb synthetic plus real splits",
        "val": "run validation on the primary split",
        "pred": "predict a single image",
        "scan": "scan a folder and save predictions",
        "bench": "benchmark labeled or unlabeled folders",
        "check": "verify source assets, dataset, checkpoint, runtime",
        "lint": "run ruff",
        "syntax": "compile Python modules",
        "smoke": "run smoke tests",
        "qa": "run lint + syntax + smoke",
    }
    if command_name in summaries:
        lines.append(
            _help_row("About:", summaries[command_name], value_style=accent_soft)
        )
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

    def _format_action(self, action):
        text = super()._format_action(action)
        if isinstance(action, argparse._SubParsersAction):
            lines = text.splitlines()
            if len(lines) > 1:
                text = "\n".join(lines[1:])
        return text


class GlyphArgumentParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        text = super().format_help().rstrip()
        if self.description:
            text = text.replace(f"{self.description}\n\n", "", 1)
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if line.startswith("usage: "):
                lines[idx] = line.replace("usage:", accent("Usage:"), 1)
                break
        text = "\n".join(lines)
        header = (
            _root_help_header()
            if self.prog == "glyph"
            else _command_help_header(self.prog)
        )
        return "\n".join([header, "", text, ""])

    def error(self, message: str) -> None:
        from core.console import print_error

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
        "--work",
        type=int,
        default=None,
        help="DataLoader worker count; default is auto",
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
        "--work",
        type=int,
        default=None,
        help="DataLoader worker count; default is auto",
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
            "Build exemplar-driven datasets, train checkpoints against real validation, "
            "export evaluation reports, and process printed Aramaic image inputs."
        ),
        formatter_class=GlyphHelpFormatter,
        epilog=(
            "Examples:\n"
            "  glyph gen --out dataset\n"
            "  glyph train --dir dataset --out artifacts\n"
            "  glyph pred --pt artifacts/best_model.pt --img sample.png\n"
            "  glyph qa"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=GlyphArgumentParser
    )

    gen_parser = subparsers.add_parser(
        "gen",
        help="Generate a dataset.",
        description="Generate RGB dataset splits from exemplar and real source images.",
        formatter_class=GlyphHelpFormatter,
    )
    gen_paths = gen_parser.add_argument_group("Paths")
    gen_paths.add_argument(
        "--out",
        type=Path,
        default=Path("dataset"),
        help="dataset output directory",
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
        "--rval",
        type=float,
        default=0.25,
        help="fraction of real crops reserved for real validation",
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
        description="Train the classifier on RGB synthetic plus real splits and monitor the primary validation split.",
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
        "--work",
        type=int,
        default=None,
        help="dataloader worker count; default is auto",
    )
    train_runtime.add_argument("--seed", type=int, default=42, help="random seed")
    train_parser.set_defaults(handler=handle_train)

    val_parser = subparsers.add_parser(
        "val",
        help="Run validation.",
        description="Evaluate a checkpoint on the primary validation split and save reports.",
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
        "--work",
        type=int,
        default=None,
        help="dataloader worker count; default is auto",
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
    pred_paths.add_argument("--img", type=Path, required=True, help="input image file")
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
            "Inspect source assets, dataset, checkpoint, class-count consistency, "
            "and runtime readiness before training or evaluation."
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
