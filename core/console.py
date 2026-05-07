from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

PURPLE = "\033[38;2;111;94;255m"
PURPLE_SOFT = "\033[38;2;144;128;255m"
WHITE = "\033[38;2;245;247;255m"
SLATE = "\033[38;2;152;160;190m"
GREEN = "\033[38;2;76;225;160m"
CYAN = "\033[38;2;90;210;255m"
AMBER = "\033[38;2;255;180;90m"
RED = "\033[38;2;255;95;125m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _supports_progress() -> bool:
    return sys.stderr.isatty()


def _apply(text: object, *styles: str) -> str:
    text = str(text)
    if not _supports_color() or not styles:
        return text
    return f"{''.join(styles)}{text}{RESET}"


def accent(text: object) -> str:
    return _apply(text, PURPLE, BOLD)


def accent_soft(text: object) -> str:
    return _apply(text, PURPLE_SOFT)


def bright(text: object) -> str:
    return _apply(text, WHITE, BOLD)


def dim(text: object) -> str:
    return _apply(text, DIM, SLATE)


def ok(text: object) -> str:
    return _apply(text, GREEN, BOLD)


def info(text: object) -> str:
    return _apply(text, CYAN, BOLD)


def warn(text: object) -> str:
    return _apply(text, AMBER, BOLD)


def danger(text: object) -> str:
    return _apply(text, RED, BOLD)


def _rgb(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def _blend_channel(left: int, right: int, ratio: float) -> int:
    return int(left + ((right - left) * ratio))


def _blend_rgb(
    left: tuple[int, int, int], right: tuple[int, int, int], ratio: float
) -> tuple[int, int, int]:
    return (
        _blend_channel(left[0], right[0], ratio),
        _blend_channel(left[1], right[1], ratio),
        _blend_channel(left[2], right[2], ratio),
    )


def cosmic_orange_block(text: object) -> str:
    raw = str(text)
    if not _supports_color() or not raw:
        return raw

    lines = raw.splitlines()
    visible_columns = [
        col for line in lines for col, char in enumerate(line) if not char.isspace()
    ]
    if not visible_columns:
        return raw

    start = min(visible_columns)
    span = max(1, max(visible_columns) - start)
    palette = (
        (255, 120, 32),
        (255, 160, 56),
        (255, 196, 88),
        (255, 224, 140),
    )
    rendered: list[str] = []
    for line in lines:
        chunks: list[str] = []
        for col, char in enumerate(line):
            if char.isspace():
                chunks.append(char)
                continue
            ratio = (col - start) / span
            scaled = ratio * (len(palette) - 1)
            index = min(int(scaled), len(palette) - 2)
            local_ratio = scaled - index
            red, green, blue = _blend_rgb(
                palette[index], palette[index + 1], local_ratio
            )
            chunks.append(f"{BOLD}{_rgb(red, green, blue)}{char}{RESET}")
        rendered.append("".join(chunks))
    return "\n".join(rendered)


def path_text(value: object) -> str:
    return _apply(value, WHITE)


def channel_label(channel: str, tone: str = "info") -> str:
    if tone == "error":
        return danger("[error]")
    if tone == "warn":
        return warn("[warning]")
    if tone == "success":
        palette = {
            "gen": ok(channel),
            "train": ok(channel),
            "val": info(channel),
            "pred": ok(channel),
            "scan": ok(channel),
            "bench": ok(channel),
            "check": info(channel),
        }
        return palette.get(channel, ok(channel))

    palette = {
        "gen": accent(channel),
        "train": info(channel),
        "val": warn(channel),
        "pred": ok(channel),
        "scan": ok(channel),
        "bench": ok(channel),
        "check": info(channel),
    }
    return palette.get(channel, info(channel))


def plain(text: object) -> str:
    return _ANSI_RE.sub("", str(text))


def _terminal_width(default: int = 100) -> int:
    return shutil.get_terminal_size((default, 24)).columns


def _wrap_plain_text(text: object, width: int) -> list[str]:
    return textwrap.wrap(
        plain(text),
        width=max(16, width),
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]


def print_banner(command: str, subtitle: str | None = None) -> None:
    print(f"{accent('›')} {bright(command)}")
    if subtitle:
        print(f"  {dim(subtitle)}")


def print_log(
    channel: str,
    message: str,
    tone: str = "info",
    *,
    wrap: bool = False,
) -> None:
    prefix = channel_label(channel, tone=tone)
    if not wrap:
        print(f"{prefix} {message}")
        return

    prefix_plain = (
        "[error]"
        if tone == "error"
        else "[warning]"
        if tone == "warn"
        else f"{channel} "
    )
    lines = _wrap_plain_text(message, _terminal_width() - len(prefix_plain) - 1)
    print(f"{prefix} {lines[0]}")
    continuation = " " * (len(prefix_plain) + 1)
    for line in lines[1:]:
        print(f"{continuation}{line}")


def print_error(message: str) -> None:
    print_log("error", message, tone="error")


def print_warning(message: str) -> None:
    print_log("warning", message, tone="warn")


def print_summary(title: str, rows: list[tuple[str, object]]) -> None:
    print(bright(title))
    width = _terminal_width()
    for label, value in rows:
        label_text = str(label).ljust(12)
        prefix = f"  {dim(label_text)} "
        continuation = "  " + (" " * 13)
        lines = _wrap_plain_text(value, width - len(plain(prefix)) - 1)
        print(f"{prefix}{path_text(lines[0])}")
        for line in lines[1:]:
            print(f"{continuation}{path_text(line)}")


def print_rank(rank: int, label: str, confidence: str) -> None:
    rank_text = accent(f"{rank:02d}")
    label_text = bright(label)
    confidence_text = ok(confidence)
    print(f"{rank_text} {label_text} {dim('·')} {confidence_text}")


def display_path(path: Path | str) -> str:
    return path_text(Path(path))


def create_progress(
    iterable=None,
    *,
    command: str,
    scope: str,
    color: str,
    leave: bool = False,
    total: Optional[int] = None,
    position: int = 0,
):
    return tqdm(
        iterable,
        total=total,
        disable=not _supports_progress(),
        leave=leave,
        position=position,
        dynamic_ncols=True,
        colour=color,
        desc=f"› {command} {scope}",
        bar_format=(
            "{desc:<22} {bar:18} "
            "{percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed}<{remaining} | {postfix}"
        ),
    )


def write_progress_line(progress, line: str) -> None:
    if getattr(progress, "disable", False):
        print(line)
        return
    progress.write(line)
