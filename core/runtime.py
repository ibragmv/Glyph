from __future__ import annotations

import io
import importlib
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"


def _show_runtime_warning(
    message: warnings.WarningMessage | str,
    category,
    filename: str,
    lineno: int,
    file=None,
    line=None,
) -> None:
    from core.console import print_warning

    category_name = getattr(category, "__name__", "Warning")
    if category_name == "PyparsingDeprecationWarning":
        return
    print_warning(f"{category_name}: {message}")


def configure_runtime() -> None:
    matplotlib_cache = CACHE_DIR / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    warnings.showwarning = _show_runtime_warning
    warnings.filterwarnings(
        "ignore",
        message="Matplotlib is building the font cache; this may take a moment.",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Error fetching version info .*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"resource_tracker: There appear to be \d+ leaked semaphore objects.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*'oneOf' deprecated - use 'one_of'.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*'parseString' deprecated - use 'parse_string'.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*'resetCache' deprecated - use 'reset_cache'.*",
    )
    warnings.simplefilter("default")


def prepare_matplotlib() -> None:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        importlib.import_module("matplotlib")
        from matplotlib import font_manager

        font_manager.findSystemFonts()
