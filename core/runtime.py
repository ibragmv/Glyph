from __future__ import annotations

import io
import importlib
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
_IGNORED_WARNING_PATTERNS = (
    "Matplotlib is building the font cache; this may take a moment.",
    r"Error fetching version info .*",
    r"resource_tracker: There appear to be \d+ leaked semaphore objects.*",
    r".*'oneOf' deprecated - use 'one_of'.*",
    r".*'parseString' deprecated - use 'parse_string'.*",
    r".*'resetCache' deprecated - use 'reset_cache'.*",
)


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
    for pattern in _IGNORED_WARNING_PATTERNS:
        warnings.filterwarnings("ignore", message=pattern)
    warnings.simplefilter("default")


def prepare_matplotlib() -> None:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        importlib.import_module("matplotlib")
        from matplotlib import font_manager

        font_manager.findSystemFonts()
