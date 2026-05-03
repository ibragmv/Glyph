from __future__ import annotations

import io
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"


def configure_runtime() -> None:
    matplotlib_cache = CACHE_DIR / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    warnings.filterwarnings(
        "ignore",
        message="Matplotlib is building the font cache; this may take a moment.",
    )


def prepare_matplotlib() -> None:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        import matplotlib  # noqa: F401
        from matplotlib import font_manager

        font_manager.findSystemFonts()
