from __future__ import annotations

import io
import importlib
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import torch


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


@dataclass(frozen=True)
class RuntimeDevice:
    device: torch.device
    type: str
    label: str
    reason: str
    cuda_built: bool
    cuda_available: bool
    cuda_version: str | None
    cuda_device_count: int
    cuda_name: str | None
    mps_built: bool
    mps_available: bool

    @property
    def pin_memory(self) -> bool:
        return self.type == "cuda"

    @property
    def amp_enabled(self) -> bool:
        return self.type == "cuda"

    @property
    def supports_channels_last(self) -> bool:
        return self.type in {"cuda", "mps"}


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
    torch.set_float32_matmul_precision("high")


def prepare_matplotlib() -> None:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        importlib.import_module("matplotlib")
        from matplotlib import font_manager

        font_manager.findSystemFonts()


def resolve_runtime_device() -> RuntimeDevice:
    cuda_built = torch.version.cuda is not None
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_name = torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else None

    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend and mps_backend.is_built())
    mps_available = bool(mps_backend and mps_backend.is_available())

    if cuda_available:
        label = "cuda" if cuda_name is None else f"cuda ({cuda_name})"
        return RuntimeDevice(
            device=torch.device("cuda"),
            type="cuda",
            label=label,
            reason="CUDA is available and selected.",
            cuda_built=cuda_built,
            cuda_available=cuda_available,
            cuda_version=torch.version.cuda,
            cuda_device_count=cuda_device_count,
            cuda_name=cuda_name,
            mps_built=mps_built,
            mps_available=mps_available,
        )

    if mps_available:
        return RuntimeDevice(
            device=torch.device("mps"),
            type="mps",
            label="mps (Apple Metal)",
            reason="Apple Metal Performance Shaders is available and selected.",
            cuda_built=cuda_built,
            cuda_available=cuda_available,
            cuda_version=torch.version.cuda,
            cuda_device_count=cuda_device_count,
            cuda_name=cuda_name,
            mps_built=mps_built,
            mps_available=mps_available,
        )

    if cuda_built:
        reason = "Torch was built with CUDA support, but no CUDA device is available."
    elif mps_built and not mps_available:
        reason = "Torch has MPS support, but Apple Metal is unavailable on this host."
    else:
        reason = "No GPU accelerator is available; using CPU."

    return RuntimeDevice(
        device=torch.device("cpu"),
        type="cpu",
        label="cpu",
        reason=reason,
        cuda_built=cuda_built,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        cuda_device_count=cuda_device_count,
        cuda_name=cuda_name,
        mps_built=mps_built,
        mps_available=mps_available,
    )


def resolve_num_workers(
    num_workers: int | None,
    *,
    min_auto_workers: int = 2,
    max_auto_workers: int = 8,
) -> int:
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        return num_workers

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 0
    return min(max_auto_workers, max(min_auto_workers, cpu_count // 2))


def optimize_model_for_device(
    model: torch.nn.Module,
    runtime_device: RuntimeDevice,
) -> torch.nn.Module:
    if runtime_device.supports_channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def prepare_image_batch(
    images: torch.Tensor,
    runtime_device: RuntimeDevice,
) -> torch.Tensor:
    if runtime_device.supports_channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images.to(runtime_device.device, non_blocking=runtime_device.pin_memory)
