from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from core.console import print_log, print_summary


def _find_root() -> Path:
    env_root = os.environ.get("GLYPH_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if (root / "pyproject.toml").is_file() and (root / "core").is_dir():
            return root

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "core").is_dir():
            return candidate

    raise RuntimeError("Project root could not be resolved for glyph qa.")


ROOT_DIR = _find_root()


def _run_step(
    name: str,
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    print_log("qa", f"run {name}", tone="info")
    completed = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")
    print_log("qa", f"{name} ok", tone="success")


def _qa_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", str(ROOT_DIR / ".cache" / "pycache"))
    return env


def run_lint() -> None:
    _run_step("lint", [sys.executable, "-m", "ruff", "check", "."], env=_qa_env())


def run_syntax() -> None:
    _run_step(
        "syntax",
        [sys.executable, "-m", "compileall", "-q", "core", "tests"],
        env=_qa_env(),
    )


def run_smoke() -> None:
    _run_step("smoke", [sys.executable, "-m", "pytest", "-m", "smoke"], env=_qa_env())


def run_qa() -> None:
    print_summary(
        "QA Summary",
        [
            ("python", sys.executable),
            ("root", ROOT_DIR),
            ("steps", "lint, syntax, smoke"),
        ],
    )
    run_lint()
    run_syntax()
    run_smoke()
    print_log("qa", "all checks passed", tone="success")
