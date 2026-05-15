from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from core.console import create_progress, print_log, print_summary, write_progress_line
from core.utils import ensure_dir, format_float


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
QA_CACHE_PATH = ROOT_DIR / ".cache" / "qa_timings.json"
QA_STEP_ORDER = ("lint", "syntax", "tests")
QA_DEFAULT_STEP_SECONDS = {
    "lint": 1.0,
    "syntax": 1.0,
    "tests": 15.0,
}
_PYTEST_PROGRESS_TOKENS = frozenset(".FsxX")


def _load_step_estimates(
    *,
    cache_path: Path,
    step_order: tuple[str, ...],
    default_seconds: dict[str, float],
) -> dict[str, float]:
    estimates = dict(default_seconds)
    if not cache_path.is_file():
        return estimates

    with cache_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if not isinstance(payload, dict):
        return estimates

    step_seconds = payload.get("step_seconds", {})
    if not isinstance(step_seconds, dict):
        return estimates

    for step_name in step_order:
        value = step_seconds.get(step_name)
        if isinstance(value, (int, float)) and value > 0:
            estimates[step_name] = float(value)
    return estimates


def _save_step_estimates(
    *,
    cache_path: Path,
    step_order: tuple[str, ...],
    step_seconds: dict[str, float],
) -> None:
    ensure_dir(cache_path.parent)
    payload = {
        "step_seconds": {
            step_name: format_float(step_seconds[step_name], digits=3)
            for step_name in step_order
        }
    }
    with cache_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def _estimate_total_runtime(progress) -> float:
    step_order = tuple(getattr(progress, "_glyph_step_order", QA_STEP_ORDER))
    step_estimates = getattr(progress, "_glyph_step_estimates", {})
    completed_actual = getattr(progress, "_glyph_completed_actual", {})
    current_step = getattr(progress, "_glyph_current_step", None)
    current_started_at = getattr(progress, "_glyph_current_step_started_at", None)
    pytest_total = int(getattr(progress, "_glyph_pytest_total", 0) or 0)
    pytest_completed = int(getattr(progress, "_glyph_pytest_completed", 0) or 0)

    total = 0.0
    for step_name in step_order:
        if step_name in completed_actual:
            total += completed_actual[step_name]
            continue
        if step_name == current_step and current_started_at is not None:
            current_elapsed = max(0.0, time.monotonic() - current_started_at)
            if step_name in {"tests", "smoke"} and pytest_completed > 0 and pytest_total > 0:
                total += max(current_elapsed, current_elapsed * (pytest_total / pytest_completed))
                continue
            total += max(current_elapsed, float(step_estimates.get(step_name, 0.0)))
            continue
        total += float(step_estimates.get(step_name, 0.0))
    return max(total, 0.0)


def _set_progress_timing(progress, message: str) -> None:
    progress.set_postfix_str(message, refresh=False)


def _collect_pytest_count(*, marker: str | None = None) -> int:
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    if marker is not None:
        command.extend(["-m", marker])

    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        env=_qa_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"pytest collect failed with exit code {completed.returncode}")

    total = 0
    for line in completed.stdout.splitlines():
        match = re.search(r":\s*(\d+)\s*$", line)
        if match:
            total += int(match.group(1))
    return total


def _run_pytest_progress(
    *,
    command_name: str,
    marker: str | None = None,
    progress=None,
    channel: str | None = None,
) -> None:
    total_tests = _collect_pytest_count(marker=marker)
    owns_progress = progress is None
    active_channel = channel or command_name
    if progress is None:
        progress = create_progress(
            command=command_name,
            scope="run",
            color="green",
            leave=True,
            total=max(total_tests, 1),
        )
    progress._glyph_pytest_total = total_tests
    progress._glyph_pytest_completed = 0
    progress._glyph_current_step = command_name
    progress._glyph_current_step_started_at = time.monotonic()
    progress.set_postfix_str(f"collect {total_tests} tests", refresh=False)
    progress.refresh()
    if owns_progress:
        write_progress_line(progress, f"{active_channel} run {command_name}")

    command = [sys.executable, "-m", "pytest", "-q"]
    if marker is not None:
        command.extend(["-m", marker])

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            env=_qa_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            bufsize=1,
        )

        summary_lines: list[str] = []
        try:
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(1)
                if chunk == "":
                    break
                if chunk in _PYTEST_PROGRESS_TOKENS and progress._glyph_pytest_completed < total_tests:
                    progress._glyph_pytest_completed += 1
                    progress.update(1)
                    progress.set_postfix_str(
                        f"tests {progress._glyph_pytest_completed}/{total_tests}",
                        refresh=False,
                    )
                    continue
                if chunk in ("\r", "\n"):
                    continue
                summary_line = chunk + process.stdout.readline()
                stripped = summary_line.strip()
                if stripped:
                    summary_lines.append(stripped)

            returncode = process.wait()
            if returncode != 0:
                progress.close()
                stderr_file.seek(0)
                stderr = stderr_file.read().strip()
                for line in summary_lines:
                    print(line)
                if stderr:
                    print(stderr, file=sys.stderr)
                raise SystemExit(
                    f"{active_channel} failed with exit code {returncode}"
                )

            if owns_progress:
                progress.n = total_tests
            progress._glyph_pytest_completed = total_tests
            progress.set_postfix_str(
                f"tests {total_tests}/{total_tests}",
                refresh=False,
            )
            progress.refresh()
        finally:
            if process.stdout is not None:
                process.stdout.close()
            progress._glyph_current_step = None
            progress._glyph_current_step_started_at = None
            if owns_progress:
                progress.close()

    if owns_progress:
        write_progress_line(progress, f"{active_channel} ok")


def _run_step(
    name: str,
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    progress=None,
    channel: str = "qa",
) -> None:
    if progress is None:
        print_log(channel, f"run {name}", tone="info")
        completed = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(f"{name} failed with exit code {completed.returncode}")
        print_log(channel, f"{name} ok", tone="success")
        return

    progress._glyph_current_step = name
    progress._glyph_current_step_started_at = time.monotonic()
    _set_progress_timing(progress, f"run {name}")
    progress.refresh()
    write_progress_line(progress, f"{channel} run {name}")

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(
        mode="w+",
        encoding="utf-8",
    ) as stderr_file:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )

        while True:
            returncode = process.poll()
            _set_progress_timing(progress, f"run {name}")
            progress.refresh()
            if returncode is not None:
                break
            time.sleep(0.1)

        if returncode != 0:
            progress.close()
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read().strip()
            stderr = stderr_file.read().strip()
            if stdout:
                print(stdout)
            if stderr:
                print(stderr, file=sys.stderr)
            raise SystemExit(f"{name} failed with exit code {returncode}")

    step_elapsed = max(
        0.0,
        time.monotonic()
        - getattr(progress, "_glyph_current_step_started_at", time.monotonic()),
    )
    progress._glyph_completed_actual[name] = step_elapsed
    progress.update(1)
    progress._glyph_current_step = None
    progress._glyph_current_step_started_at = None
    _set_progress_timing(progress, f"{name} ok")
    write_progress_line(progress, f"{channel} {name} ok")


def _qa_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", str(ROOT_DIR / ".cache" / "pycache"))
    return env


def run_lint(*, progress=None) -> None:
    _run_step(
        "lint",
        [sys.executable, "-m", "ruff", "check", "."],
        env=_qa_env(),
        progress=progress,
        channel="qa",
    )


def run_syntax(*, progress=None) -> None:
    _run_step(
        "syntax",
        [sys.executable, "-m", "compileall", "-q", "core", "tests"],
        env=_qa_env(),
        progress=progress,
        channel="qa",
    )


def run_smoke(*, progress=None) -> None:
    if progress is None:
        _run_pytest_progress(command_name="smoke", marker="smoke", channel="smoke")
        return

    _run_step(
        "smoke",
        [sys.executable, "-m", "pytest", "-m", "smoke"],
        env=_qa_env(),
        progress=progress,
        channel="smoke",
    )


def run_tests(*, progress=None) -> None:
    if progress is None:
        _run_pytest_progress(command_name="tests", channel="tests")
        return

    _set_progress_timing(progress, "run tests")
    progress.refresh()
    write_progress_line(progress, "qa run tests")

    started_at = time.monotonic()
    _run_pytest_progress(command_name="tests", progress=progress, channel="qa")
    progress._glyph_completed_actual["tests"] = max(0.0, time.monotonic() - started_at)
    _set_progress_timing(progress, "tests ok")
    write_progress_line(progress, "qa tests ok")


def run_qa() -> None:
    total_tests = _collect_pytest_count()
    print_summary(
        "QA Summary",
        [
            ("python", sys.executable),
            ("root", ROOT_DIR),
            ("steps", f"lint, syntax, tests ({total_tests})"),
        ],
    )
    progress = create_progress(
        command="qa",
        scope="run",
        color="green",
        leave=True,
        total=2 + total_tests,
    )
    progress._glyph_step_estimates = _load_step_estimates(
        cache_path=QA_CACHE_PATH,
        step_order=QA_STEP_ORDER,
        default_seconds=QA_DEFAULT_STEP_SECONDS,
    )
    progress._glyph_step_order = QA_STEP_ORDER
    progress._glyph_completed_actual = {}
    progress._glyph_current_step = None
    progress._glyph_current_step_started_at = None
    progress._glyph_pytest_total = total_tests
    progress._glyph_pytest_completed = 0
    progress._glyph_estimate_total_fn = _estimate_total_runtime
    _set_progress_timing(progress, "run lint")
    try:
        run_lint(progress=progress)
        run_syntax(progress=progress)
        run_tests(progress=progress)
        _set_progress_timing(progress, "all checks passed")
        progress.refresh()
    finally:
        progress.close()
    _save_step_estimates(
        cache_path=QA_CACHE_PATH,
        step_order=QA_STEP_ORDER,
        step_seconds={
            step_name: progress._glyph_completed_actual.get(
                step_name,
                progress._glyph_step_estimates[step_name],
            )
            for step_name in QA_STEP_ORDER
        },
    )
    print_log("qa", "all checks passed", tone="success")
