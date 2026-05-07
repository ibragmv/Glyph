from __future__ import annotations

import shutil
import stat
import sys
from pathlib import Path
from textwrap import dedent


def _copy_package(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    shutil.copytree(source_dir, target_dir)


def _write_launcher(output_path: Path, python_executable: Path) -> None:
    launcher = dedent(
        f"""\
        #!{python_executable}
        from __future__ import annotations

        import sys
        from pathlib import Path

        APP_ROOT = Path(__file__).resolve().parent
        LIB_DIR = APP_ROOT / ".glyph" / "lib"

        if str(LIB_DIR) not in sys.path:
            sys.path.insert(0, str(LIB_DIR))

        from core.cli import main


        if __name__ == "__main__":
            main()
        """
    )
    output_path.write_text(launcher, encoding="utf-8")
    current_mode = output_path.stat().st_mode
    output_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def sync_launcher(
    output_path: Path,
    python_executable: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    resolved_project_root = (
        project_root or Path(__file__).resolve().parent.parent
    ).resolve()
    resolved_output_path = output_path.resolve()
    selected_python = (python_executable or Path(sys.executable)).expanduser()
    resolved_python = selected_python.resolve()

    if not resolved_python.is_file():
        raise FileNotFoundError(f"Python executable not found: {resolved_python}")

    runtime_dir = resolved_output_path.parent / ".glyph"
    lib_dir = runtime_dir / "lib"
    package_target_dir = lib_dir / "core"
    package_source_dir = resolved_project_root / "core"

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    _copy_package(package_source_dir, package_target_dir)
    _write_launcher(resolved_output_path, selected_python.absolute())
    return resolved_output_path


def main() -> None:
    from core.console import display_path, print_banner, print_log

    root_dir = Path(__file__).resolve().parent.parent
    print_banner("glyph", "launcher sync")
    built_path = sync_launcher(
        output_path=root_dir / "glyph",
        python_executable=Path(sys.executable),
        project_root=root_dir,
    )
    print_log("glyph", f"ready {display_path(built_path)}", tone="success")


if __name__ == "__main__":
    main()
