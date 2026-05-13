from __future__ import annotations

import stat
import sys
from pathlib import Path
from textwrap import dedent


def _write_launcher(output_path: Path, python_executable: Path) -> None:
    launcher = dedent(
        f"""\
        #!{python_executable}
        from __future__ import annotations

        import sys
        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parent

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

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
) -> Path:
    resolved_output_path = output_path.resolve()
    selected_python = (python_executable or Path(sys.executable)).expanduser()
    resolved_python = selected_python.resolve()

    if not resolved_python.is_file():
        raise FileNotFoundError(f"Python executable not found: {resolved_python}")

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_launcher(resolved_output_path, selected_python.absolute())
    return resolved_output_path


def main() -> None:
    from core.console import display_path, print_banner, print_log

    root_dir = Path(__file__).resolve().parent.parent
    print_banner("glyph", "launcher sync")
    built_path = sync_launcher(
        output_path=root_dir / "glyph",
        python_executable=Path(sys.executable),
    )
    print_log("glyph", f"ready {display_path(built_path)}", tone="success")


if __name__ == "__main__":
    main()
