from __future__ import annotations

import sys
from pathlib import Path

def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from core.builder import build_binary

    python_path = root_dir / ".venv" / "bin" / "python"
    output_path = root_dir / "glyph"
    built_path = build_binary(
        output_path=output_path,
        python_executable=python_path,
        project_root=root_dir,
    )
    print(f"[build] ready: {built_path}")


if __name__ == "__main__":
    main()
