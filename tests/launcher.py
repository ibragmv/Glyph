from __future__ import annotations

from pathlib import Path

from core.launcher import sync_launcher


def test_sync_launcher_avoids_shadow_copy_runtime(tmp_path: Path) -> None:
    launcher_path = sync_launcher(
        output_path=tmp_path / "glyph",
        python_executable=Path("/usr/bin/python3"),
    )

    payload = launcher_path.read_text(encoding="utf-8")

    assert "PROJECT_ROOT" in payload
    assert ".glyph" not in payload
    assert "sys.path.insert(0, str(PROJECT_ROOT))" in payload
