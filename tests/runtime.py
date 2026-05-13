from __future__ import annotations

import warnings

from core.runtime import configure_runtime


def test_runtime_warning_filters_ignore_known_noise(monkeypatch) -> None:
    shown: list[str] = []

    monkeypatch.setattr("core.console.print_warning", shown.append)
    configure_runtime()

    warnings.warn("Matplotlib is building the font cache; this may take a moment.")
    warnings.warn("custom runtime warning")

    assert shown == ["UserWarning: custom runtime warning"]
