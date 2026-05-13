from __future__ import annotations

import warnings

from core.runtime import configure_runtime, runtime_warning_context


def test_configure_runtime_does_not_override_warning_dispatch() -> None:
    original_showwarning = warnings.showwarning
    configure_runtime()
    assert warnings.showwarning is original_showwarning


def test_runtime_warning_context_filters_known_noise() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with runtime_warning_context():
            warnings.warn("Matplotlib is building the font cache; this may take a moment.")
            warnings.warn("custom runtime warning")

    assert [str(item.message) for item in captured] == ["custom runtime warning"]
