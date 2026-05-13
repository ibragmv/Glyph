from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
DEPENDENCY_MANIFESTS: tuple[Path, ...] = (PROJECT_ROOT / "pyproject.toml",)


def _load_toml_module():
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:
        pass

    try:
        import tomli

        return tomli
    except ModuleNotFoundError:
        pass

    from pip._vendor import tomli

    return tomli


def _read_pyproject() -> dict[str, Any]:
    toml = _load_toml_module()
    with PYPROJECT_PATH.open("rb") as fp:
        return toml.load(fp)


def bootstrap_dependencies(*, include_dev: bool = True) -> tuple[str, ...]:
    payload = _read_pyproject()
    project = payload.get("project", {})
    runtime_dependencies = tuple(project.get("dependencies", []))
    optional_dependencies = project.get("optional-dependencies", {})
    dev_dependencies = tuple(optional_dependencies.get("dev", []))
    return runtime_dependencies + (dev_dependencies if include_dev else ())


def dependency_manifest_hash(paths: Iterable[Path] | None = None) -> str:
    manifest_paths = tuple(DEPENDENCY_MANIFESTS if paths is None else paths)
    payload = bytearray()
    for path in manifest_paths:
        resolved_path = Path(path).resolve()
        payload.extend(str(resolved_path).encode("utf-8"))
        payload.extend(b"\0")
        payload.extend(resolved_path.read_bytes())
        payload.extend(b"\0")
    return hashlib.sha256(payload).hexdigest()
