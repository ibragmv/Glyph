#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
STAMP_FILE="$VENV_DIR/.requirements.sha256"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "[startup] project: $ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[startup] creating virtual environment"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[startup] using existing virtual environment"
fi

REQ_HASH="$(shasum -a 256 "$ROOT_DIR/requirements.txt" | awk '{print $1}')"

if [[ -f "$STAMP_FILE" ]] && [[ "$(cat "$STAMP_FILE")" == "$REQ_HASH" ]]; then
  echo "[startup] dependencies are up to date"
else
  echo "[startup] installing dependencies"
  "$VENV_DIR/bin/python" -m pip install --quiet -r "$ROOT_DIR/requirements.txt"
  printf '%s\n' "$REQ_HASH" > "$STAMP_FILE"
fi

echo "[startup] environment is ready"
