#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
STAMP_FILE="$VENV_DIR/.startup.stamp"
DEFAULT_FONT="$ROOT_DIR/fonts/NotoSansImperialAramaic-Regular.ttf"

RECREATE_VENV=0
FORCE_INSTALL=0
CHECK_ONLY=0
VENV_RECREATED=0
UPGRADE_PACKAGING=0

export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1


log() {
  printf '[startup] %s\n' "$1"
}


warn() {
  printf '[warning] %s\n' "$1"
}


die() {
  printf '[error] %s\n' "$1" >&2
  exit 1
}


usage() {
  cat <<EOF
Usage: ./scripts/startup.sh [options]

Options:
  --recreate           Remove and recreate .venv before installing dependencies
  --force-install      Reinstall dependencies even if the environment stamp matches
  --upgrade-packaging  Upgrade pip, setuptools, and wheel before install
  --check              Validate the environment and exit without installing anything
  -h, --help           Show this help message
EOF
}


parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --recreate)
        RECREATE_VENV=1
        shift
        ;;
      --force-install)
        FORCE_INSTALL=1
        shift
        ;;
      --upgrade-packaging)
        UPGRADE_PACKAGING=1
        shift
        ;;
      --check)
        CHECK_ONLY=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
}


ensure_bootstrap_python() {
  command -v python3 >/dev/null 2>&1 || die "python3 not found in PATH"
}


file_sha256() {
  python3 - "$1" <<'PY'
from __future__ import annotations

import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print(digest)
PY
}


python_version() {
  "$1" - <<'PY'
from __future__ import annotations

import sys

print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
}


write_stamp() {
  local requirements_sha="$1"
  local venv_python="$2"
  local venv_python_version="$3"

  cat >"$STAMP_FILE" <<EOF
requirements_sha256=$requirements_sha
venv_python=$venv_python
venv_python_version=$venv_python_version
EOF
}


load_stamp() {
  [[ -f "$STAMP_FILE" ]] || return 1
  # shellcheck disable=SC1090
  source "$STAMP_FILE"
}


recreate_venv() {
  if [[ -e "$VENV_DIR" ]]; then
    log "removing existing virtual environment"
    rm -rf "$VENV_DIR"
  fi
  log "creating virtual environment with python3"
  python3 -m venv "$VENV_DIR"
  VENV_RECREATED=1
}


ensure_venv() {
  if (( RECREATE_VENV )); then
    recreate_venv
    return
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    recreate_venv
    return
  fi

  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    warn "virtual environment is incomplete; recreating .venv"
    recreate_venv
    return
  fi

  log "using existing virtual environment"
}


needs_reinstall() {
  local requirements_sha="$1"
  local venv_python="$2"
  local venv_python_version="$3"

  if (( FORCE_INSTALL )); then
    log "forcing dependency installation"
    return 0
  fi

  if [[ ! -f "$STAMP_FILE" ]]; then
    log "dependency stamp not found"
    return 0
  fi

  load_stamp || return 0

  if [[ "${requirements_sha256:-}" != "$requirements_sha" ]]; then
    log "requirements.txt changed"
    return 0
  fi

  if [[ "${venv_python:-}" != "$venv_python" ]]; then
    log "virtualenv interpreter changed"
    return 0
  fi

  if [[ "${venv_python_version:-}" != "$venv_python_version" ]]; then
    log "virtualenv Python version changed"
    return 0
  fi

  return 1
}


upgrade_packaging_tools() {
  log "upgrading pip tooling"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
}


install_dependencies() {
  log "installing dependencies from requirements.txt"
  "$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"
}


validate_environment() {
  log "running environment validation"
  "$VENV_DIR/bin/python" -m pip --version >/dev/null
  "$VENV_DIR/bin/python" -m pip check >/dev/null

  if [[ ! -f "$DEFAULT_FONT" ]]; then
    warn "default Imperial Aramaic font not found at $DEFAULT_FONT"
  fi
}


print_next_steps() {
  if (( VENV_RECREATED )); then
    log "virtual environment was recreated; rebuild the launcher with: ./.venv/bin/python -m core build"
    return
  fi

  log "next step: ./.venv/bin/python -m core build"
}


main() {
  parse_args "$@"
  ensure_bootstrap_python

  [[ -f "$ROOT_DIR/requirements.txt" ]] || die "requirements.txt not found in project root"

  log "project: $ROOT_DIR"

  ensure_venv

  local venv_python="$VENV_DIR/bin/python"
  local requirements_sha
  local venv_python_version
  requirements_sha="$(file_sha256 "$ROOT_DIR/requirements.txt")"
  venv_python_version="$(python_version "$venv_python")"

  if (( CHECK_ONLY )); then
    validate_environment
    log "environment check passed"
    print_next_steps
    return
  fi

  if needs_reinstall "$requirements_sha" "$venv_python" "$venv_python_version"; then
    if (( UPGRADE_PACKAGING )); then
      upgrade_packaging_tools
    fi
    install_dependencies
    validate_environment
    write_stamp "$requirements_sha" "$venv_python" "$venv_python_version"
  else
    log "dependencies are up to date"
  fi

  log "environment is ready"
  print_next_steps
}


main "$@"
