SHELL := bash

.PHONY: glyph

VENV_DIR ?= ./.venv
VENV_PYTHON ?= $(VENV_DIR)/bin/python
VENV_STAMP ?= $(VENV_DIR)/.deps-ready

glyph:
	@set -euo pipefail; \
	if [[ ! -x "$(VENV_PYTHON)" ]]; then \
		echo "[glyph] creating .venv"; \
		rm -rf "$(VENV_DIR)"; \
		python3 -m venv "$(VENV_DIR)"; \
	fi; \
	requirements_sha="$$(python3 -c 'import hashlib, pathlib; print(hashlib.sha256(pathlib.Path("requirements.txt").read_bytes()).hexdigest())')"; \
	current_python_version="$$("$(VENV_PYTHON)" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"; \
	stored_requirements_sha=""; \
	stored_python_version=""; \
	if [[ -f "$(VENV_STAMP)" ]]; then \
		source "$(VENV_STAMP)"; \
		stored_requirements_sha="$${requirements_sha256:-}"; \
		stored_python_version="$${python_version:-}"; \
	fi; \
	if [[ "$$stored_requirements_sha" != "$$requirements_sha" || "$$stored_python_version" != "$$current_python_version" ]]; then \
		echo "[glyph] installing dependencies"; \
		PIP_CACHE_DIR="$(CURDIR)/.cache/pip" PIP_DISABLE_PIP_VERSION_CHECK=1 "$(VENV_PYTHON)" -m pip install -r requirements.txt; \
	else \
		echo "[glyph] dependencies are up to date"; \
	fi; \
	printf 'requirements_sha256=%s\npython_version=%s\n' "$$requirements_sha" "$$current_python_version" >"$(VENV_STAMP)"; \
	"$(VENV_PYTHON)" -m core.builder
