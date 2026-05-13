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
	manifest_sha="$$(python3 -c 'from core.bootstrap import dependency_manifest_hash; print(dependency_manifest_hash())')"; \
	deps="$$(python3 -c 'from core.bootstrap import bootstrap_dependencies; print("\n".join(bootstrap_dependencies()))')"; \
	current_python_version="$$("$(VENV_PYTHON)" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"; \
	stored_manifest_sha=""; \
	stored_python_version=""; \
	if [[ -f "$(VENV_STAMP)" ]]; then \
		source "$(VENV_STAMP)"; \
		stored_manifest_sha="$${manifest_sha256:-}"; \
		stored_python_version="$${python_version:-}"; \
	fi; \
	if [[ "$$stored_manifest_sha" != "$$manifest_sha" || "$$stored_python_version" != "$$current_python_version" ]]; then \
		echo "[glyph] installing dependencies"; \
		while IFS= read -r dep; do \
			[[ -n "$$dep" ]] || continue; \
			PIP_CACHE_DIR="$(CURDIR)/.cache/pip" PIP_DISABLE_PIP_VERSION_CHECK=1 "$(VENV_PYTHON)" -m pip install "$$dep"; \
		done <<< "$$deps"; \
	else \
		echo "[glyph] dependencies are up to date"; \
	fi; \
	printf 'manifest_sha256=%s\npython_version=%s\n' "$$manifest_sha" "$$current_python_version" >"$(VENV_STAMP)"; \
	"$(VENV_PYTHON)" -m core.launcher
