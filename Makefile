.PHONY: glyph startup lint syntax smoke qa clean cleanartifacts cleandataset cleangenerated

PYTHON ?= python3
PYTHONPYCACHEPREFIX ?= $(CURDIR)/.cache/pycache

startup:
	@./scripts/startup.sh

glyph: startup
	@./.venv/bin/python -m core build

lint:
	@$(PYTHON) -m ruff check .

syntax:
	@PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) -m compileall -q core scripts tests

smoke:
	@PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) -m pytest -m smoke

qa:
	@$(PYTHON) -m core qa

clean:
	rm -rf .cache .glyph glyph

cleanartifacts:
	rm -rf artifacts

cleandataset:
	rm -rf dataset

cleangenerated: clean cleanartifacts cleandataset
