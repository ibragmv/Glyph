.PHONY: glyph startup clean clean-artifacts clean-dataset clean-generated

startup:
	@./scripts/startup.sh

glyph: startup
	@./.venv/bin/python ./scripts/build.py

clean:
	rm -rf .cache .glyph glyph

clean-artifacts:
	rm -rf artifacts

clean-dataset:
	rm -rf dataset

clean-generated: clean clean-artifacts clean-dataset
