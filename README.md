# Glyph

`Glyph` is a CLI for classifying printed Imperial Aramaic letters from images.

It covers the full workflow:

- prepare the local environment
- generate a synthetic dataset
- train a classifier
- validate a checkpoint
- predict one image
- scan or benchmark folders of images

## Quick Start

Use one command to prepare the environment and build the local launcher:

```bash
make glyph
```

Then run the default workflow:

```bash
./glyph gen
./glyph train
./glyph val --pt artifacts/best_model.pt
```

Predict one image:

```bash
./glyph pred --pt artifacts/best_model.pt --img path/to/image.png
```

## How It Works

There are only two layers you need to think about.

### `make glyph`

This is the only `make` entry point.

It:

- creates or refreshes `.venv`
- installs or refreshes dependencies
- validates the environment
- builds the local `./glyph` launcher

It is safe to run again when you want to refresh the local setup.

### `./glyph ...`

This is the product CLI.

Use it after `make glyph` exists and the launcher has been built.

## First Run

For a clean local setup:

```bash
make glyph
./glyph --help
```

That is the standard happy path.

## Common Tasks

### Generate a dataset

```bash
./glyph gen
```

Smaller test run:

```bash
./glyph gen --train 50 --val 10
```

Default output:

- `dataset/train/`
- `dataset/val/`
- `dataset/metadata.json`

### Train a model

```bash
./glyph train
```

Default output:

- `artifacts/best_model.pt`
- `artifacts/last_model.pt`
- training reports in `artifacts/`

### Validate a checkpoint

```bash
./glyph val --pt artifacts/best_model.pt
```

Default output:

- `artifacts/val/`

### Predict one image

```bash
./glyph pred --pt artifacts/best_model.pt --img path/to/image.png
```

This prints ranked predictions with confidence.

### Scan a folder

```bash
./glyph scan --pt artifacts/best_model.pt path/to/folder
```

This saves a new run under `artifacts/scans/`.

### Benchmark an external set

```bash
./glyph bench --pt artifacts/best_model.pt --csv labels.csv path/to/folder
```

Use this when you have external images and want metrics.

Expected CSV shape:

```csv
file,true_label
sample_01.png,aleph
sample_02.png,beth
```

### Check project readiness

```bash
./glyph check
```

This verifies:

- runtime imports
- dataset structure
- checkpoint readability
- font discovery
- class consistency

### Run fast QA

```bash
./glyph qa
```

This runs:

- `lint`
- `syntax`
- `smoke`

## Command Reference

Show root help:

```bash
./glyph --help
```

Show help for one command:

```bash
./glyph train --help
```

Main commands:

- `gen` — generate dataset images
- `train` — train a classifier
- `val` — validate a checkpoint
- `pred` — predict one image
- `scan` — process a folder of images
- `bench` — benchmark a folder, with or without labels
- `check` — verify local readiness
- `qa` — run fast project checks

## Project Layout

- `fonts/` — compatible fonts used for generation
- `dataset/` — generated training and validation data
- `artifacts/` — checkpoints, reports, scans, benchmarks
- `.venv/` — local Python environment
- `.glyph/` — packaged launcher runtime

## If Something Fails

If the local setup or launcher looks broken:

```bash
rm -rf .venv .glyph glyph
make glyph
```

If `artifacts/best_model.pt` does not exist yet:

```bash
./glyph gen
./glyph train
./glyph val --pt artifacts/best_model.pt
```

If you want the fastest sanity check:

```bash
./glyph check
./glyph qa
```

## License

Under MIT License
