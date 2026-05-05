# Glyph

`Glyph` is a CLI tool for classifying printed Imperial Aramaic letters.

It can:

- generate a synthetic dataset from fonts
- train a classifier
- validate a checkpoint
- predict one image
- process a folder of external images

## What This Project Is

This project is built around one task: take an image of a printed Imperial Aramaic letter and predict which letter it is.

Typical workflow:

1. generate a dataset
2. train the model
3. validate the checkpoint
4. test on your own images

Main command:

```bash
./glyph
```

## Quick Start

If you are running the project for the first time:

```bash
./scripts/startup.sh
./.venv/bin/python -m core build
./glyph gen
./glyph train
./glyph val --pt artifacts/best_model.pt
```

After that you can test a single image:

```bash
./glyph pred --pt artifacts/best_model.pt --img path/to/image.png
```

## The 5 Commands You Actually Need

### 1. Prepare the environment

```bash
./scripts/startup.sh
```

This creates or refreshes `.venv/`.

If you want to quickly verify the environment without reinstalling anything:

```bash
./scripts/startup.sh --check
```

### 2. Build the launcher

```bash
./.venv/bin/python -m core build
```

This creates the local `./glyph` command.

### 3. Generate data

```bash
./glyph gen
```

By default this creates:

- `dataset/train/`
- `dataset/val/`
- `dataset/metadata.json`

If you want a smaller test run:

```bash
./glyph gen --train 50 --val 10
```

### 4. Train the model

```bash
./glyph train
```

By default training reads from `dataset/` and writes to `artifacts/`.

The main output files are:

- `artifacts/best_model.pt`
- `artifacts/last_model.pt`

### 5. Validate the checkpoint

```bash
./glyph val --pt artifacts/best_model.pt
```

This evaluates the model on `dataset/val` and writes reports to `artifacts/val/`.

## Test On Your Own Images

### One image

```bash
./glyph pred --pt artifacts/best_model.pt --img path/to/image.png
```

This prints the top predictions and confidence scores.

### A folder of images

```bash
./glyph scan --pt artifacts/best_model.pt path/to/folder
```

This creates a run under `artifacts/scans/` and saves predictions for the whole folder.

### A labeled external test set

```bash
./glyph bench --pt artifacts/best_model.pt --csv labels.csv path/to/folder
```

Use this when you have external images and want real benchmark metrics.

Expected CSV format:

```csv
file,true_label
sample_01.png,aleph
sample_02.png,beth
```

## Before You Run Large Jobs

Use:

```bash
./glyph check
```

This checks:

- runtime
- dataset structure
- checkpoint availability
- class consistency
- fonts

## Where Things Go

- `dataset/` — generated training and validation images
- `artifacts/` — checkpoints and reports
- `fonts/` — compatible fonts
- `.venv/` — local Python environment
- `.glyph/` — packaged launcher runtime

## Helpful Commands

Show help:

```bash
./glyph --help
```

Show help for one command:

```bash
./glyph train --help
```

Run fast project checks:

```bash
./glyph qa
```

## If Something Fails

If `./glyph` is missing:

```bash
./.venv/bin/python -m core build
```

If the environment looks broken:

```bash
./scripts/startup.sh --recreate
./.venv/bin/python -m core build
```

If `best_model.pt` does not exist yet, that is normal. You need to run:

```bash
./glyph gen
./glyph train
./glyph val --pt artifacts/best_model.pt
```

## License

MIT
