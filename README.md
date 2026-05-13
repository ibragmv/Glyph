# Glyph

`Glyph` is a CLI for classifying printed Aramaic letters from RGB images.

The project is built around **real crops and canonical exemplars**.

## What It Does

- builds RGB dataset splits from exemplar and real letter images
- keeps real crops in `r_train` and `r_val` with light preprocessing only
- trains a classifier against a real validation split
- validates checkpoints
- predicts one image
- scans image folders or runs labeled benchmarks

## Quick Start

```bash
make glyph
./glyph gen
./glyph train
./glyph val --pt artifacts/best_model.pt
```

Predict one image:

```bash
./glyph pred --pt artifacts/best_model.pt --img path/to/image.png
```

## Data Model

`Glyph` uses four dataset splits:

- `train/` — generated RGB training images
- `val/` — generated RGB validation images
- `r_train/` — real cropped letters for training support
- `r_val/` — real cropped letters for primary validation

When `r_val/` exists, it is treated as the main validation split.

## Source Assets

Project assets live in `source/`:

- `source/alphabet/` — reference glyph images
- `source/exemplars/` — exemplar letter variants
- `source/real/` — real cropped letters
- `source/textures/` — texture backgrounds for RGB synthetic generation

## Main Commands

```bash
./glyph gen
./glyph train
./glyph val --pt artifacts/best_model.pt
./glyph pred --pt artifacts/best_model.pt --img path/to/image.png
./glyph scan --pt artifacts/best_model.pt path/to/folder
./glyph bench --pt artifacts/best_model.pt --csv labels.csv path/to/folder
./glyph check
./glyph qa
```

## Useful Short Runs

Small dataset run:

```bash
./glyph gen --train 50 --val 10
```

External folder benchmark:

```bash
./glyph bench --pt artifacts/best_model.pt --csv labels.csv path/to/folder
```

## Output

- `dataset/` — generated synthetic and real split data
- `artifacts/best_model.pt` — best checkpoint
- `artifacts/last_model.pt` — last checkpoint
- `artifacts/val/` — validation reports
- `artifacts/scans/` — folder prediction runs
- `artifacts/benchmarks/` — benchmark runs

## Quality Checks

```bash
./glyph lint
./glyph syntax
./glyph smoke
./glyph qa
```

`glyph smoke` runs the smoke marker only.
`glyph qa` runs lint, syntax, and the full pytest suite.

For development and local verification, `make glyph` is the only bootstrap step:

```bash
make glyph
python -m pytest -q
python -m core qa
```

## License

MIT
