# Glyph

`Glyph` is a CLI for classifying printed Aramaic letters from real images.

The project is built around **real crops and canonical exemplars**.

## What It Does

- builds a synthetic dataset from canonical letter exemplars
- copies real crops into `realtrain` and `realval`
- trains a classifier against a real validation split
- validates checkpoints
- predicts one image
- scans or benchmarks image folders

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

- `train/` — synthetic training images
- `val/` — synthetic validation images
- `realtrain/` — real cropped letters for training support
- `realval/` — real cropped letters for primary validation

When `realval/` exists, it is treated as the main validation split.

## Source Assets

Project assets live in `source/`:

- `source/alphabet/` — canonical reference images
- `source/exemplars/` — exemplar letter variants
- `source/real/` — real cropped letters
- `source/textures/` — texture backgrounds for synthetic generation

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

Small dataset build:

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

## License

MIT
