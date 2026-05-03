# Glyph

Glyph is a local CLI toolkit for classifying printed Imperial Aramaic letters.

The project includes the full pipeline:

- synthetic dataset generation
- model training
- checkpoint evaluation
- single-image inference

It is designed as a compact research and experimentation project: generate glyph images, train a classifier, inspect evaluation outputs, and run predictions on new samples from the command line.

## What The Project Does

Glyph trains a neural network to recognize printed Imperial Aramaic characters from grayscale images.

The current workflow is built around synthetic data. The generator renders glyphs from Imperial Aramaic fonts and applies scan-like distortions such as blur, background texture, corruption, geometric warping, and split-specific difficulty. This makes it possible to train a classifier without first hand-labeling a large real dataset.

The project is useful if you want to:

- experiment with Imperial Aramaic OCR-style classification
- train a compact image classifier on glyph crops
- benchmark synthetic train/validation splits
- inspect prediction quality with confusion matrices and Grad-CAM visualizations

## What You Get

After setup, the project provides a single local command:

```bash
./glyph
```

That command supports:

- `data` - generate a synthetic dataset
- `train` - train the classifier
- `eval` - evaluate a checkpoint on the validation split
- `infer` - predict a single image

## Quick Start

### 1. Prepare the environment

Create the virtual environment and install dependencies:

```bash
./scripts/startup.sh
```

What this does:

- creates `.venv/` if needed
- installs packages from `requirements.txt`
- skips reinstalling dependencies if nothing changed

### 2. Build the launcher

Build the local executable:

```bash
make glyph
```

This creates:

- `./glyph` - executable launcher
- `./.glyph/` - copied runtime package used by the launcher

If `.venv` is recreated, run `make glyph` again.

### 3. Generate a dataset

Generate the default dataset:

```bash
./glyph data --output-dir dataset
```

If you want a harder validation split:

```bash
./glyph data --output-dir dataset --train-hardness 1.0 --val-hardness 1.45
```

This writes:

- `dataset/train/...`
- `dataset/val/...`
- `dataset/metadata.json`

By default:

- `train` contains `1200` images per class
- `val` contains `300` images per class
- the validation split is intentionally harder than the training split

### 4. Train the model

Train a classifier on the generated dataset:

```bash
./glyph train --data-dir dataset --output-dir artifacts
```

Typical outputs:

- `artifacts/best_model.pt`
- `artifacts/last_model.pt`
- `artifacts/history.json`
- `artifacts/training_curves.png`

Useful optional flags:

```bash
./glyph train --data-dir dataset --output-dir artifacts --epochs 30 --batch-size 128 --lr 1e-3
./glyph train --data-dir dataset --output-dir artifacts --pretrained
./glyph train --data-dir dataset --output-dir artifacts --progress
```

### 5. Evaluate the checkpoint

Run evaluation on the validation split:

```bash
./glyph eval --data-dir dataset --checkpoint artifacts/best_model.pt
```

This writes:

- `artifacts/eval/confusion_matrix.png`
- `artifacts/eval/classification_report.json`
- `artifacts/eval/random_predictions.png`
- `artifacts/eval/gradcam_examples.png`

### 6. Run inference on one image

Predict a single image:

```bash
./glyph infer --checkpoint artifacts/best_model.pt --image path/to/sample.png
```

The command prints the top predicted classes with confidence scores.

## Full Example

Minimal end-to-end flow:

```bash
./scripts/startup.sh
make glyph
./glyph data --output-dir dataset
./glyph train --data-dir dataset --output-dir artifacts
./glyph eval --data-dir dataset --checkpoint artifacts/best_model.pt
./glyph infer --checkpoint artifacts/best_model.pt --image path/to/sample.png
```

## Project Structure

Source files:

- `core/` - main application code
- `scripts/startup.sh` - environment bootstrap
- `scripts/build.py` - launcher build entrypoint
- `Makefile` - setup and cleanup commands
- `fonts/` - local Imperial Aramaic font assets

Generated files:

- `dataset/` - generated synthetic images and metadata
- `artifacts/` - model checkpoints and evaluation outputs
- `.glyph/` - generated launcher runtime
- `glyph` - generated executable launcher

## Dataset Generation Notes

The synthetic generator uses split-specific difficulty instead of producing nearly identical train and validation distributions.

In practice this means:

- `train` is augmented to improve robustness
- `val` is harder and more scan-like
- when more than one font is available, part of the font set can be held out for validation

The generated metadata file records:

- class list
- image counts
- font paths
- split-specific font allocation
- hardness settings

## Artifacts Notes

`artifacts/` is generated output, not source code.

In the current workspace it contains large checkpoint files, so treat it as disposable unless you want to keep the trained weights.

## Cleanup

Remove only the launcher and local runtime:

```bash
make clean
```

Remove generated checkpoints and evaluation outputs:

```bash
make clean-artifacts
```

Remove the generated dataset:

```bash
make clean-dataset
```

Remove all generated outputs:

```bash
make clean-generated
```

## License

This project is licensed under the MIT License. See [LICENSE](/Users/ibragimibragimov/Eldenlord/Glyph/LICENSE).
