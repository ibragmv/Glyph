# Glyph

Glyph is a local CLI toolkit for classifying printed Imperial Aramaic letters from grayscale images.

It covers the full workflow:

- synthetic dataset generation
- model training
- checkpoint evaluation
- single-image inference on new samples

The project is built for experimentation with Imperial Aramaic character classification when a large real labeled dataset is not yet available. It generates synthetic glyph images, trains a CNN classifier, evaluates checkpoints, and lets you test the model on your own external images.

## What This Project Is

At a high level, Glyph is a printed-letter classifier for Imperial Aramaic.

The current dataset pipeline is synthetic-first:

- glyphs are rendered from Imperial Aramaic fonts
- scan-like corruption is added: blur, texture, background noise, geometric distortions, damage
- training and validation splits use different difficulty profiles

This means the model can be trained without manually annotating thousands of real images first.

## What The Commands Mean

The project exposes one main command after build:

```bash
./glyph
```

Available subcommands:

- `data` - generate a synthetic dataset
- `train` - train the classifier on `dataset/train`
- `eval` - evaluate a checkpoint on `dataset/val`
- `infer` - run prediction on one image

The normal sequence is:

1. `data`
2. `train`
3. `eval`
4. `infer` on your own images

## Quick Start

### 1. Prepare the environment

```bash
./scripts/startup.sh
```

This creates `.venv/`, installs dependencies, and skips reinstalling them if nothing changed.

### 2. Build the launcher

```bash
make glyph
```

This creates:

- `./glyph` - executable launcher
- `./.glyph/` - copied runtime used by the launcher

If you recreate `.venv`, run `make glyph` again.

### 3. Generate the dataset

Default dataset:

```bash
./glyph data --output-dir dataset
```

Harder validation split:

```bash
./glyph data --output-dir dataset --train-hardness 1.0 --val-hardness 1.45
```

This writes:

- `dataset/train/...`
- `dataset/val/...`
- `dataset/metadata.json`

Default counts:

- `1200` training images per class
- `300` validation images per class

### 4. Train the model

```bash
./glyph train --data-dir dataset --output-dir artifacts
```

Typical outputs:

- `artifacts/best_model.pt`
- `artifacts/last_model.pt`
- `artifacts/history.json`
- `artifacts/training_curves.png`

### 5. Evaluate the model

```bash
./glyph eval --data-dir dataset --checkpoint artifacts/best_model.pt
```

This writes:

- `artifacts/eval/confusion_matrix.png`
- `artifacts/eval/classification_report.json`
- `artifacts/eval/random_predictions.png`
- `artifacts/eval/gradcam_examples.png`

### 6. Test on your own image

```bash
./glyph infer --checkpoint artifacts/best_model.pt --image path/to/sample.png
```

This prints top predictions and confidence scores.

## How To Test On External Images

If you have images from a teacher, colleague, paper, archive, or your own scans, do not put them into `dataset/train` or `dataset/val` unless you explicitly want to rebuild the training pipeline around them.

The simplest workflow is:

1. Keep external images in a separate folder, for example `teacher_test/`
2. Train your model normally on synthetic data
3. Run `infer` on each external image
4. Compare predictions with the true labels manually

Example for one image:

```bash
./glyph infer --checkpoint artifacts/best_model.pt --image teacher_test/sample_01.png
```

Example for many images:

```bash
for f in teacher_test/*; do
  echo "== $f =="
  ./glyph infer --checkpoint artifacts/best_model.pt --image "$f"
done
```

How to interpret this:

- if `eval` is very high and external images also work well, the model is genuinely useful
- if `eval` is very high but external images fail, synthetic validation is still overestimating real performance
- if confidence is high on wrong answers, the model is confidently miscalibrated for your external domain

Recommended manual protocol:

1. Take 20-50 external images
2. Write down the true label for each image
3. Run `infer` on all of them
4. Record `file | true_label | predicted_label | confidence`
5. Compute your real external accuracy separately from synthetic validation accuracy

## Full Example

```bash
./scripts/startup.sh
make glyph
./glyph data --output-dir dataset
./glyph train --data-dir dataset --output-dir artifacts
./glyph eval --data-dir dataset --checkpoint artifacts/best_model.pt
./glyph infer --checkpoint artifacts/best_model.pt --image path/to/sample.png
```

## Outputs

### `dataset/`

Generated synthetic data:

- `dataset/train/` - training split
- `dataset/val/` - validation split
- `dataset/metadata.json` - generation metadata

### `artifacts/`

Training and evaluation outputs:

- `best_model.pt` - checkpoint with best validation accuracy
- `last_model.pt` - most recent checkpoint
- `history.json` - training metrics history
- `training_curves.png` - loss and accuracy curves
- `eval/` - evaluation reports and visualizations

`artifacts/` is generated output, not source code.

## Command Reference

### `./glyph data`

Generates the synthetic dataset.

Example:

```bash
./glyph data --output-dir dataset
```

Flags:

- `--output-dir` - where to write the dataset. Default: `dataset`
- `--train-per-class` - number of training images per class. Default: `1200`
- `--val-per-class` - number of validation images per class. Default: `300`
- `--canvas-size` - internal render canvas size before final resize. Default: `128`
- `--output-size` - final saved image size. Default: `64`
- `--seed` - random seed for reproducibility. Default: `42`
- `--font` - explicit font path. Can be passed multiple times. Default: none
- `--font-dir` - extra directory to search for fonts. Can be passed multiple times. Default: none
- `--train-hardness` - corruption severity for training images. Default: `0.95`
- `--val-hardness` - corruption severity for validation images. Default: `1.35`
- `--holdout-font-fraction` - fraction of fonts held out for validation when multiple fonts are available. Default: `0.35`
- `--progress` - show progress bars. Default: off

### `./glyph train`

Trains the classifier on `dataset/train` and validates on `dataset/val`.

Example:

```bash
./glyph train --data-dir dataset --output-dir artifacts
```

Flags:

- `--data-dir` - dataset root containing `train/` and `val/`. Default: `dataset`
- `--output-dir` - where to save checkpoints and training artifacts. Default: `artifacts`
- `--epochs` - number of training epochs. Default: `30`
- `--batch-size` - batch size. Default: `128`
- `--lr` - learning rate. Default: `1e-3`
- `--num-workers` - DataLoader workers. Default: `0`
- `--seed` - random seed. Default: `42`
- `--pretrained` - initialize ResNet-18 from pretrained weights. Default: off
- `--no-small-image-stem` - disable the small-image input stem optimization. Default: off
- `--progress` - show progress bars. Default: off

Defaults that matter:

- if `--pretrained` is not passed, training starts without pretrained weights
- if `--no-small-image-stem` is not passed, the small-image stem stays enabled
- output goes to `artifacts/` unless you override it

### `./glyph eval`

Evaluates a checkpoint on the validation split and writes reports to `artifacts/eval` by default.

Example:

```bash
./glyph eval --data-dir dataset --checkpoint artifacts/best_model.pt
```

Flags:

- `--data-dir` - dataset root used for validation. Default: `dataset`
- `--checkpoint` - checkpoint file to evaluate. Required
- `--output-dir` - where to save evaluation outputs. Default: `artifacts/eval`
- `--batch-size` - evaluation batch size. Default: `128`
- `--num-workers` - DataLoader workers. Default: `0`
- `--seed` - seed used for random prediction examples. Default: `42`
- `--progress` - show progress bars. Default: off

Important:

- `eval` runs after `train`, not directly after `data`
- `eval` needs both the dataset and a trained checkpoint

### `./glyph infer`

Runs prediction for one image.

Example:

```bash
./glyph infer --checkpoint artifacts/best_model.pt --image teacher_test/sample_01.png
```

Flags:

- `--checkpoint` - checkpoint file to load. Required
- `--image` - image file to classify. Required
- `--top-k` - number of top predictions to print. Default: `3`

Use `infer` when:

- you want to test the model on external images
- you want to inspect one image at a time
- you want to compare model behavior on synthetic vs real samples

## Project Structure

- `core/` - application code
- `scripts/startup.sh` - environment bootstrap
- `scripts/build.py` - launcher build step
- `fonts/` - Imperial Aramaic font files
- `dataset/` - generated synthetic data
- `artifacts/` - generated checkpoints and evaluation reports
- `.glyph/` - generated runtime for the launcher
- `glyph` - generated executable launcher

## Cleanup

Remove only the launcher and local runtime:

```bash
make clean
```

Remove generated checkpoints and evaluation output:

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
