from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transforms(mean: float, std: float) -> A.Compose:
    return A.Compose(
        [
            A.GaussNoise(std_range=(0.04, 0.12), mean_range=(0.0, 0.0), p=0.55),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                ],
                p=0.35,
            ),
            A.Affine(
                translate_percent={"x": (-0.12, 0.12), "y": (-0.12, 0.12)},
                scale=(0.82, 1.14),
                rotate=(-24, 24),
                shear=(-12, 12),
                border_mode=0,
                fill=255,
                p=0.70,
            ),
            A.OneOf(
                [
                    A.Perspective(
                        scale=(0.04, 0.10), keep_size=True, fit_output=False, fill=255
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=(-0.22, 0.22),
                        border_mode=0,
                        fill=255,
                    ),
                    A.ElasticTransform(alpha=6.0, sigma=24.0, border_mode=0, fill=255),
                ],
                p=0.35,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.28, 0.12),
                        contrast_limit=(-0.35, 0.20),
                    ),
                    A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8)),
                ],
                p=0.35,
            ),
            A.Downscale(scale_range=(0.45, 0.82), p=0.30),
            A.CoarseDropout(
                num_holes_range=(1, 5),
                hole_height_range=(0.06, 0.22),
                hole_width_range=(0.06, 0.22),
                fill=255,
                p=0.45,
            ),
            A.Normalize(mean=(mean,), std=(std,), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(mean: float, std: float) -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=(mean,), std=(std,), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
