from __future__ import annotations

from pathlib import Path

import torch

from core.models import build_model


def load_model_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        num_classes=len(checkpoint["class_names"]),
        dropout=checkpoint.get("dropout", 0.3),
        pretrained=False,
        small_image_stem=checkpoint.get("small_image_stem", True),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
