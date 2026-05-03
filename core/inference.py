from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from core.augmentations import build_eval_transforms
from core.checkpoints import load_model_checkpoint


def predict_image(
    checkpoint_path: Path,
    image_path: Path,
    top_k: int = 3,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device)

    image = Image.open(image_path).convert("L").resize((64, 64))
    array = np.asarray(image)
    transform = build_eval_transforms(checkpoint["mean"], checkpoint["std"])
    tensor = transform(image=array)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.numel()))

    predictions = [
        {
            "rank": rank,
            "label": checkpoint["class_names"][class_idx],
            "confidence": float(score),
        }
        for rank, (score, class_idx) in enumerate(
            zip(top_probs.tolist(), top_indices.tolist()),
            start=1,
        )
    ]
    return {
        "device": device.type,
        "image_path": str(image_path),
        "predictions": predictions,
    }
