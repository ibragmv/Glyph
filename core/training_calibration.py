from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.runtime import RuntimeDevice, prepare_image_batch
from core.utils import format_float


class FocalLoss(nn.Module):
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: float | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        num_classes = logits.shape[1]

        target_dist = F.one_hot(targets, num_classes=num_classes).to(logits.dtype)
        if self.label_smoothing > 0.0:
            smoothing = self.label_smoothing / num_classes
            target_dist = target_dist * (1.0 - self.label_smoothing) + smoothing

        ce_per_sample = -(target_dist * log_probs).sum(dim=1)
        pt = (target_dist * probs).sum(dim=1).clamp_min(1e-8)
        focal_weight = (1.0 - pt).pow(self.gamma)
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha
        return (focal_weight * ce_per_sample).mean()


def build_criterion(
    *,
    loss_name: str,
    label_smoothing: float,
    focal_gamma: float,
    focal_alpha: float | None,
) -> nn.Module:
    normalized_loss_name = loss_name.strip().lower()
    if normalized_loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if normalized_loss_name == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


def apply_temperature(
    logits: torch.Tensor, temperature: float | torch.Tensor
) -> torch.Tensor:
    if isinstance(temperature, torch.Tensor):
        safe_temperature = temperature.clamp_min(1e-6)
    else:
        safe_temperature = max(float(temperature), 1e-6)
    return logits / safe_temperature


def collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: RuntimeDevice,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = prepare_image_batch(images, device)
            logits = model(images)
            logits_chunks.append(logits.detach())
            label_chunks.append(
                labels.to(device.device, non_blocking=device.pin_memory)
            )
    return torch.cat(logits_chunks, dim=0), torch.cat(label_chunks, dim=0)


def fit_temperature_scaling(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: RuntimeDevice,
    max_iter: int,
) -> dict[str, Any]:
    logits, labels = collect_logits_and_labels(model, loader, device)
    before_nll = float(F.cross_entropy(logits, labels).item())
    before_acc = float((logits.argmax(dim=1) == labels).float().mean().item())

    temperature = torch.ones(1, device=device.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [temperature],
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(apply_temperature(logits, temperature), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    final_temperature = max(float(temperature.detach().item()), 1e-3)
    calibrated_logits = apply_temperature(logits, final_temperature)
    after_nll = float(F.cross_entropy(calibrated_logits, labels).item())
    after_acc = float((calibrated_logits.argmax(dim=1) == labels).float().mean().item())
    return {
        "enabled": True,
        "method": "temperature_scaling",
        "temperature": format_float(final_temperature),
        "optimizer": "LBFGS",
        "max_iter": max_iter,
        "metrics": {
            "nll_before": format_float(before_nll),
            "nll_after": format_float(after_nll),
            "accuracy_before": format_float(before_acc),
            "accuracy_after": format_float(after_acc),
        },
    }


def create_disabled_calibration() -> dict[str, Any]:
    return {
        "enabled": False,
        "method": "temperature_scaling",
        "temperature": 1.0,
        "optimizer": None,
        "max_iter": None,
        "metrics": {},
    }
