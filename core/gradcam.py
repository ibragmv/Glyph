from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(
        self, image_tensor: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


def overlay_heatmap(
    grayscale_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    import matplotlib.pyplot as plt

    grayscale = grayscale_image.astype(np.float32)
    grayscale -= grayscale.min()
    grayscale /= grayscale.max() + 1e-8
    grayscale_rgb = np.stack([grayscale, grayscale, grayscale], axis=-1)
    heatmap_rgb = plt.get_cmap("inferno")(heatmap)[..., :3]
    overlay = (1.0 - alpha) * grayscale_rgb + alpha * heatmap_rgb
    return np.clip(overlay, 0.0, 1.0)
