"""Lazy MiDaS depth estimator, kept isolated from the CPU runtime."""
from __future__ import annotations

import numpy as np


class DepthEstimator:
    """Estimate relative depth; larger returned values are nearer pixels."""

    def __init__(self, model_name: str = "MiDaS_small", device: str = "auto") -> None:
        import torch

        self.torch = torch
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        self.model = torch.hub.load("intel-isl/MiDaS", model_name).to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform if model_name == "MiDaS_small" else transforms.dpt_transform

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return a normalized relative-depth map matching the input dimensions."""
        import cv2

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        with self.torch.no_grad():
            prediction = self.model(self.transform(rgb_frame).to(self.device))
            prediction = self.torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=bgr_frame.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
        depth = prediction.cpu().numpy()
        return cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
