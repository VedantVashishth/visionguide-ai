"""Feature-flagged access to the optional MiDaS estimator."""
from __future__ import annotations

import numpy as np

from backend.core.config import get_settings

_estimator = None


def estimate_depth(frame: np.ndarray) -> np.ndarray | None:
    """Return relative depth when enabled; otherwise do no model work."""
    global _estimator
    settings = get_settings()
    if not settings.enable_depth_estimation:
        return None
    if _estimator is None:
        from ai_engine.depth.depth_estimator import DepthEstimator

        _estimator = DepthEstimator(settings.depth_model_name, settings.depth_device)
    return _estimator.estimate(frame)
