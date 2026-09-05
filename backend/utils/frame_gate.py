# backend/utils/frame_gate.py
"""
Cheap frame-change gating for Gemini-backed features.
Uses grayscale histogram comparison — no ML, no extra API calls.
"""
import cv2
import numpy as np

from backend.core.logger import get_logger

logger = get_logger(__name__)

# Bhattacharyya distance below this threshold => frame treated as unchanged.
SIMILARITY_THRESHOLD = 0.12

_gate_cache: dict[str, tuple[np.ndarray, str]] = {}


def _frame_histogram(image_bytes: bytes) -> np.ndarray | None:
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None

    small = cv2.resize(frame, (160, 120))
    hist = cv2.calcHist([small], [0], None, [64], [0, 256])
    return cv2.normalize(hist, hist).flatten()


def get_cached_result_if_unchanged(image_bytes: bytes, feature_key: str) -> str | None:
    """
    Return the last cached result for this feature if the frame hasn't changed
    meaningfully since the last call. Otherwise return None.
    """
    hist = _frame_histogram(image_bytes)
    if hist is None:
        return None

    cached = _gate_cache.get(feature_key)
    if cached is None:
        logger.info("Frame gate [%s]: no prior frame — will call API", feature_key)
        return None

    prev_hist, last_result = cached
    distance = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

    if distance < SIMILARITY_THRESHOLD:
        logger.info(
            "Frame gate [%s]: unchanged (dist=%.3f) — reusing cached result",
            feature_key,
            distance,
        )
        return last_result

    logger.info(
        "Frame gate [%s]: changed (dist=%.3f) — will call API",
        feature_key,
        distance,
    )
    return None


def store_result(image_bytes: bytes, feature_key: str, result: str) -> None:
    hist = _frame_histogram(image_bytes)
    if hist is not None:
        _gate_cache[feature_key] = (hist, result)
