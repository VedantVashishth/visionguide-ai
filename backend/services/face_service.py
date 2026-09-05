# backend/services/face_service.py
import cv2
import numpy as np

from ai_engine.face.face_recognizer import FaceRecognizer
from backend.core.logger import get_logger

logger = get_logger(__name__)

_face_recognizer = FaceRecognizer()


def get_face_recognizer() -> FaceRecognizer:
    return _face_recognizer


def _decode_frame(image_bytes: bytes) -> np.ndarray:
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image — file may be corrupted or not a valid image.")
    return frame


def list_known_faces() -> dict:
    names = _face_recognizer.list_known_people()
    logger.info("Listed %d known face(s)", len(names))
    return {"names": names}


def forget_face(name: str) -> dict:
    removed = _face_recognizer.forget_person(name)
    if removed:
        spoken_text = f"Okay, I've forgotten {name.strip()}."
    else:
        spoken_text = f"I don't have anyone saved as {name.strip()}."
    return {"removed": removed, "spoken_text": spoken_text}


def save_person_faces(image_bytes_list: list[bytes], name: str) -> dict:
    frames = [_decode_frame(image_bytes) for image_bytes in image_bytes_list]
    success = _face_recognizer.save_face(frames, name)

    if success:
        sample_count = len(image_bytes_list)
        spoken_text = f"Got it, I'll remember {name.strip()}."
        logger.info("Saved face '%s' from %d frame(s)", name.strip(), sample_count)
    else:
        spoken_text = "I couldn't find a clear face to save. Please hold still facing the camera and try again."

    return {"spoken_text": spoken_text, "saved": success}
