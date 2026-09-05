# ai_engine/face/face_recognizer.py
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from backend.core.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACE_DATA_DIR = PROJECT_ROOT / "models" / "faces"
SAMPLES_DIR = FACE_DATA_DIR / "samples"
LABELS_PATH = FACE_DATA_DIR / "labels.json"
MODEL_PATH = FACE_DATA_DIR / "lbph_model.yml"
CASCADE_PATH = PROJECT_ROOT / "models" / "haarcascade_frontalface_default.xml"

# LBPH: lower confidence = better match. Tune after multi-sample training.
RECOGNITION_CONFIDENCE_THRESHOLD = 70
MIN_SAMPLES_PER_SAVE = 1
TARGET_SAMPLES_PER_SAVE = 4


class FaceRecognizer:
    """
    Local, privacy-first face recognition using OpenCV's LBPH recognizer.
    No cloud calls — labels + model stay on disk under models/faces/.

    Same-name behavior: if the spoken name already exists (case-insensitive),
    new samples are appended to that person's label id instead of creating a
    duplicate entry.
    """

    def __init__(self):
        FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

        if not CASCADE_PATH.is_file():
            raise FileNotFoundError(f"Haar cascade not found at {CASCADE_PATH}")

        self.face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
        if self.face_cascade.empty():
            raise RuntimeError(f"Haar cascade failed to load (empty classifier): {CASCADE_PATH}")

        logger.info(f"Haar cascade loaded from {CASCADE_PATH}")

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels: dict[int, str] = {}
        self._load()

    def _load(self):
        if LABELS_PATH.is_file():
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = {int(k): v for k, v in json.load(f).items()}
            logger.info(f"Loaded {len(self.labels)} face label(s) from {LABELS_PATH}")
        else:
            logger.info("No labels.json found — starting with empty face database")

        if MODEL_PATH.is_file():
            self.recognizer.read(str(MODEL_PATH))
            logger.info(f"Loaded LBPH model from {MODEL_PATH}")
        else:
            logger.info("No lbph_model.yml found — model will be trained on first save")

    def _save(self):
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2)
        self.recognizer.write(str(MODEL_PATH))
        logger.info(f"Persisted face data ({len(self.labels)} label(s)) to {FACE_DATA_DIR}")

    def _prepare_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    def _detect_face(self, frame: np.ndarray) -> np.ndarray | None:
        """Detect the largest face and return a normalized 200x200 grayscale crop."""
        gray = self._prepare_gray(frame)
        height, width = gray.shape
        min_dim = min(height, width)
        min_size = max(30, int(min_dim * 0.08))

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_size, min_size),
        )

        if len(faces) == 0:
            logger.debug(
                "No face detected (frame=%dx%d, minSize=%d)",
                width,
                height,
                min_size,
            )
            return None

        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        logger.debug("Face detected at x=%d y=%d w=%d h=%d", x, y, w, h)
        return cv2.resize(gray[y : y + h, x : x + w], (200, 200))

    def _label_id_for_name(self, name: str) -> int | None:
        normalized = name.strip().lower()
        for label_id, stored_name in self.labels.items():
            if stored_name.lower() == normalized:
                return label_id
        return None

    def _next_label_id(self) -> int:
        if not self.labels:
            return 0
        return max(self.labels.keys()) + 1

    def _save_samples(self, label_id: int, faces: list[np.ndarray]) -> None:
        sample_dir = SAMPLES_DIR / str(label_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(sample_dir.glob("*.npy")))
        for offset, face in enumerate(faces):
            np.save(sample_dir / f"{existing + offset}.npy", face)

    def _collect_all_training_data(self) -> tuple[list[np.ndarray], list[int]]:
        faces: list[np.ndarray] = []
        ids: list[int] = []
        for label_id in sorted(self.labels.keys()):
            sample_dir = SAMPLES_DIR / str(label_id)
            if not sample_dir.is_dir():
                continue
            for sample_path in sorted(sample_dir.glob("*.npy")):
                faces.append(np.load(sample_path))
                ids.append(label_id)
        return faces, ids

    def _retrain_all(self) -> None:
        faces, ids = self._collect_all_training_data()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not faces:
            logger.info("No face samples left — cleared LBPH model")
            if MODEL_PATH.is_file():
                MODEL_PATH.unlink()
            return
        self.recognizer.train(faces, np.array(ids))
        logger.info("Retrained LBPH model from %d sample(s) across %d label(s)", len(faces), len(set(ids)))

    def list_known_people(self) -> list[str]:
        """Return unique remembered names (sorted alphabetically)."""
        return sorted(set(self.labels.values()), key=str.lower)

    def forget_person(self, name: str) -> bool:
        """
        Remove a person by name. Deletes their label mapping and on-disk samples,
        then rebuilds the LBPH model from remaining samples.
        """
        normalized = name.strip().lower()
        to_remove = [label_id for label_id, stored_name in self.labels.items() if stored_name.lower() == normalized]
        if not to_remove:
            logger.warning("forget_person: no match for name '%s'", name)
            return False

        for label_id in to_remove:
            del self.labels[label_id]
            sample_dir = SAMPLES_DIR / str(label_id)
            if sample_dir.is_dir():
                shutil.rmtree(sample_dir)
            logger.info("Removed label_id=%d for '%s'", label_id, name)

        self._retrain_all()
        self._save()
        return True

    def recognize(self, frame: np.ndarray) -> str | None:
        """Returns the person's name if recognized, None if no face or unknown face."""
        face = self._detect_face(frame)
        if face is None:
            return None

        if not self.labels or not MODEL_PATH.is_file():
            logger.debug("recognize: model or labels not ready")
            return None

        label_id, confidence = self.recognizer.predict(face)
        logger.info("recognize: label_id=%s confidence=%.1f threshold=%d", label_id, confidence, RECOGNITION_CONFIDENCE_THRESHOLD)

        if confidence < RECOGNITION_CONFIDENCE_THRESHOLD:
            name = self.labels.get(int(label_id))
            if name:
                logger.info("recognize: matched '%s'", name)
                return name
            logger.warning("recognize: label_id %s missing from labels.json", label_id)
            return None

        logger.info("recognize: no confident match (confidence %.1f)", confidence)
        return None

    def save_face(self, frames: np.ndarray | list[np.ndarray], name: str) -> bool:
        """
        Register or update a face under the given name using one or more frames.
        Collects up to TARGET_SAMPLES_PER_SAVE detections across the provided frames.
        """
        if isinstance(frames, np.ndarray):
            frame_list = [frames]
        else:
            frame_list = list(frames)

        clean_name = name.strip()
        if not clean_name:
            logger.warning("save_face: empty name provided")
            return False

        detected_faces: list[np.ndarray] = []
        for index, frame in enumerate(frame_list):
            face = self._detect_face(frame)
            if face is not None:
                detected_faces.append(face)
                logger.info("save_face: captured sample %d/%d", len(detected_faces), len(frame_list))
            else:
                logger.debug("save_face: no face in frame %d", index + 1)

        if len(detected_faces) < MIN_SAMPLES_PER_SAVE:
            logger.warning(
                "save_face: need at least %d face sample(s), got %d from %d frame(s)",
                MIN_SAMPLES_PER_SAVE,
                len(detected_faces),
                len(frame_list),
            )
            return False

        existing_id = self._label_id_for_name(clean_name)
        if existing_id is not None:
            label_id = existing_id
            logger.info("save_face: updating existing person '%s' (label_id=%d)", clean_name, label_id)
        else:
            label_id = self._next_label_id()
            logger.info("save_face: registering new person '%s' (label_id=%d)", clean_name, label_id)

        self.labels[label_id] = clean_name
        self._save_samples(label_id, detected_faces)

        if MODEL_PATH.is_file() and existing_id is not None:
            ids = [label_id] * len(detected_faces)
            self.recognizer.update(detected_faces, np.array(ids))
            logger.info("save_face: updated LBPH model with %d new sample(s)", len(detected_faces))
        else:
            self._retrain_all()

        self._save()
        logger.info(
            "save_face: saved %d sample(s) for '%s' (label_id=%d, total names=%d)",
            len(detected_faces),
            clean_name,
            label_id,
            len(self.list_known_people()),
        )
        return True
