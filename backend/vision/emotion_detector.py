"""
Facial-expression classification with the project's EfficientNetB0 model.

The project docs report about 51% validation accuracy on FER2013 (7
classes, posed lab images). Webcam conditions differ, so predictions are
shown as informational estimates and are not part of the practice score.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

EMOTION_LABELS = ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
INPUT_SIZE = (224, 224)
CROP_MARGIN = 20

BBox = tuple[int, int, int, int]


def face_bbox(face: Sequence[Sequence[float]], frame_shape: Sequence[int],
              margin: int = CROP_MARGIN) -> Optional[BBox]:
    """Bounding box around the landmarks, padded and clipped to the frame."""
    if not face:
        return None
    xs = [int(p[0]) for p in face]
    ys = [int(p[1]) for p in face]
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x1, y1 = max(0, min(xs) - margin), max(0, min(ys) - margin)
    x2, y2 = min(w, max(xs) + margin), min(h, max(ys) + margin)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def preprocess(frame_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
    import cv2
    from tensorflow.keras.applications.efficientnet import preprocess_input

    x1, y1, x2, y2 = bbox
    crop = cv2.resize(frame_bgr[y1:y2, x1:x2], INPUT_SIZE)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype("float32")
    return np.expand_dims(preprocess_input(crop), axis=0)


def predict_emotion(model: Any, frame_bgr: np.ndarray, bbox: BBox) -> tuple[str, float]:
    """(label, softmax confidence) for the face in ``bbox``."""
    batch = preprocess(frame_bgr, bbox)
    # model.predict() measured ~310 ms per call here; calling the model
    # directly measured ~1010 ms (scripts/benchmark.py, 2026-10-19), so keep predict().
    probs = np.asarray(model.predict(batch, verbose=0))[0]
    if probs.shape != (len(EMOTION_LABELS),) or not np.all(np.isfinite(probs)):
        raise ValueError(f"Unexpected emotion model output shape {probs.shape}")
    idx = int(np.argmax(probs))
    return EMOTION_LABELS[idx], float(probs[idx])


def detect_emotion(frame):
    """Legacy entry point returning (label, confidence, bbox)."""
    from backend.core.models import get_registry

    registry = get_registry()
    face = registry.get("face_mesh").detect(frame)
    bbox = face_bbox(face, frame.shape) if face else None
    if bbox is None:
        return "No Face", 0.0, None
    label, confidence = predict_emotion(registry.get("emotion"), frame, bbox)
    return label, confidence, bbox
