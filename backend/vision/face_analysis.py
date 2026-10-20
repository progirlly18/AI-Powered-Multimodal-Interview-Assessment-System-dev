"""
Per-frame face analysis: one landmark pass shared by gaze, head pose and
the expression model (the original code ran face detection three times per
analysed frame).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from backend.vision.emotion_detector import BBox, face_bbox, predict_emotion
from backend.vision.eye_contact import eye_direction_from_landmarks
from backend.vision.head_pose import head_pose_from_landmarks


@dataclass(frozen=True)
class FrameAnalysis:
    timestamp: float
    face_detected: bool
    bbox: Optional[BBox] = None
    eye: Optional[str] = None
    head: Optional[str] = None
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    timings_ms: dict[str, float] = field(default_factory=dict)


class FaceAnalyzer:
    """Runs landmarks, gaze, head pose and (optionally) expression on a frame.

    Not thread-safe: call ``analyze`` from a single worker thread.
    """

    def __init__(
        self,
        landmarker_provider: Callable[[], Any],
        emotion_model_provider: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._landmarker_provider = landmarker_provider
        self._emotion_model_provider = emotion_model_provider

    def analyze(self, frame_bgr: np.ndarray, timestamp: float) -> FrameAnalysis:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        face = self._landmarker_provider().detect(frame_bgr)
        timings["landmarks"] = (time.perf_counter() - start) * 1000
        if face is None:
            return FrameAnalysis(timestamp, face_detected=False, timings_ms=timings)

        bbox = face_bbox(face, frame_bgr.shape)
        eye = eye_direction_from_landmarks(face)
        head = head_pose_from_landmarks(face)

        emotion: Optional[str] = None
        confidence: Optional[float] = None
        if self._emotion_model_provider is not None and bbox is not None:
            start = time.perf_counter()
            emotion, confidence = predict_emotion(self._emotion_model_provider(), frame_bgr, bbox)
            timings["emotion"] = (time.perf_counter() - start) * 1000

        return FrameAnalysis(
            timestamp=timestamp,
            face_detected=True,
            bbox=bbox,
            eye=eye,
            head=head,
            emotion=emotion,
            emotion_confidence=confidence,
            timings_ms=timings,
        )
