"""
Per-frame face analysis: one landmark pass shared by gaze, head pose and
the expression model (the original code ran face detection three times per
analysed frame).

The expression model is the expensive step (~310 ms per call on the test
laptop, scripts/benchmark.py), so it runs at most once per
``emotion_interval_s``; gaze and head pose run on every analysed frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from backend.vision.emotion_detector import (
    EMOTION_LABELS,
    BBox,
    emotion_probabilities,
    face_bbox,
)
from backend.vision.eye_contact import eye_direction_from_landmarks
from backend.vision.head_pose import head_pose_from_landmarks


@dataclass(frozen=True)
class FrameAnalysis:
    timestamp: float
    face_detected: bool
    bbox: Optional[BBox] = None
    eye: Optional[str] = None
    head: Optional[str] = None
    # Raw expression output for this frame, or None if the model did not run.
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    face_count: int = 0
    emotion_probs: Optional[tuple[float, ...]] = None


class FaceAnalyzer:
    """Runs landmarks, gaze, head pose and (optionally) expression on a frame.

    Not thread-safe: call ``analyze`` from a single worker thread.
    """

    def __init__(
        self,
        landmarker_provider: Callable[[], Any],
        emotion_model_provider: Optional[Callable[[], Any]] = None,
        emotion_interval_s: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._landmarker_provider = landmarker_provider
        self._emotion_model_provider = emotion_model_provider
        self._emotion_interval_s = emotion_interval_s
        self._clock = clock
        self._last_emotion_at = float("-inf")

    def _faces(self, frame_bgr: np.ndarray) -> list:
        landmarker = self._landmarker_provider()
        if hasattr(landmarker, "detect_all"):
            return list(landmarker.detect_all(frame_bgr))
        face = landmarker.detect(frame_bgr)
        return [face] if face is not None else []

    def analyze(self, frame_bgr: np.ndarray, timestamp: float) -> FrameAnalysis:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        faces = self._faces(frame_bgr)
        timings["landmarks"] = (time.perf_counter() - start) * 1000
        if not faces:
            return FrameAnalysis(timestamp, face_detected=False, timings_ms=timings)

        face = faces[0]
        bbox = face_bbox(face, frame_bgr.shape)
        eye = eye_direction_from_landmarks(face)
        head = head_pose_from_landmarks(face)

        emotion: Optional[str] = None
        confidence: Optional[float] = None
        probs: Optional[tuple[float, ...]] = None
        now = self._clock()
        due = now - self._last_emotion_at >= self._emotion_interval_s
        if self._emotion_model_provider is not None and bbox is not None and due:
            self._last_emotion_at = now
            start = time.perf_counter()
            p = emotion_probabilities(self._emotion_model_provider(), frame_bgr, bbox)
            timings["emotion"] = (time.perf_counter() - start) * 1000
            idx = int(np.argmax(p))
            emotion, confidence = EMOTION_LABELS[idx], float(p[idx])
            probs = tuple(float(x) for x in p)

        return FrameAnalysis(
            timestamp=timestamp,
            face_detected=True,
            bbox=bbox,
            eye=eye,
            head=head,
            emotion=emotion,
            emotion_confidence=confidence,
            timings_ms=timings,
            face_count=len(faces),
            emotion_probs=probs,
        )
