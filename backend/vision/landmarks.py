"""
Face landmark detection with the MediaPipe FaceLandmarker task.

Equivalent to cvzone's FaceMeshDetector (same model, VIDEO running mode,
pixel coordinates), but loads ``face_landmarker.task`` from an absolute
path. cvzone loads it relative to the current working directory and
downloads it from the internet when it is missing.

A FaceLandmarker instance is not thread-safe; use it from one thread.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

Landmarks = list[tuple[int, int]]

# 468 face-mesh points + 10 iris points.
FULL_LANDMARK_COUNT = 478


class FaceLandmarker:
    def __init__(self, model_path: Path, max_faces: int = 1) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Face landmark model not found: {model_path}")
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect_all(self, frame_bgr: np.ndarray) -> list[Landmarks]:
        """Landmarks of every detected face (up to ``max_faces``), in pixels."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps.
        timestamp = max(self._last_timestamp_ms + 1, int(time.monotonic() * 1000))
        self._last_timestamp_ms = timestamp
        result = self._detector.detect_for_video(image, timestamp)
        h, w = frame_bgr.shape[:2]
        return [[(int(lm.x * w), int(lm.y * h)) for lm in face]
                for face in (result.face_landmarks or [])]

    def detect(self, frame_bgr: np.ndarray) -> Optional[Landmarks]:
        """Landmarks of the first face, or None."""
        faces = self.detect_all(frame_bgr)
        return faces[0] if faces else None

    def close(self) -> None:
        self._detector.close()
