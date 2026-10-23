"""
Horizontal gaze estimate from iris position (heuristic).

The ratio of each iris centre between the eye corners is averaged; below
0.35 or above 0.65 counts as looking to one side. These thresholds have not
been validated, the labels depend on whether the image is mirrored, and
webcam gaze estimation is imprecise. Use it as a coaching hint only.
"""

from __future__ import annotations

from typing import Sequence

LEFT_IRIS, RIGHT_IRIS = 468, 473
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263

LOOKING_CENTER = "Looking Center"
LOOKING_LEFT = "Looking Left"
LOOKING_RIGHT = "Looking Right"


def gaze_ratio(face: Sequence[Sequence[float]]) -> float:
    if len(face) <= RIGHT_IRIS:
        raise ValueError("Iris landmarks are missing (need 478 landmarks).")
    left = (face[LEFT_IRIS][0] - face[LEFT_EYE_OUTER][0]) / (
        face[LEFT_EYE_INNER][0] - face[LEFT_EYE_OUTER][0] + 1e-6
    )
    right = (face[RIGHT_IRIS][0] - face[RIGHT_EYE_INNER][0]) / (
        face[RIGHT_EYE_OUTER][0] - face[RIGHT_EYE_INNER][0] + 1e-6
    )
    return (left + right) / 2


def eye_direction_from_landmarks(face: Sequence[Sequence[float]]) -> str:
    ratio = gaze_ratio(face)
    if ratio < 0.35:
        return LOOKING_RIGHT
    if ratio > 0.65:
        return LOOKING_LEFT
    return LOOKING_CENTER


def detect_eye_contact(frame) -> str:
    """Legacy entry point: detect landmarks in ``frame`` and classify gaze."""
    from backend.core.models import get_registry

    face = get_registry().get("face_mesh").detect(frame)
    if face is None:
        return "No Face"
    return eye_direction_from_landmarks(face)
