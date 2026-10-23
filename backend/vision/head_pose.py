"""
Head direction estimate from nose position relative to the face (heuristic).

Thresholds are in pixels (15 px horizontal, -10/+15 px vertical), so their
meaning changes with distance from the camera. Not validated; coaching hint
only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

NOSE_TIP, LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD, CHIN = 1, 234, 454, 10, 152
CENTER = "Center"


def head_pose_from_landmarks(face: Sequence[Sequence[float]]) -> str:
    if len(face) <= RIGHT_CHEEK:
        raise ValueError("Face landmarks are incomplete.")
    nose = np.asarray(face[NOSE_TIP], dtype=float)
    center = (np.asarray(face[LEFT_CHEEK], dtype=float) + np.asarray(face[RIGHT_CHEEK], dtype=float)) / 2
    horizontal = nose[0] - center[0]
    vertical = nose[1] - (face[FOREHEAD][1] + face[CHIN][1]) / 2

    if horizontal < -15:
        return "Looking Right"
    if horizontal > 15:
        return "Looking Left"
    if vertical < -10:
        return "Looking Up"
    if vertical > 15:
        return "Looking Down"
    return CENTER


def detect_head_pose(frame) -> str:
    """Legacy entry point: detect landmarks in ``frame`` and classify the pose."""
    from backend.core.models import get_registry

    face = get_registry().get("face_mesh").detect(frame)
    if face is None:
        return "No Face"
    return head_pose_from_landmarks(face)
