"""
Live positioning guidance with temporal smoothing.

A warning is shown only when most of the recent analysed frames agree, so a
single noisy frame does not make the message flicker. Gaze and head pose
are coaching hints; they say nothing about honesty or ability.
"""

from __future__ import annotations

from typing import Optional, Sequence

from backend.vision.eye_contact import LOOKING_CENTER
from backend.vision.face_analysis import FrameAnalysis
from backend.vision.head_pose import CENTER

# About 2 seconds at the default analysis interval of 0.33 s.
WINDOW = 6
MAJORITY = 0.6

OK = "ok"
WARNING = "warning"
WAITING = "waiting"


def _share(items: Sequence[bool]) -> float:
    return sum(items) / len(items) if items else 0.0


def guidance(recent: Sequence[FrameAnalysis]) -> tuple[str, str]:
    """(level, message) from the most recent analyses (oldest first)."""
    window = list(recent)[-WINDOW:]
    if not window:
        return WAITING, "Live feedback appears once recording starts."
    if _share([not a.face_detected for a in window]) >= MAJORITY:
        return WARNING, "Your face is not clearly visible. Move into the frame and check the lighting."
    faces = [a for a in window if a.face_detected]
    if _share([a.head != CENTER for a in faces]) >= MAJORITY:
        return WARNING, "Try turning your head towards the camera."
    if _share([a.eye != LOOKING_CENTER for a in faces]) >= MAJORITY:
        return WARNING, "Try to look towards the camera while you speak."
    return OK, "Good positioning."


def latest_labels(latest: Optional[FrameAnalysis]) -> dict[str, str]:
    """Labels for the live estimate display."""
    if latest is None:
        return {"face": "—", "eye": "—", "head": "—", "emotion": "—"}
    if not latest.face_detected:
        return {"face": "Not detected", "eye": "—", "head": "—", "emotion": "—"}
    emotion = "—"
    if latest.emotion:
        emotion = f"{latest.emotion} ({latest.emotion_confidence:.0%})" \
            if latest.emotion_confidence is not None else latest.emotion
    return {
        "face": "Detected",
        "eye": latest.eye or "—",
        "head": latest.head or "—",
        "emotion": emotion,
    }
