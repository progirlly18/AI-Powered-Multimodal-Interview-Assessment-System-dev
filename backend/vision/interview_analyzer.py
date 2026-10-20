"""
Compatibility wrapper around ``VisionSession``.

The earlier implementations of this module (on ``main`` and in the
2026-09-27 working copy) opened the camera and ran inference inside the
capture loop. That logic now lives in:

* ``backend.vision.capture``   - capture thread + background analysis
* ``backend.vision.guidance``  - smoothed live guidance messages
* ``backend.services.interview_service`` - countdown, states, cleanup

``analyze_interview`` remains for scripts that want vision-only analysis.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from backend.config import load_settings
from backend.core.models import get_registry
from backend.core.results import VisionResult
from backend.vision.capture import OpenCVCamera, VisionSession
from backend.vision.face_analysis import FaceAnalyzer, FrameAnalysis
from backend.vision.guidance import guidance

FrameCallback = Callable[[object, Optional[FrameAnalysis], str], None]


def analyze_interview(
    duration: float = 20,
    start_event: Optional[threading.Event] = None,
    frame_callback: Optional[FrameCallback] = None,
) -> VisionResult:
    """Record ``duration`` seconds of video analysis and return the summary.

    ``frame_callback(frame_bgr, latest_analysis, guidance_text)`` is called
    from this thread about 10 times per second; it must not modify the frame.
    """
    settings = load_settings()
    registry = get_registry()
    analyzer = FaceAnalyzer(lambda: registry.get("face_mesh"), lambda: registry.get("emotion"))
    session = VisionSession(OpenCVCamera(settings.camera_index), analyzer,
                            settings.vision_interval_s)
    session.start()
    try:
        if start_event is not None:
            start_event.wait()
        session.begin_recording()
        end = time.monotonic() + duration
        while time.monotonic() < end:
            if session.error:
                break
            if frame_callback is not None:
                frame, latest = session.snapshot()
                _level, text = guidance(session.recent_analyses(6))
                frame_callback(frame, latest, text)
            time.sleep(0.1)
        session.end_recording()
    finally:
        session.stop()
    return session.result()


if __name__ == "__main__":
    print(analyze_interview(duration=10))
