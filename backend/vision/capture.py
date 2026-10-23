"""
Camera capture and background face analysis.

Two threads keep inference out of the capture loop:

* the capture thread only reads frames and stores the newest one;
* the analysis thread analyses the newest frame at a fixed interval.

Only the latest frame is ever analysed, so a slow model cannot build up a
backlog. The camera is released on every exit path.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from typing import Any, Callable, Optional, Protocol

import numpy as np

from backend.audio.recorder import DeviceUnavailableError
from backend.core.results import Metric, Status, VisionResult
from backend.vision.eye_contact import LOOKING_CENTER
from backend.vision.face_analysis import FaceAnalyzer, FrameAnalysis
from backend.vision.head_pose import CENTER

log = logging.getLogger(__name__)

FIRST_FRAME_TIMEOUT_S = 5.0
MAX_CONSECUTIVE_READ_FAILURES = 30
MAX_ANALYSIS_ERRORS = 5
MIN_FRAMES_ANALYSED = 3
MIN_FACE_PRESENCE_PCT = 50.0


class CameraSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> tuple[bool, Optional[np.ndarray]]: ...
    def release(self) -> None: ...


class OpenCVCamera:
    """A webcam opened with OpenCV."""

    def __init__(self, index: int = 0) -> None:
        self._index = index
        self._cap: Any = None

    def open(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            cap.release()
            raise DeviceUnavailableError(
                "camera",
                "The camera could not be opened. Check that a webcam is connected, "
                "that Windows allows apps to use it (Settings > Privacy & security > "
                "Camera) and that no other app (Teams, Zoom, browser) is using it.",
            )
        self._cap = cap

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        return bool(ok), frame if ok else None

    def release(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()


CameraFactory = Callable[[], CameraSource]


def probe_camera(camera: CameraSource) -> tuple[bool, str, Optional[np.ndarray]]:
    """Open the camera, grab one frame, release it.

    Returns (ok, message, frame). The frame is only for an on-screen preview;
    callers must not store it beyond the page that shows it.
    """
    try:
        camera.open()
    except DeviceUnavailableError as exc:
        return False, str(exc), None
    try:
        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_S
        while time.monotonic() < deadline:
            ok, frame = camera.read()
            if ok and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
                h, w = frame.shape[:2]
                return True, f"Camera ready ({w}x{h}).", frame
            time.sleep(0.05)
        return False, "The camera opened but did not deliver any video frames.", None
    finally:
        camera.release()


def check_camera(camera: CameraSource) -> tuple[bool, str]:
    """Open the camera, confirm one frame arrives, release it. Nothing is kept."""
    ok, message, _frame = probe_camera(camera)
    return ok, message


class VisionSession:
    """Owns the camera for one recording and analyses frames in the background."""

    def __init__(self, camera: CameraSource, analyzer: Optional[FaceAnalyzer],
                 interval_s: float = 0.33) -> None:
        self._camera = camera
        self._analyzer = analyzer
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._recording = threading.Event()
        self._threads: list[threading.Thread] = []
        self._latest: Optional[tuple[np.ndarray, float, int]] = None
        self._latest_analysis: Optional[FrameAnalysis] = None
        self._analyses: list[FrameAnalysis] = []
        self._frames_captured = 0
        self._first_frame_time: Optional[float] = None
        self._recording_started_at = float("inf")
        self._error: Optional[str] = None
        self._analysis_error: Optional[str] = None
        self._opened = False
        self._released = False

    # ---------------- lifecycle ----------------
    def start(self) -> None:
        """Open the camera, wait for a first frame and start the threads."""
        self._camera.open()
        self._opened = True
        try:
            deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_S
            while True:
                ok, frame = self._camera.read()
                if ok and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
                    self._latest = (frame, time.perf_counter(), 1)
                    break
                if time.monotonic() > deadline:
                    raise DeviceUnavailableError(
                        "camera", "The camera opened but did not deliver any video frames."
                    )
                time.sleep(0.02)
        except BaseException:
            self.stop()
            raise
        self._threads = [
            threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True),
            threading.Thread(target=self._analysis_loop, name="face-analysis", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def begin_recording(self) -> None:
        with self._lock:
            self._recording_started_at = time.perf_counter()
        self._recording.set()

    def end_recording(self) -> None:
        self._recording.clear()

    def stop(self, join_timeout_s: float = 5.0) -> None:
        """Stop the threads and release the camera. Safe to call repeatedly."""
        self._recording.clear()
        self._stop.set()
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(join_timeout_s)
                if t.is_alive():
                    log.warning("Thread %s did not stop within %.1f s", t.name, join_timeout_s)
        with self._lock:
            if self._released or not self._opened:
                return
            self._released = True
        try:
            self._camera.release()
        except Exception:  # noqa: BLE001
            log.warning("Releasing the camera failed", exc_info=True)

    @property
    def released(self) -> bool:
        return self._released

    # ---------------- threads ----------------
    def _capture_loop(self) -> None:
        failures = 0
        seq = 1
        while not self._stop.is_set():
            try:
                ok, frame = self._camera.read()
            except Exception as exc:  # noqa: BLE001
                ok, frame = False, None
                log.warning("Camera read raised %s", type(exc).__name__)
            now = time.perf_counter()
            if ok and frame is not None:
                failures = 0
                seq += 1
                with self._lock:
                    self._latest = (frame, now, seq)
                    if self._recording.is_set():
                        self._frames_captured += 1
                        if self._first_frame_time is None:
                            self._first_frame_time = now
            else:
                failures += 1
                if failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    with self._lock:
                        self._error = "The camera stopped delivering video frames."
                    return
                time.sleep(0.01)

    def _analysis_loop(self) -> None:
        if self._analyzer is None:
            return
        last_seq = 0
        last_time = 0.0
        errors = 0
        while not self._stop.is_set():
            if not self._recording.is_set():
                time.sleep(0.01)
                continue
            with self._lock:
                latest = self._latest
                started = self._recording_started_at
            now = time.perf_counter()
            if (latest is None or latest[2] == last_seq or now - last_time < self._interval_s
                    or latest[1] < started):  # never analyse a countdown frame
                time.sleep(0.005)
                continue
            frame, frame_time, seq = latest
            last_seq, last_time = seq, now
            try:
                analysis = self._analyzer.analyze(frame, frame_time)
            except Exception as exc:  # noqa: BLE001 - recorded and reported
                errors += 1
                log.exception("Face analysis failed")
                with self._lock:
                    if self._analysis_error is None:
                        self._analysis_error = f"Face analysis failed ({type(exc).__name__})."
                if errors >= MAX_ANALYSIS_ERRORS and not self._analyses:
                    return
                continue
            with self._lock:
                self._analyses.append(analysis)
                self._latest_analysis = analysis

    # ---------------- accessors ----------------
    def snapshot(self) -> tuple[Optional[np.ndarray], Optional[FrameAnalysis]]:
        """Newest frame (do not modify it) and newest analysis, for display."""
        with self._lock:
            frame = self._latest[0] if self._latest else None
            return frame, self._latest_analysis

    def recent_analyses(self, n: int) -> list[FrameAnalysis]:
        with self._lock:
            return list(self._analyses[-n:])

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    @property
    def first_frame_time(self) -> Optional[float]:
        with self._lock:
            return self._first_frame_time

    def analyses(self) -> list[FrameAnalysis]:
        with self._lock:
            return list(self._analyses)

    def result(self) -> VisionResult:
        with self._lock:
            return aggregate_vision(
                list(self._analyses), self._frames_captured,
                self._error or self._analysis_error,
            )

    def timing_summary_ms(self) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for a in self.analyses():
            for k, v in a.timings_ms.items():
                totals.setdefault(k, []).append(v)
        return {f"{k}_mean_ms": sum(v) / len(v) for k, v in totals.items() if v}


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def aggregate_vision(analyses: list[FrameAnalysis], frames_captured: int,
                     error: Optional[str] = None) -> VisionResult:
    """Summarise per-frame analyses into percentages with explicit availability."""
    analysed = len(analyses)
    if analysed == 0 and error:
        return VisionResult(status=Status.FAILED, reason=error, frames_captured=frames_captured)
    if analysed < MIN_FRAMES_ANALYSED:
        reason = error or "Too few video frames were analysed."
        return VisionResult(status=Status.UNAVAILABLE, reason=reason,
                            frames_captured=frames_captured, frames_analysed=analysed)

    with_face = [a for a in analyses if a.face_detected]
    presence = _pct(len(with_face), analysed)
    eye_counts = Counter(a.eye for a in with_face if a.eye)
    head_counts = Counter(a.head for a in with_face if a.head)
    emotion_counts = Counter(a.emotion for a in with_face if a.emotion)
    common = dict(
        frames_captured=frames_captured,
        frames_analysed=analysed,
        frames_with_face=len(with_face),
        frames_multiple_faces=sum(1 for a in with_face if a.face_count > 1),
        face_presence=Metric.available("Face in view", presence, "%"),
        eye_counts=dict(eye_counts),
        head_counts=dict(head_counts),
        emotion_counts=dict(emotion_counts),
        dominant_emotion=emotion_counts.most_common(1)[0][0] if emotion_counts else None,
    )
    if presence < MIN_FACE_PRESENCE_PCT:
        reason = f"Your face was visible in only {presence:.0f}% of analysed frames."
        return VisionResult(
            status=Status.UNAVAILABLE, reason=reason,
            eye_contact=Metric.unavailable("Looking at camera", reason, "%"),
            head_facing=Metric.unavailable("Facing camera", reason, "%"),
            **common,
        )
    return VisionResult(
        status=Status.AVAILABLE,
        reason=error,
        eye_contact=Metric.available(
            "Looking at camera", _pct(eye_counts.get(LOOKING_CENTER, 0), len(with_face)), "%"),
        head_facing=Metric.available(
            "Facing camera", _pct(head_counts.get(CENTER, 0), len(with_face)), "%"),
        **common,
    )
