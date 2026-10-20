"""Gaze/head rules, bounding boxes, aggregation, guidance and VisionSession."""

import threading
import time

import numpy as np
import pytest

from backend.audio.recorder import DeviceUnavailableError
from backend.core.results import Status
from backend.vision.capture import (
    MIN_FACE_PRESENCE_PCT,
    VisionSession,
    aggregate_vision,
    check_camera,
)
from backend.vision.emotion_detector import face_bbox
from backend.vision.eye_contact import eye_direction_from_landmarks, gaze_ratio
from backend.vision.face_analysis import FaceAnalyzer, FrameAnalysis
from backend.vision.guidance import OK, WAITING, WARNING, guidance, latest_labels
from backend.vision.head_pose import head_pose_from_landmarks
from tests.conftest import FakeCamera, FakeFaceAnalyzer


def make_face(iris_ratio=0.5, nose_dx=0.0, nose_dy=0.0):
    """478 synthetic landmarks with controllable iris and nose positions."""
    face = [[100, 100] for _ in range(478)]
    face[33], face[133] = [100, 100], [140, 100]      # left eye corners
    face[362], face[263] = [160, 100], [200, 100]     # right eye corners
    face[468] = [100 + 40 * iris_ratio, 100]
    face[473] = [160 + 40 * iris_ratio, 100]
    face[234], face[454] = [80, 150], [220, 150]       # cheeks -> centre x = 150
    face[10], face[152] = [150, 60], [150, 240]        # forehead/chin -> centre y = 150
    face[1] = [150 + nose_dx, 150 + nose_dy]
    return face


# ---------------- pure rules ----------------

@pytest.mark.parametrize("ratio,label", [
    (0.5, "Looking Center"), (0.36, "Looking Center"), (0.64, "Looking Center"),
    (0.2, "Looking Right"), (0.8, "Looking Left"),
])
def test_eye_direction(ratio, label):
    face = make_face(iris_ratio=ratio)
    assert gaze_ratio(face) == pytest.approx(ratio, abs=1e-3)
    assert eye_direction_from_landmarks(face) == label


def test_eye_direction_requires_iris_landmarks():
    with pytest.raises(ValueError):
        eye_direction_from_landmarks(make_face()[:468])


@pytest.mark.parametrize("dx,dy,label", [
    (0, 0, "Center"), (-20, 0, "Looking Right"), (20, 0, "Looking Left"),
    (0, -15, "Looking Up"), (0, 20, "Looking Down"), (15, 15, "Center"),
])
def test_head_pose(dx, dy, label):
    assert head_pose_from_landmarks(make_face(nose_dx=dx, nose_dy=dy)) == label


def test_face_bbox_clips_to_frame():
    assert face_bbox([[5, 5], [30, 40]], (50, 50, 3)) == (0, 0, 50, 50)
    assert face_bbox([[20, 20], [30, 30]], (100, 100, 3), margin=5) == (15, 15, 35, 35)
    assert face_bbox([], (10, 10, 3)) is None


def test_face_analyzer_single_landmark_pass():
    class CountingLandmarker:
        calls = 0
        def detect(self, frame):
            CountingLandmarker.calls += 1
            return make_face()
    analyzer = FaceAnalyzer(lambda: CountingLandmarker(), emotion_model_provider=None)
    result = analyzer.analyze(np.zeros((300, 300, 3), np.uint8), 1.0)
    assert CountingLandmarker.calls == 1  # was three detections per frame
    assert result.face_detected and result.eye == "Looking Center" and result.head == "Center"
    assert result.emotion is None


def test_face_analyzer_no_face():
    class NoFace:
        def detect(self, frame):
            return None
    r = FaceAnalyzer(lambda: NoFace()).analyze(np.zeros((10, 10, 3), np.uint8), 0.0)
    assert not r.face_detected and r.eye is None and r.bbox is None


# ---------------- aggregation ----------------

def fa(face=True, eye="Looking Center", head="Center", emotion="Neutral"):
    return FrameAnalysis(0.0, face, (0, 0, 1, 1) if face else None,
                         eye if face else None, head if face else None, emotion if face else None, 0.5)


def test_aggregate_percentages():
    analyses = [fa(), fa(), fa(eye="Looking Left"), fa(head="Looking Down")]
    r = aggregate_vision(analyses, frames_captured=40)
    assert r.status is Status.AVAILABLE
    assert r.face_presence.value == 100
    assert r.eye_contact.value == 75 and r.head_facing.value == 75
    assert r.dominant_emotion == "Neutral"


def test_no_face_is_unavailable_not_poor_eye_contact():
    analyses = [fa(face=False)] * 8 + [fa()] * 2
    r = aggregate_vision(analyses, frames_captured=100)
    assert r.status is Status.UNAVAILABLE
    assert r.face_presence.value == 20
    assert not r.eye_contact.is_available and not r.head_facing.is_available
    assert "20%" in r.reason


def test_presence_threshold_boundary():
    n_face = int(MIN_FACE_PRESENCE_PCT / 10)
    analyses = [fa()] * n_face + [fa(face=False)] * (10 - n_face)
    assert aggregate_vision(analyses, 10).status is Status.AVAILABLE


def test_too_few_frames_and_errors():
    assert aggregate_vision([fa()], 5).status is Status.UNAVAILABLE
    failed = aggregate_vision([], 0, error="Face analysis failed (RuntimeError).")
    assert failed.status is Status.FAILED and "RuntimeError" in failed.reason


# ---------------- guidance ----------------

def test_guidance_smoothing():
    assert guidance([])[0] == WAITING
    # One stray frame does not trigger a warning.
    assert guidance([fa()] * 5 + [fa(eye="Looking Left")])[0] == OK
    assert guidance([fa(eye="Looking Left")] * 4 + [fa()] * 2) == (WARNING, "Try to look towards the camera while you speak.")
    assert "not clearly visible" in guidance([fa(face=False)] * 6)[1]
    assert "head" in guidance([fa(head="Looking Down")] * 6)[1]


def test_latest_labels():
    assert latest_labels(None)["face"] == "—"
    assert latest_labels(fa(face=False))["face"] == "Not detected"
    labels = latest_labels(fa())
    assert labels["eye"] == "Looking Center" and labels["emotion"] == "Neutral (50%)"


# ---------------- VisionSession ----------------

def run_session(camera, analyzer, seconds=0.3):
    s = VisionSession(camera, analyzer, interval_s=0.02)
    s.start()
    s.begin_recording()
    time.sleep(seconds)
    s.end_recording()
    s.stop()
    return s


def test_session_analyses_in_background_and_releases():
    camera, analyzer = FakeCamera(), FakeFaceAnalyzer()
    s = run_session(camera, analyzer)
    assert camera.release_count == 1 and s.released
    assert analyzer.calls >= 3
    assert analyzer.threads == {"face-analysis"}   # never in the capture thread
    r = s.result()
    assert r.status is Status.AVAILABLE and r.frames_analysed == analyzer.calls
    assert r.frames_captured > r.frames_analysed  # only the latest frame is analysed


def test_slow_analysis_does_not_block_capture():
    class SlowAnalyzer(FakeFaceAnalyzer):
        def analyze(self, frame, timestamp):
            time.sleep(0.1)
            return super().analyze(frame, timestamp)
    camera = FakeCamera()
    s = run_session(camera, SlowAnalyzer(), seconds=0.5)
    r = s.result()
    assert r.frames_captured >= 3 * r.frames_analysed


def test_camera_open_failure():
    camera = FakeCamera(fail_open=True)
    s = VisionSession(camera, FakeFaceAnalyzer())
    with pytest.raises(DeviceUnavailableError):
        s.start()
    assert camera.release_count == 0  # nothing was opened


def test_camera_without_frames(monkeypatch):
    monkeypatch.setattr("backend.vision.capture.FIRST_FRAME_TIMEOUT_S", 0.1)
    camera = FakeCamera(no_frames=True)
    with pytest.raises(DeviceUnavailableError, match="did not deliver"):
        VisionSession(camera, FakeFaceAnalyzer()).start()
    assert camera.release_count == 1


def test_camera_disconnect_is_reported():
    camera = FakeCamera(stop_after=5, frame_delay=0.001)
    s = VisionSession(camera, FakeFaceAnalyzer(), interval_s=0.02)
    s.start()
    s.begin_recording()
    deadline = time.monotonic() + 2
    while s.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    s.stop()
    assert s.error == "The camera stopped delivering video frames."
    assert camera.release_count == 1


def test_analysis_errors_are_reported():
    s = run_session(FakeCamera(), FakeFaceAnalyzer(fail=True))
    r = s.result()
    assert r.status is Status.FAILED and "RuntimeError" in r.reason


def test_stop_is_idempotent_and_thread_safe():
    camera = FakeCamera()
    s = VisionSession(camera, FakeFaceAnalyzer(), interval_s=0.02)
    s.start()
    threads = [threading.Thread(target=s.stop) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    s.stop()
    assert camera.release_count == 1


def test_nothing_analysed_before_recording():
    analyzer = FakeFaceAnalyzer()
    s = VisionSession(FakeCamera(), analyzer, interval_s=0.01)
    s.start()
    time.sleep(0.15)
    assert analyzer.calls == 0   # countdown frames are shown but not analysed
    s.stop()


def test_only_frames_after_recording_start_are_analysed():
    analyzer = FakeFaceAnalyzer()
    s = VisionSession(FakeCamera(), analyzer, interval_s=0.01)
    s.start()
    time.sleep(0.05)
    s.begin_recording()
    started = s._recording_started_at
    time.sleep(0.2)
    s.stop()
    assert s.analyses() and all(a.timestamp >= started for a in s.analyses())


def test_check_camera():
    camera = FakeCamera()
    ok, msg = check_camera(camera)
    assert ok and "64x48" in msg and camera.release_count == 1
    ok, msg = check_camera(FakeCamera(fail_open=True))
    assert not ok and "could not be opened" in msg
