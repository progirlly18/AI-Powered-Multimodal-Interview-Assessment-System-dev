"""Expression output validation, sampling, smoothing and multi-face handling."""

import numpy as np
import pytest

from backend.vision.capture import aggregate_vision
from backend.vision.emotion_detector import EMOTION_LABELS, validate_probabilities
from backend.vision.emotion_smoothing import latest_raw, smooth_emotion
from backend.vision.face_analysis import FaceAnalyzer, FrameAnalysis
from backend.vision.guidance import WARNING, guidance
from tests.unit.test_vision import make_face


def probs(label, p=0.7):
    v = np.full(7, (1 - p) / 6)
    v[EMOTION_LABELS.index(label)] = p
    return tuple(v)


def fa(t, label=None, p=0.7, faces=1):
    return FrameAnalysis(t, True, (0, 0, 1, 1), "Looking Center", "Center",
                         label, p if label else None, {}, faces,
                         probs(label, p) if label else None)


# ---------------- validation ----------------

def test_valid_probabilities_pass():
    assert validate_probabilities(probs("Happy")).shape == (7,)


@pytest.mark.parametrize("bad", [
    np.ones(6) / 6,                     # wrong number of classes
    np.full(7, np.nan),                 # NaN
    np.array([2, -1, 0, 0, 0, 0, 0.0]), # negative
    np.full(7, 0.5),                    # does not sum to 1
])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        validate_probabilities(bad)


# ---------------- smoothing ----------------

def test_smoothing_averages_recent_probabilities():
    analyses = [fa(0.0, "Happy", 0.9), fa(1.0, "Neutral", 0.6), fa(2.0, "Happy", 0.5),
                fa(2.3)]  # last frame: model did not run
    s = smooth_emotion(analyses, now=2.5, window_s=3.0)
    assert s.label == "Happy" and s.samples == 3
    assert s.probability == pytest.approx((0.9 + (1 - 0.6) / 6 + 0.5) / 3)


def test_smoothing_needs_enough_samples_and_respects_window():
    assert smooth_emotion([fa(0.0, "Sad")], now=0.1) is None
    old = [fa(0.0, "Sad"), fa(0.5, "Sad")]
    assert smooth_emotion(old, now=10.0, window_s=3.0) is None


def test_single_flicker_does_not_change_smoothed_label():
    analyses = [fa(t, "Neutral", 0.8) for t in (0.0, 1.0, 2.0)] + [fa(2.5, "Angry", 0.6)]
    assert smooth_emotion(analyses, now=2.6).label == "Neutral"


def test_latest_raw_skips_frames_without_prediction():
    analyses = [fa(0.0, "Sad"), fa(1.0)]
    assert latest_raw(analyses).emotion == "Sad"
    assert latest_raw([fa(0.0)]) is None


# ---------------- sampling ----------------

class Landmarker:
    def __init__(self, faces=1):
        self.faces = faces

    def detect_all(self, frame):
        return [make_face()] * self.faces


class CountingModel:
    calls = 0

    def predict(self, batch, verbose=0):
        CountingModel.calls += 1
        return np.array([probs("Neutral")])


def test_emotion_model_runs_at_most_once_per_interval(monkeypatch):
    monkeypatch.setattr("backend.vision.emotion_detector.preprocess", lambda f, b: np.zeros((1, 2)))
    clock = {"t": 0.0}
    CountingModel.calls = 0
    analyzer = FaceAnalyzer(lambda: Landmarker(), lambda: CountingModel(),
                            emotion_interval_s=1.0, clock=lambda: clock["t"])
    frame = np.zeros((300, 300, 3), np.uint8)
    results = []
    for t in (0.0, 0.33, 0.66, 1.0, 1.33, 2.1):
        clock["t"] = t
        results.append(analyzer.analyze(frame, t))
    assert CountingModel.calls == 3   # at t = 0, 1.0 and 2.1
    assert [r.emotion is not None for r in results] == [True, False, False, True, False, True]
    assert all(r.eye == "Looking Center" for r in results)   # gaze runs every frame


def test_emotion_output_is_validated(monkeypatch):
    monkeypatch.setattr("backend.vision.emotion_detector.preprocess", lambda f, b: np.zeros((1, 2)))

    class Broken:
        def predict(self, batch, verbose=0):
            return np.array([[0.5] * 7])

    analyzer = FaceAnalyzer(lambda: Landmarker(), lambda: Broken())
    with pytest.raises(ValueError):
        analyzer.analyze(np.zeros((300, 300, 3), np.uint8), 0.0)


# ---------------- multiple faces ----------------

def test_multiple_faces_are_counted_and_flagged():
    analyzer = FaceAnalyzer(lambda: Landmarker(faces=2))
    r = analyzer.analyze(np.zeros((300, 300, 3), np.uint8), 0.0)
    assert r.face_count == 2 and r.face_detected
    window = [fa(t, faces=2) for t in range(6)]
    level, text = guidance(window)
    assert level == WARNING and "More than one face" in text
    summary = aggregate_vision(window, frames_captured=60)
    assert summary.frames_multiple_faces == 6


def test_emotion_never_affects_score_inputs():
    happy = aggregate_vision([fa(t, "Happy") for t in range(5)], 50)
    sad = aggregate_vision([fa(t, "Sad") for t in range(5)], 50)
    assert happy.eye_contact == sad.eye_contact and happy.head_facing == sad.head_facing
