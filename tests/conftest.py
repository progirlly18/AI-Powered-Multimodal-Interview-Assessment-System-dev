"""Shared fakes for hardware and models. No test here needs a webcam,
microphone, network or downloaded model."""

from __future__ import annotations

import hashlib
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ai.question_bank import Concept, Question  # noqa: E402
from backend.audio.recorder import DeviceUnavailableError  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.vision.face_analysis import FrameAnalysis  # noqa: E402


# --------------------------------------------------
# Text encoder: deterministic bag-of-words embeddings
# --------------------------------------------------

class FakeEncoder:
    """Hashes words into a 256-d vector; similar wording -> similar vectors."""

    DIM = 256

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def encode(self, sentences, normalize_embeddings=True):
        self.calls += 1
        if self.fail:
            raise RuntimeError("encoder exploded")
        out = np.zeros((len(sentences), self.DIM))
        for i, s in enumerate(sentences):
            for word in re.findall(r"[a-z0-9]+", s.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % self.DIM] += 1.0
            norm = np.linalg.norm(out[i])
            if norm and normalize_embeddings:
                out[i] /= norm
        return out


# --------------------------------------------------
# Speech models
# --------------------------------------------------

class FakeWhisper:
    def __init__(self, text: str = "", segments=None, fail: bool = False) -> None:
        self.text = text
        self.segments = segments
        self.fail = fail
        self.received = None

    def transcribe(self, audio, **kwargs):
        self.received = (audio, kwargs)
        if self.fail:
            raise RuntimeError("whisper failed")
        return {"text": self.text, "segments": self.segments or []}


class FakeTensor:
    def __init__(self, array):
        self._a = np.asarray(array)

    def numpy(self):
        return self._a

    def __array__(self, dtype=None, copy=None):
        return self._a if dtype is None else self._a.astype(dtype)


class FakeYamnet:
    def __init__(self, label_index: int = 0, fail: bool = False) -> None:
        self.label_index = label_index
        self.fail = fail

    def __call__(self, waveform):
        if self.fail:
            raise RuntimeError("yamnet failed")
        scores = np.zeros((4, 3))
        scores[:, self.label_index] = 0.9
        return FakeTensor(scores), None, None


YAMNET_LABELS = ["Speech", "Music", "Silence"]


# --------------------------------------------------
# Devices
# --------------------------------------------------

class FakeCamera:
    """Delivers synthetic frames. Never touches real hardware."""

    def __init__(self, fail_open: bool = False, stop_after: Optional[int] = None,
                 no_frames: bool = False, shape=(48, 64, 3), frame_delay: float = 0.005) -> None:
        self.fail_open = fail_open
        self.stop_after = stop_after
        self.no_frames = no_frames
        self.shape = shape
        self.frame_delay = frame_delay
        self.opened = False
        self.release_count = 0
        self.reads = 0
        self._lock = threading.Lock()

    def open(self) -> None:
        if self.fail_open:
            raise DeviceUnavailableError("camera", "The camera could not be opened.")
        self.opened = True

    def read(self):
        time.sleep(self.frame_delay)
        with self._lock:
            if not self.opened or self.release_count:
                return False, None
            self.reads += 1
            if self.no_frames:
                return False, None
            if self.stop_after is not None and self.reads > self.stop_after:
                return False, None
            return True, np.full(self.shape, self.reads % 255, dtype=np.uint8)

    def release(self) -> None:
        with self._lock:
            self.release_count += 1
            self.opened = False


class FakeMicrophone:
    """Produces a tone while capturing (int16, 16 kHz)."""

    sample_rate = 16000

    def __init__(self, fail_open: bool = False, silent: bool = False,
                 fail_during: bool = False) -> None:
        self.fail_open = fail_open
        self.silent = silent
        self.fail_during = fail_during
        self.opened = False
        self.closed = 0
        self._capture_start: Optional[float] = None
        self._capture_end: Optional[float] = None
        self._first: Optional[float] = None

    def open(self) -> None:
        if self.fail_open:
            raise DeviceUnavailableError("microphone", "The microphone could not be opened.")
        self.opened = True

    def start_capture(self) -> None:
        self._capture_start = time.perf_counter()
        self._first = self._capture_start

    def stop_capture(self) -> None:
        self._capture_end = time.perf_counter()

    def close(self) -> None:
        self.closed += 1
        self.opened = False

    def audio(self) -> np.ndarray:
        if self._capture_start is None:
            return np.zeros(0, dtype=np.int16)
        end = self._capture_end or time.perf_counter()
        n = int((end - self._capture_start) * self.sample_rate)
        if self.silent:
            return np.zeros(n, dtype=np.int16)
        t = np.arange(n) / self.sample_rate
        return (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)

    @property
    def first_sample_time(self):
        return self._first

    @property
    def error(self):
        if self.fail_during and self._capture_start is not None:
            return "The microphone stream stopped unexpectedly."
        return None


class FakeFaceAnalyzer:
    """Returns scripted analyses (cycled)."""

    def __init__(self, script=None, fail: bool = False) -> None:
        self.script = script or [dict(face_detected=True, eye="Looking Center", head="Center",
                                      emotion="Neutral", emotion_confidence=0.6)]
        self.fail = fail
        self.calls = 0
        self.threads: set[str] = set()

    def analyze(self, frame, timestamp):
        self.threads.add(threading.current_thread().name)
        self.calls += 1
        if self.fail:
            raise RuntimeError("model crashed")
        spec = dict(self.script[(self.calls - 1) % len(self.script)])
        bbox = (1, 1, 10, 10) if spec.get("face_detected") else None
        return FrameAnalysis(timestamp=timestamp, bbox=bbox, timings_ms={"landmarks": 1.0}, **spec)


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def overfitting_question() -> Question:
    return Question(
        id="test-01",
        role="Machine Learning Engineer",
        category="technical",
        difficulty="Medium",
        text="What is overfitting and how can you prevent it?",
        concepts=(
            Concept("Generalization", ("fails to generalize",), ("generalize", "generalization")),
            Concept("Regularization", ("use regularization",), ("regularization", "L1", "L2")),
            Concept("Cross validation", ("use k fold cross validation",), ("cross validation",)),
        ),
    )


@pytest.fixture
def fast_settings(tmp_path) -> Settings:
    return Settings(
        countdown_s=0,
        vision_interval_s=0.02,
        display_fps=30,
        recordings_dir=tmp_path / "recordings",
        log_dir=tmp_path / "logs",
    )


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()
