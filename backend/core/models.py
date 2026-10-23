"""
Lazy, cached model loading.

Heavy models are loaded on first use, once per process, and their load
times are recorded. Loading failures raise ``ModelLoadError`` with an
actionable message; they are never replaced by a fake model or a default
prediction. A failed load is not cached, so the user can retry.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from backend.config import Settings, load_settings
from backend.core.states import DeviceStatus

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """A model could not be loaded."""


@dataclass
class ModelInfo:
    name: str
    label: str
    status: DeviceStatus = DeviceStatus.NOT_CHECKED
    load_seconds: Optional[float] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ModelSpec:
    label: str
    loader: Callable[[], Any]
    help: str


def _help_for(exc: BaseException, default_help: str) -> str:
    """Advice that matches the kind of failure."""
    if isinstance(exc, ImportError):
        missing = getattr(exc, "name", None) or str(exc)
        return (f"A required Python package is missing or incompatible ({missing}). "
                "Reinstall the environment with requirements-lock.txt.")
    if isinstance(exc, FileNotFoundError):
        return f"{exc}."
    return default_help


class ModelRegistry:
    """Loads each model at most once and remembers how long it took."""

    def __init__(self, specs: dict[str, ModelSpec]) -> None:
        self._specs = specs
        self._models: dict[str, Any] = {}
        self._info = {name: ModelInfo(name, spec.label) for name, spec in specs.items()}
        self._locks = {name: threading.Lock() for name in specs}

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def info(self, name: str) -> ModelInfo:
        return self._info[name]

    def is_loaded(self, name: str) -> bool:
        return name in self._models

    def get(self, name: str) -> Any:
        if name not in self._specs:
            raise KeyError(f"Unknown model '{name}'")
        if name in self._models:
            return self._models[name]
        with self._locks[name]:
            if name in self._models:
                return self._models[name]
            spec = self._specs[name]
            info = self._info[name]
            info.status = DeviceStatus.INITIALIZING
            info.error = None
            start = time.perf_counter()
            try:
                model = spec.loader()
            except Exception as exc:  # noqa: BLE001 - reported, not hidden
                info.status = DeviceStatus.FAILED
                info.load_seconds = time.perf_counter() - start
                info.error = f"{type(exc).__name__}: {exc}"
                log.exception("Loading %s failed", name)
                raise ModelLoadError(
                    f"{spec.label} could not be loaded. {_help_for(exc, spec.help)}"
                ) from exc
            info.load_seconds = time.perf_counter() - start
            info.status = DeviceStatus.READY
            log.info("Loaded %s in %.1f s", name, info.load_seconds)
            self._models[name] = model
            return model


# --------------------------------------------------
# Real model loaders (imports are local so that the
# rest of the backend can be imported without them)
# --------------------------------------------------

def _load_emotion_model(settings: Settings) -> Any:
    path = settings.emotion_model_path
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    from tensorflow.keras.models import load_model

    return load_model(str(path))


def _load_face_mesh(settings: Settings) -> Any:
    from backend.vision.landmarks import FaceLandmarker

    # Two faces so a second person in view can be detected and flagged;
    # analysis always uses the first face MediaPipe returns.
    return FaceLandmarker(settings.face_landmarker_path, max_faces=2)


def _load_whisper(settings: Settings) -> Any:
    import whisper

    return whisper.load_model(settings.whisper_model)


def _load_yamnet(settings: Settings) -> Any:
    import tensorflow_hub as hub

    model = hub.load(settings.yamnet_url)
    import csv

    class_map = model.class_map_path().numpy().decode("utf-8")
    # Class names such as "Child speech, kid speaking" contain commas and are
    # quoted, so the file must be parsed as CSV (the original split(",")
    # truncated them).
    with open(class_map, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    labels = [row[2] for row in rows[1:]]
    return model, labels


def _load_sentence_model(settings: Settings) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.sentence_model)


def build_default_registry(settings: Optional[Settings] = None) -> ModelRegistry:
    s = settings or load_settings()
    first_run = (
        "The first run downloads it from the internet; check your connection, "
        "then retry."
    )
    return ModelRegistry(
        {
            "face_mesh": ModelSpec(
                "Face landmark detector (MediaPipe)",
                lambda: _load_face_mesh(s),
                f"Expected file: {s.face_landmarker_path}; check that mediapipe is installed.",
            ),
            "emotion": ModelSpec(
                "Facial expression model (EfficientNet)",
                lambda: _load_emotion_model(s),
                f"Expected file: {s.emotion_model_path}.",
            ),
            "whisper": ModelSpec(
                f"Speech-to-text (Whisper {s.whisper_model})",
                lambda: _load_whisper(s), first_run,
            ),
            "yamnet": ModelSpec(
                "Background sound classifier (YAMNet)",
                lambda: _load_yamnet(s), first_run,
            ),
            "sentence": ModelSpec(
                f"Answer evaluation model ({s.sentence_model})",
                lambda: _load_sentence_model(s), first_run,
            ),
        }
    )


_default_registry: Optional[ModelRegistry] = None
_default_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    """Process-wide registry used by the application."""
    global _default_registry
    with _default_lock:
        if _default_registry is None:
            _default_registry = build_default_registry()
        return _default_registry
