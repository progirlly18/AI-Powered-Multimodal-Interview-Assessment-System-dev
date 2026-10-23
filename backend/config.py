"""
Application settings read from environment variables.

All settings are optional; see .env.example for the list. The application
uses no API keys or other secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WHISPER_MODELS = ("tiny", "base", "small", "medium")


class ConfigError(ValueError):
    """Raised when an environment variable has an invalid value."""


@dataclass(frozen=True)
class Settings:
    camera_index: int = 0
    whisper_model: str = "base"
    # Seconds between vision analyses (the original code analysed every
    # 10th frame, i.e. about 3 times per second at 30 fps).
    vision_interval_s: float = 0.33
    # The expression model costs ~310 ms per call, so it runs less often.
    emotion_interval_s: float = 1.0
    # Maximum refresh rate of the live video in the browser.
    display_fps: float = 10.0
    countdown_s: int = 3
    keep_recordings: bool = False
    recordings_dir: Path = PROJECT_ROOT / "recordings"
    log_dir: Path = PROJECT_ROOT / "logs"
    # Local SQLite file for saved interview summaries (git-ignored).
    history_path: Path = PROJECT_ROOT / "instance" / "history.db"
    emotion_model_path: Path = PROJECT_ROOT / "backend" / "models" / "best_model.keras"
    # MediaPipe FaceLandmarker model (478 landmarks including irises).
    face_landmarker_path: Path = PROJECT_ROOT / "face_landmarker.task"
    language: str = "en"
    sentence_model: str = "all-MiniLM-L6-v2"
    yamnet_url: str = "https://tfhub.dev/google/yamnet/1"


def _int(env: Mapping[str, str], key: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{key} must be between {lo} and {hi}, got {value}")
    return value


def _float(env: Mapping[str, str], key: str, default: float, lo: float, hi: float) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{key} must be between {lo} and {hi}, got {value}")
    return value


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{key} must be true or false, got {raw!r}")


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Build Settings from ``env`` (defaults to ``os.environ``)."""
    env = os.environ if env is None else env

    whisper_model = env.get("INTERVIEW_AI_WHISPER_MODEL", "").strip() or "base"
    if whisper_model not in WHISPER_MODELS:
        raise ConfigError(
            f"INTERVIEW_AI_WHISPER_MODEL must be one of {', '.join(WHISPER_MODELS)}, "
            f"got {whisper_model!r}"
        )

    recordings_dir = env.get("INTERVIEW_AI_RECORDINGS_DIR", "").strip()
    history_path = env.get("INTERVIEW_AI_HISTORY_PATH", "").strip()

    return Settings(
        camera_index=_int(env, "INTERVIEW_AI_CAMERA_INDEX", 0, 0, 9),
        whisper_model=whisper_model,
        vision_interval_s=_float(env, "INTERVIEW_AI_VISION_INTERVAL", 0.33, 0.05, 2.0),
        emotion_interval_s=_float(env, "INTERVIEW_AI_EMOTION_INTERVAL", 1.0, 0.0, 10.0),
        display_fps=_float(env, "INTERVIEW_AI_DISPLAY_FPS", 10.0, 1.0, 30.0),
        countdown_s=_int(env, "INTERVIEW_AI_COUNTDOWN", 3, 0, 10),
        keep_recordings=_bool(env, "INTERVIEW_AI_KEEP_RECORDINGS", False),
        recordings_dir=Path(recordings_dir) if recordings_dir else Settings.recordings_dir,
        language=env.get("INTERVIEW_AI_LANGUAGE", "").strip() or "en",
        history_path=Path(history_path) if history_path else Settings.history_path,
    )
