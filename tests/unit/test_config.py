"""Environment settings and the Streamlit security configuration."""

import tomllib
from pathlib import Path

import pytest

from backend.config import ConfigError, load_settings

ROOT = Path(__file__).resolve().parents[2]


def test_defaults():
    s = load_settings({})
    assert s.camera_index == 0 and s.whisper_model == "base" and s.countdown_s == 3
    assert s.keep_recordings is False
    assert s.emotion_model_path.is_absolute()  # no working-directory dependence
    assert s.face_landmarker_path.is_absolute()


def test_valid_overrides(tmp_path):
    s = load_settings({
        "INTERVIEW_AI_CAMERA_INDEX": "1",
        "INTERVIEW_AI_WHISPER_MODEL": "tiny",
        "INTERVIEW_AI_COUNTDOWN": "5",
        "INTERVIEW_AI_KEEP_RECORDINGS": "yes",
        "INTERVIEW_AI_RECORDINGS_DIR": str(tmp_path),
        "INTERVIEW_AI_DISPLAY_FPS": "15",
    })
    assert (s.camera_index, s.whisper_model, s.countdown_s, s.keep_recordings) == (1, "tiny", 5, True)
    assert s.recordings_dir == tmp_path and s.display_fps == 15


@pytest.mark.parametrize("env", [
    {"INTERVIEW_AI_CAMERA_INDEX": "abc"},
    {"INTERVIEW_AI_CAMERA_INDEX": "42"},
    {"INTERVIEW_AI_WHISPER_MODEL": "huge"},
    {"INTERVIEW_AI_COUNTDOWN": "-1"},
    {"INTERVIEW_AI_KEEP_RECORDINGS": "maybe"},
    {"INTERVIEW_AI_DISPLAY_FPS": "0"},
    {"INTERVIEW_AI_VISION_INTERVAL": "nan"},
])
def test_invalid_values_raise(env):
    with pytest.raises(ConfigError):
        load_settings(env)


def test_streamlit_config_is_secure():
    with open(ROOT / ".streamlit" / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    server = cfg["server"]
    assert server["enableXsrfProtection"] is True
    assert server["enableCORS"] is True
    assert server["address"] == "localhost"
    assert server.get("fileWatcherType", "auto") != "poll"
    assert cfg["browser"]["gatherUsageStats"] is False


def test_env_example_has_no_secrets():
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assignments = [l for l in lines if "=" in l and not l.lstrip().startswith("#")]
    assert assignments, "expected placeholder assignments"
    for line in assignments:
        name = line.split("=", 1)[0].upper()
        assert name.startswith("INTERVIEW_AI_")
        for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"):
            assert word not in name


def test_gitignore_covers_local_data():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in ("logs/", "recordings/", ".venv", ".env", "benchmarks/results/"):
        assert entry in text
