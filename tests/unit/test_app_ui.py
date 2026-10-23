"""Headless UI tests of frontend/app.py with Streamlit's AppTest and fakes."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from backend.ai import question_bank as qb
from backend.ai.answer_evaluator import AnswerEvaluator
from backend.audio.speech_analyzer import SpeechAnalyzer
from backend.core.models import ModelRegistry, ModelSpec
from backend.services import interview_service as svc
from backend.storage.history import LOCAL_OWNER, HistoryStore
from tests.conftest import (
    YAMNET_LABELS,
    FakeCamera,
    FakeEncoder,
    FakeFaceAnalyzer,
    FakeMicrophone,
    FakeWhisper,
    FakeYamnet,
)

APP = str(Path(__file__).resolve().parents[2] / "frontend" / "app.py")
TRANSCRIPT = ("<script>alert('x')</script> First, supervised learning uses labeled data "
              "because the model learns to predict outputs, for example 95% accuracy.")
EVIL_NAME = "<img src=x onerror=alert(1)>"


@pytest.fixture
def fake_app(monkeypatch, fast_settings, tmp_path):
    """Patch the service, device checks, models, logging and history location."""
    monkeypatch.setattr(svc, "MIN_DURATION_S", 0.5)
    monkeypatch.setenv("INTERVIEW_AI_COUNTDOWN", "0")
    # These tests cover the interview flow; sign-in is tested in test_auth_ui.py.
    monkeypatch.setenv("INTERVIEW_AI_REQUIRE_SIGN_IN", "false")
    history_path = tmp_path / "history.db"
    monkeypatch.setenv("INTERVIEW_AI_HISTORY_PATH", str(history_path))
    settings = replace(fast_settings, countdown_s=0)
    names = ["face_mesh", "emotion", "whisper", "sentence", "yamnet"]
    registry = ModelRegistry({n: ModelSpec(n.title(), object, "") for n in names})
    state = {"camera_ok": True, "history": history_path}
    real_service = svc.InterviewService

    def factory(settings_arg=None, **_kw):
        return real_service(
            settings=settings,
            registry=registry,
            camera_factory=lambda: FakeCamera(),
            microphone_factory=lambda: FakeMicrophone(),
            face_analyzer=FakeFaceAnalyzer(),
            speech_analyzer=SpeechAnalyzer(
                lambda: FakeWhisper(TRANSCRIPT, [{"start": 0, "end": 9.5}]),
                lambda: (FakeYamnet(0), YAMNET_LABELS)),
            evaluator=AnswerEvaluator(lambda: FakeEncoder()),
        )

    monkeypatch.setattr(svc, "InterviewService", lambda settings=None, **kw: factory(settings, **kw))
    # Keep test runs out of the real logs/app.log.
    monkeypatch.setattr("backend.logging_setup.configure_logging", lambda *a, **k: None)
    monkeypatch.setattr("backend.audio.recorder.check_microphone",
                        lambda *a, **k: (True, "Microphone ready (fake)."))

    def probe(cam):
        if state["camera_ok"]:
            return True, "Camera ready (fake).", np.zeros((48, 64, 3), np.uint8)
        return False, "The camera could not be opened.", None

    monkeypatch.setattr("backend.vision.capture.probe_camera", probe)
    st.cache_resource.clear()
    yield state
    st.cache_resource.clear()


def click(at: AppTest, label: str) -> AppTest:
    matches = [b for b in at.button if b.label == label]
    assert matches, f"No button {label!r}; have {[b.label for b in at.button]}"
    matches[0].click()
    return at.run()


def button(at: AppTest, label: str):
    return [b for b in at.button if b.label == label][0]


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [s.value for s in at.subheader] + \
            [t.value for t in at.title] + [c.value for c in at.caption]
    for kind in ("success", "warning", "info", "error"):
        parts += [e.value for e in getattr(at, kind)]
    return "\n".join(str(p) for p in parts)


def html_blocks(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown if getattr(m.proto, "allow_html", False)]


def to_question_page(at: AppTest, use_camera=True, name="") -> AppTest:
    at.run()
    click(at, "Start Interview")
    at.slider[0].set_value(10)
    if name:
        at.text_input[0].set_value(name)
    if not use_camera:
        at.toggle[0].set_value(False)
    click(at, "Continue")
    click(at, "Load models")
    click(at, "Check devices")
    click(at, "Continue")
    at.checkbox[0].check()
    return click(at, "Start interview")


def test_landing_page_brand_cta_and_limits(fake_app):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    page = texts(at)
    assert "HUMAN NEXT" in page and "Interview Potential" in page
    for feature in ("Live Behavioral Insights", "AI Response Evaluation", "Personalized Feedback"):
        assert feature in page
    assert "not a hiring decision system" in page
    assert button(at, "Start Interview").proto.type == "primary"


def test_full_flow_with_history_and_export(fake_app):
    at = to_question_page(AppTest.from_file(APP, default_timeout=60), name=EVIL_NAME)
    assert not at.exception
    page = texts(at)
    question = qb.get_questions("Software Engineer")[0]
    assert question.text in page
    for concept in question.concepts:           # the rubric is not shown before answering
        assert concept.name not in page
    assert "Camera standby" in page and "Pause/resume is not supported" in page

    at = click(at, "Start recording")   # renders the live console until the run ends
    at.run()                            # review page
    assert not at.exception
    page = texts(at)
    assert "PRACTICE SCORE" in page and "Why the answer scored this way" in page
    assert "Examples &amp; evidence" in page or "Examples & evidence" in page
    # User-controlled text is never rendered as raw HTML.
    for block in html_blocks(at):
        assert "<script>" not in block and "<img src=x" not in block
    assert "<script>" in page  # shown, but as plain escaped text in the transcript

    at = click(at, "Finish")
    assert "Interview results" in texts(at)
    assert [d.label for d in at.get("download_button")] == [
        "Download report (HTML → print to PDF)", "Download data (CSV)"]
    at = click(at, "Save to history")
    assert any("Saved." in s.value for s in at.success)
    saved = HistoryStore(fake_app["history"], LOCAL_OWNER).list()
    assert len(saved) == 1 and saved[0].candidate == EVIL_NAME
    # Transcripts are not stored unless requested.
    assert HistoryStore(fake_app["history"], LOCAL_OWNER).get(saved[0].session_id).answers[0].transcript is None

    at = click(at, "History")
    assert "Interview history" in texts(at)
    for block in html_blocks(at):
        assert "<img src=x" not in block
    at = click(at, "New interview")
    assert "Configure your interview" in texts(at)


def test_continue_disabled_until_ready(fake_app):
    at = AppTest.from_file(APP, default_timeout=30).run()
    click(at, "Start Interview")
    click(at, "Continue")
    assert button(at, "Continue").disabled  # models not loaded, devices not checked
    assert "Not checked" in texts(at)


def test_camera_failure_offers_audio_only(fake_app):
    fake_app["camera_ok"] = False
    at = AppTest.from_file(APP, default_timeout=30).run()
    click(at, "Start Interview")
    click(at, "Continue")
    click(at, "Load models")
    click(at, "Check devices")
    assert "The camera could not be opened." in texts(at)
    assert button(at, "Continue").disabled
    click(at, "Continue with audio only")
    click(at, "Check devices")
    assert not button(at, "Continue").disabled


def test_consent_required(fake_app):
    at = AppTest.from_file(APP, default_timeout=30).run()
    click(at, "Start Interview")
    click(at, "Continue")
    click(at, "Load models")
    click(at, "Check devices")
    click(at, "Continue")
    assert button(at, "Start interview").disabled


def test_difficulty_filter_changes_questions(fake_app):
    at = AppTest.from_file(APP, default_timeout=30).run()
    click(at, "Start Interview")
    at.selectbox[0].set_value("Machine Learning Engineer").run()
    at.selectbox[1].set_value("Medium").run()
    options = at.multiselect[0].options
    medium = [q for q in qb.get_questions("Machine Learning Engineer") if q.difficulty == "Medium"]
    assert len(options) == len(medium)


def test_empty_history_page(fake_app):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at = click(at, "View history")
    assert "No saved sessions yet" in texts(at)
