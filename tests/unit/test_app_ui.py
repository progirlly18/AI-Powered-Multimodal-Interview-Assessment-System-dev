"""Headless UI tests of frontend/app.py with Streamlit's AppTest and fakes."""

from dataclasses import replace
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from backend.ai import question_bank as qb
from backend.ai.answer_evaluator import AnswerEvaluator
from backend.audio.speech_analyzer import SpeechAnalyzer
from backend.core.models import ModelRegistry, ModelSpec
from backend.services import interview_service as svc
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
TRANSCRIPT = "<script>alert('x')</script> Supervised learning uses labeled data to predict outputs."


@pytest.fixture
def fake_app(monkeypatch, fast_settings):
    """Patch the service, device checks and models used by the app."""
    monkeypatch.setattr(svc, "MIN_DURATION_S", 0.5)
    monkeypatch.setenv("INTERVIEW_AI_COUNTDOWN", "0")
    settings = replace(fast_settings, countdown_s=0)
    names = ["face_mesh", "emotion", "whisper", "sentence", "yamnet"]
    registry = ModelRegistry({n: ModelSpec(n.title(), object, "") for n in names})
    state = {"camera_ok": True}
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
    monkeypatch.setattr("backend.vision.capture.check_camera",
                        lambda cam: (state["camera_ok"], "Camera ready (fake)." if state["camera_ok"]
                                     else "The camera could not be opened."))
    st.cache_resource.clear()
    yield state
    st.cache_resource.clear()


def click(at: AppTest, label: str) -> AppTest:
    matches = [b for b in at.button if b.label == label]
    assert matches, f"No button {label!r}; have {[b.label for b in at.button]}"
    matches[0].click()
    return at.run()


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [s.value for s in at.subheader] + \
            [t.value for t in at.title] + [c.value for c in at.caption]
    for kind in ("success", "warning", "info", "error"):
        parts += [e.value for e in getattr(at, kind)]
    return "\n".join(str(p) for p in parts)


def to_question_page(at: AppTest, use_camera=True) -> AppTest:
    at.run()
    click(at, "Get started")
    at.slider[0].set_value(10)
    if not use_camera:
        at.toggle[0].set_value(False)
    click(at, "Continue")
    click(at, "Load models")
    click(at, "Check devices")
    click(at, "Continue")
    at.checkbox[0].check()
    return click(at, "Start interview")


def test_welcome_page_states_limitations(fake_app):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "not be used to make hiring decisions" in texts(at)


def test_full_flow(fake_app):
    at = to_question_page(AppTest.from_file(APP, default_timeout=60))
    assert not at.exception
    page = texts(at)
    question = qb.get_questions("Software Engineer")[0]
    assert question.text in page
    # The rubric is not shown before answering.
    for concept in question.concepts:
        assert concept.name not in page

    at = click(at, "Start recording")  # renders the live view until the run ends
    at.run()                             # review page
    assert not at.exception
    page = texts(at)
    assert [m.label for m in at.metric][0].startswith("Practice score")
    assert "Recording" not in [b.label for b in at.button]
    # Transcript text is escaped, never rendered as HTML.
    assert all(not getattr(m.proto, "allow_html", False) for m in at.markdown)
    assert "<script>" in page

    at = click(at, "Finish")
    assert "Interview summary" in texts(at)
    at = click(at, "Start a new interview")
    assert "Interview setup" in texts(at)


def test_continue_disabled_until_ready(fake_app):
    at = AppTest.from_file(APP, default_timeout=30).run()
    click(at, "Get started")
    click(at, "Continue")
    cont = [b for b in at.button if b.label == "Continue"][0]
    assert cont.disabled  # models not loaded, devices not checked
    assert "Not checked" in texts(at)


def test_camera_failure_offers_audio_only(fake_app):
    fake_app["camera_ok"] = False
    at = AppTest.from_file(APP, default_timeout=30).run()
    click(at, "Get started")
    click(at, "Continue")
    click(at, "Load models")
    click(at, "Check devices")
    assert "The camera could not be opened." in texts(at)
    assert [b for b in at.button if b.label == "Continue"][0].disabled
    click(at, "Continue with audio only")
    click(at, "Check devices")
    assert not [b for b in at.button if b.label == "Continue"][0].disabled


def test_consent_required(fake_app):
    at = AppTest.from_file(APP, default_timeout=30).run()
    click(at, "Get started")
    click(at, "Continue")
    click(at, "Load models")
    click(at, "Check devices")
    click(at, "Continue")
    assert [b for b in at.button if b.label == "Start interview"][0].disabled
