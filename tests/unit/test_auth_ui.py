"""Headless UI tests of the sign-in gate (fake mailer, temporary databases)."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from backend.auth import service as auth_service_module
from backend.auth.email_rules import load_disposable_domains
from backend.auth.service import AuthService
from backend.auth.store import AuthStore
from backend.storage.history import HistoryStore
from backend.storage.records import build_session_record
from tests.unit.test_auth import FakeMailer

APP = str(Path(__file__).resolve().parents[2] / "frontend" / "app.py")


@pytest.fixture
def auth_app(monkeypatch, tmp_path):
    monkeypatch.setenv("INTERVIEW_AI_REQUIRE_SIGN_IN", "true")
    monkeypatch.setenv("INTERVIEW_AI_SMTP_HOST", "smtp.test.invalid")
    monkeypatch.setenv("INTERVIEW_AI_SMTP_USERNAME", "sender@uni.edu")
    monkeypatch.setenv("INTERVIEW_AI_HISTORY_PATH", str(tmp_path / "history.db"))
    monkeypatch.setattr("backend.logging_setup.configure_logging", lambda *a, **k: None)
    mailer = FakeMailer()

    def build(settings):
        return AuthService(settings, AuthStore(tmp_path / "auth.db"), mailer,
                           load_disposable_domains())

    monkeypatch.setattr(auth_service_module, "build_auth_service", build)
    st.cache_resource.clear()
    yield mailer, tmp_path
    st.cache_resource.clear()


def page(at: AppTest) -> str:
    parts = [m.value for m in at.markdown]
    for kind in ("success", "error", "warning", "info"):
        parts += [e.value for e in getattr(at, kind)]
    return "\n".join(parts)


def submit(at: AppTest, label: str) -> AppTest:
    [b for b in at.button if b.label == label][0].click()
    return at.run()


def sign_in(at: AppTest, mailer, email="ada@uni.edu") -> AppTest:
    at.text_input[0].set_value(email)
    at = submit(at, "Send sign-in code")
    at.text_input[0].set_value(mailer.last_code)
    return submit(at, "Verify and sign in")


def test_app_is_locked_until_signed_in(auth_app):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    text = page(at)
    assert "Sign in to HUMAN NEXT" in text
    assert "Decode Your" not in text            # landing page not rendered
    assert not [b for b in at.button if b.label == "Start Interview"]


def test_fake_emails_are_refused_without_sending(auth_app):
    mailer, _ = auth_app
    at = AppTest.from_file(APP, default_timeout=30).run()
    for bad, message in [("not-an-email", "doesn't look like"),
                         ("someone@mailinator.com", "Disposable"),
                         ("test@example.com", "reserved")]:
        at.text_input[0].set_value(bad)
        at = submit(at, "Send sign-in code")
        assert message in page(at)
    assert mailer.sent == []


def test_full_sign_in_and_sign_out(auth_app):
    mailer, _ = auth_app
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].set_value("Ada@Uni.edu")
    at = submit(at, "Send sign-in code")
    assert mailer.sent[0][0] == "ada@uni.edu"
    assert "We sent a 6-digit code to ada@uni.edu" in page(at)
    assert mailer.last_code not in page(at)      # the code is never shown on screen

    at.text_input[0].set_value("000000" if mailer.last_code != "000000" else "111111")
    at = submit(at, "Verify and sign in")
    assert "not correct" in page(at)

    at.text_input[0].set_value(mailer.last_code)
    at = submit(at, "Verify and sign in")
    assert not at.exception
    assert "Decode Your" in page(at) and "ada@uni.edu" in page(at)

    at = submit(at, "Sign out")
    assert "Sign in to HUMAN NEXT" in page(at) and "signed out" in page(at)


def test_history_shows_only_own_sessions(auth_app):
    mailer, tmp = auth_app
    other = build_session_record("bob-1", "2026-10-23 09:00", "AI Engineer", "Bob", [])
    HistoryStore(tmp / "history.db", "bob@uni.edu").save(other)
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), mailer)
    at = submit(at, "View history")
    assert "No saved sessions yet" in page(at)   # Bob's session is not visible to Ada


def test_not_configured_shows_setup_and_blocks(auth_app, monkeypatch):
    monkeypatch.delenv("INTERVIEW_AI_SMTP_HOST")
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()
    text = page(at)
    assert "Email sending is not set up" in text
    assert not at.text_input            # no way to proceed
    assert "Decode Your" not in text
