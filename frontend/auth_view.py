"""
Sign-in page: email -> emailed 6-digit code -> signed in.

The page never reveals whether an address already has an account, never
shows or logs codes, and refuses to proceed when email sending is not
configured (it does not fall back to letting people in).
"""

from __future__ import annotations

import streamlit as st

from backend.auth.email_rules import EmailRejected
from backend.auth.mailer import MailerError
from backend.auth.service import AuthError, AuthService
from backend.auth.store import AuthStoreError
from frontend.components import brand_bar, bullet_list, card, chip, para, show

MAX_REQUESTS_PER_BROWSER_SESSION = 10


def _setup_help() -> str:
    return card(
        para("Sign-in sends a one-time code by email, so this computer needs an email account "
             "to send from. Ask the person running the app to configure it.")
        + "<p>Set these environment variables (or the <code>[auth]</code> section of "
          "<code>.streamlit/secrets.toml</code>), then restart the app:</p>"
        + bullet_list([
            "INTERVIEW_AI_SMTP_HOST — e.g. smtp.gmail.com",
            "INTERVIEW_AI_SMTP_USERNAME — the sending address",
            "INTERVIEW_AI_SMTP_PASSWORD — an app password (never your normal password)",
        ])
        + para("Setup guide: docs/AUTH.md. To run without sign-in on a private computer, set "
               "INTERVIEW_AI_REQUIRE_SIGN_IN=false."),
        label="Sign-in not configured", title="Email sending is not set up")


def _explainer() -> str:
    return card(
        bullet_list([
            "Enter your email and we send a 6-digit code to that inbox.",
            "Only someone who can read the inbox can sign in, so made-up or mistyped "
            "addresses can't be used. Throwaway-mail domains are refused.",
            "Codes expire after 10 minutes, work once, and allow 5 tries.",
            "No password is created or stored.",
            "Stored about you: your email and sign-in times. Your saved interview history "
            "is visible only to you.",
        ]),
        label="How sign-in works", title="Passwordless and verified")


def render_sign_in(auth: AuthService) -> None:
    """Draw the sign-in page. On success sets ``st.session_state.user`` and reruns."""
    ss = st.session_state
    ss.setdefault("auth_email", None)
    ss.setdefault("auth_requests", 0)
    show(brand_bar([chip("Secure sign-in", "measured")]))
    left, right = st.columns([3, 2], gap="large")
    with right:
        show(_explainer())
    with left:
        st.markdown("## Sign in to HUMAN NEXT")
        if not auth.settings.smtp_configured:
            show(_setup_help())
            return
        notice = ss.pop("auth_notice", None)
        if notice:
            st.success(notice)

        if not ss.auth_email:
            with st.form("auth_email_form", clear_on_submit=False):
                email = st.text_input("Email address", placeholder="you@university.edu",
                                      max_chars=254)
                submitted = st.form_submit_button("Send sign-in code", type="primary")
            if submitted:
                _request(auth, email)
            return

        show(card(para(f"We sent a 6-digit code to {ss.auth_email}. It expires in "
                       f"{auth.settings.code_ttl_s // 60} minutes. Check your spam folder if "
                       "it doesn't arrive within a minute."),
                  label="Check your inbox"))
        with st.form("auth_code_form", clear_on_submit=True):
            code = st.text_input("6-digit code", max_chars=7, placeholder="123456")
            verify = st.form_submit_button("Verify and sign in", type="primary")
        if verify:
            try:
                ss.user = auth.verify_code(ss.auth_email, code)
            except AuthError as exc:
                st.error(str(exc))
            except AuthStoreError:
                st.error("Sign-in is temporarily unavailable. Please try again.")
            else:
                ss.auth_email = None
                ss.auth_requests = 0
                st.rerun()
        c1, c2 = st.columns(2)
        if c1.button("Resend code", width="stretch"):
            _request(auth, ss.auth_email)
        if c2.button("Use a different email", width="stretch"):
            ss.auth_email = None
            st.rerun()


def _request(auth: AuthService, email: str) -> None:
    ss = st.session_state
    if ss.auth_requests >= MAX_REQUESTS_PER_BROWSER_SESSION:
        st.error("Too many code requests from this browser. Close the tab and try again later.")
        return
    try:
        normalised = auth.request_code(email)
    except (EmailRejected, AuthError, MailerError) as exc:
        st.error(str(exc))
        return
    except AuthStoreError:
        st.error("Sign-in is temporarily unavailable. Please try again.")
        return
    ss.auth_requests += 1
    ss.auth_email = normalised
    ss.auth_notice = f"Code sent to {normalised}."
    st.rerun()


def signed_in_chip(email: str) -> str:
    return chip(email, "ok", dot=True)  # chip() escapes the text
