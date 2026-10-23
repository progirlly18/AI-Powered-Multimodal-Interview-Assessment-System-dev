"""Sending sign-in codes by email with the standard-library ``smtplib``."""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Protocol

from backend.auth.config import AuthSettings
from backend.auth.email_rules import mask_email

log = logging.getLogger(__name__)
TIMEOUT_S = 15


class MailerError(RuntimeError):
    """The code could not be sent; ``str(exc)`` is safe to show."""


class Mailer(Protocol):
    def send_code(self, to: str, code: str, valid_minutes: int) -> None: ...


def build_message(sender: str, to: str, code: str, valid_minutes: int) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = f"Your HUMAN NEXT sign-in code: {code}"
    msg["From"] = sender
    msg["To"] = to
    msg.set_content(
        f"Your HUMAN NEXT sign-in code is:\n\n    {code}\n\n"
        f"It expires in {valid_minutes} minutes and can be used once.\n\n"
        "If you did not try to sign in, you can ignore this email; "
        "nobody can sign in without this code.\n"
    )
    return msg


class SmtpMailer:
    def __init__(self, settings: AuthSettings) -> None:
        self._s = settings

    def send_code(self, to: str, code: str, valid_minutes: int) -> None:
        s = self._s
        if not s.smtp_configured:
            raise MailerError("Email sending is not configured on this computer.")
        msg = build_message(s.sender or "", to, code, valid_minutes)
        try:
            if s.smtp_security == "ssl":
                server: smtplib.SMTP = smtplib.SMTP_SSL(
                    s.smtp_host, s.smtp_port, timeout=TIMEOUT_S, context=ssl.create_default_context())
            else:
                server = smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=TIMEOUT_S)
            with server:
                if s.smtp_security == "starttls":
                    server.starttls(context=ssl.create_default_context())
                if s.smtp_username and s.smtp_password:
                    server.login(s.smtp_username, s.smtp_password)
                refused = server.send_message(msg)
        except smtplib.SMTPAuthenticationError as exc:
            log.error("SMTP login failed (%s)", exc.smtp_code)
            raise MailerError("The email account used to send codes rejected its password. "
                              "Check the SMTP settings.") from exc
        except smtplib.SMTPRecipientsRefused as exc:
            log.warning("SMTP refused recipient %s", mask_email(to))
            raise MailerError("The mail server refused that address. Check it and try again.") from exc
        except (smtplib.SMTPException, OSError) as exc:
            log.error("Sending sign-in code failed: %s", type(exc).__name__)
            raise MailerError("The code could not be sent right now. Check the internet "
                              "connection and try again.") from exc
        if refused:
            raise MailerError("The mail server refused that address. Check it and try again.")
        log.info("Sign-in code sent to %s", mask_email(to))
