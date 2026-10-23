"""
Passwordless sign-in: email -> one-time code by email -> verified session.

Why this blocks fake emails: a code is only ever delivered to the inbox of
the address entered, so only someone who can read that inbox can sign in.
The format, reserved-domain and disposable-domain checks run first so
obvious fakes are rejected without sending anything.

Limits (AuthSettings): a code lives 10 minutes, allows 5 wrong guesses,
can be resent after 60 s, and at most 5 codes per address per hour.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from backend.auth.codes import clean_code, code_matches, hash_code, new_code, new_salt
from backend.auth.config import AuthSettings
from backend.auth.email_rules import (
    EmailRejected,
    load_disposable_domains,
    mask_email,
    normalize_email,
    validate_email,
)
from backend.auth.mailer import Mailer, MailerError, SmtpMailer
from backend.auth.store import AuthStore, PendingCode, User

log = logging.getLogger(__name__)
HOUR_S = 3600


class AuthError(Exception):
    """A sign-in step failed; ``str(exc)`` is safe to show to the user."""


class RateLimited(AuthError):
    def __init__(self, message: str, retry_after_s: int) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class InvalidCode(AuthError):
    def __init__(self, message: str, attempts_left: int) -> None:
        super().__init__(message)
        self.attempts_left = attempts_left


@dataclass(frozen=True)
class SignedInUser:
    email: str
    signed_in_at: float


class AuthService:
    def __init__(self, settings: AuthSettings, store: AuthStore, mailer: Mailer,
                 disposable: Iterable[str] = (), clock: Callable[[], float] = time.time) -> None:
        self.settings = settings
        self._store = store
        self._mailer = mailer
        self._disposable = frozenset(disposable)
        self._clock = clock

    # ---------------- step 1: request a code ----------------
    def request_code(self, raw_email: str) -> str:
        """Validate the address, enforce limits and email a fresh code.

        Returns the normalised address. Raises EmailRejected, RateLimited,
        MailerError or AuthError.
        """
        email = validate_email(raw_email, self._disposable,
                               self.settings.allowed_domains or None)
        now = self._clock()
        existing = self._store.get_code(email)
        window_start, sends = now, 0
        if existing is not None:
            wait = int(existing.sent_at + self.settings.resend_cooldown_s - now)
            if wait > 0:
                raise RateLimited(f"Please wait {wait} s before requesting another code.", wait)
            if now - existing.window_start < HOUR_S:
                window_start, sends = existing.window_start, existing.sends_in_window
            if sends >= self.settings.max_codes_per_hour:
                retry = int(window_start + HOUR_S - now)
                raise RateLimited("Too many codes requested for this address. "
                                  f"Try again in {max(1, retry // 60)} min.", retry)

        code, salt = new_code(), new_salt()
        pending = PendingCode(email, hash_code(code, salt), salt,
                              now + self.settings.code_ttl_s, 0, now, window_start, sends + 1)
        self._store.put_code(pending)
        try:
            self._mailer.send_code(email, code, self.settings.code_ttl_s // 60)
        except MailerError:
            # The code never reached anyone: make it unusable, keep the counters.
            self._store.void_code(email)
            raise
        log.info("Sign-in code issued for %s", mask_email(email))
        return email

    # ---------------- step 2: verify the code ----------------
    def verify_code(self, raw_email: str, raw_code: str) -> SignedInUser:
        email = normalize_email(raw_email)
        code = clean_code(raw_code)
        if not code:
            raise AuthError("Enter the 6-digit code from the email.")
        pending = self._store.get_code(email)
        now = self._clock()
        if pending is None or not pending.code_hash:
            raise AuthError("There is no active code for this address. Request a new one.")
        if now > pending.expires_at:
            self._store.void_code(email)
            raise AuthError("This code has expired. Request a new one.")
        if pending.attempts >= self.settings.max_attempts:
            self._store.void_code(email)
            raise AuthError("Too many wrong codes. Request a new one.")
        if not code_matches(code, pending.salt, pending.code_hash):
            self._store.add_attempt(email)
            left = self.settings.max_attempts - pending.attempts - 1
            if left <= 0:
                self._store.void_code(email)
                raise AuthError("Too many wrong codes. Request a new one.")
            raise InvalidCode(f"That code is not correct. {left} attempt(s) left.", left)
        self._store.void_code(email)  # single use
        user: User = self._store.record_login(email, now)
        log.info("Signed in %s", mask_email(email))
        return SignedInUser(user.email, now)

    def session_valid(self, user: Optional[SignedInUser]) -> bool:
        """A browser session stays signed in for at most ``session_max_age_s``."""
        return user is not None and self._clock() - user.signed_in_at < self.settings.session_max_age_s


def build_auth_service(settings: AuthSettings) -> AuthService:
    """The application's service: SQLite store + SMTP mailer + disposable list."""
    return AuthService(settings, AuthStore(settings.auth_db_path), SmtpMailer(settings),
                       load_disposable_domains())


__all__ = ["AuthError", "AuthService", "EmailRejected", "InvalidCode", "MailerError",
           "RateLimited", "SignedInUser", "build_auth_service"]
