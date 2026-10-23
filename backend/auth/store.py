"""SQLite storage for signed-in users and pending one-time codes.

Stored per user: email, first and last sign-in time. Stored per pending
code: its salted hash (never the code), expiry, attempts and send counters.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        created_at REAL NOT NULL,
        last_login_at REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS login_codes (
        email TEXT PRIMARY KEY,
        code_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        expires_at REAL NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        sent_at REAL NOT NULL,
        window_start REAL NOT NULL,
        sends_in_window INTEGER NOT NULL
    )""",
)


class AuthStoreError(RuntimeError):
    """The sign-in database could not be used."""


@dataclass
class PendingCode:
    email: str
    code_hash: str
    salt: str
    expires_at: float
    attempts: int
    sent_at: float
    window_start: float
    sends_in_window: int


@dataclass(frozen=True)
class User:
    email: str
    created_at: float
    last_login_at: float


class AuthStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=5)
            for statement in SCHEMA:
                conn.execute(statement)
            return conn
        except (sqlite3.Error, OSError) as exc:
            raise AuthStoreError(f"The sign-in database could not be opened ({exc}).") from exc

    def get_code(self, email: str) -> Optional[PendingCode]:
        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT email, code_hash, salt, expires_at, attempts, sent_at, window_start, "
                    "sends_in_window FROM login_codes WHERE email = ?", (email,)).fetchone()
        except sqlite3.Error as exc:
            raise AuthStoreError(str(exc)) from exc
        return PendingCode(*row) if row else None

    def put_code(self, code: PendingCode) -> None:
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute("INSERT OR REPLACE INTO login_codes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (code.email, code.code_hash, code.salt, code.expires_at, code.attempts,
                              code.sent_at, code.window_start, code.sends_in_window))
        except sqlite3.Error as exc:
            raise AuthStoreError(str(exc)) from exc

    def add_attempt(self, email: str) -> None:
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute("UPDATE login_codes SET attempts = attempts + 1 WHERE email = ?", (email,))
        except sqlite3.Error as exc:
            raise AuthStoreError(str(exc)) from exc

    def void_code(self, email: str) -> None:
        """Make the pending code unusable but keep the send counters (rate limits)."""
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute("UPDATE login_codes SET code_hash = '', expires_at = 0 WHERE email = ?",
                             (email,))
        except sqlite3.Error as exc:
            raise AuthStoreError(str(exc)) from exc

    def record_login(self, email: str, now: float) -> User:
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute(
                    "INSERT INTO users (email, created_at, last_login_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(email) DO UPDATE SET last_login_at = excluded.last_login_at",
                    (email, now, now))
                row = conn.execute("SELECT email, created_at, last_login_at FROM users "
                                   "WHERE email = ?", (email,)).fetchone()
        except sqlite3.Error as exc:
            raise AuthStoreError(str(exc)) from exc
        return User(*row)

    def get_user(self, email: str) -> Optional[User]:
        try:
            with closing(self._connect()) as conn:
                row = conn.execute("SELECT email, created_at, last_login_at FROM users "
                                   "WHERE email = ?", (email,)).fetchone()
        except sqlite3.Error as exc:
            raise AuthStoreError(str(exc)) from exc
        return User(*row) if row else None
