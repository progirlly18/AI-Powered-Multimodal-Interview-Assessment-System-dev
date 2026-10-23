"""
Local interview history in SQLite (Python standard library, no server).

Each saved session belongs to one owner (the signed-in email, or "local"
when sign-in is turned off). Every query is filtered by owner, so users
only ever see, delete or clear their own sessions.

What is stored: the ``SessionRecord`` (scores, explanations, feedback and,
only if the user chose to, transcripts). Never audio, video or frames.
The database file is ``instance/history.db`` by default (git-ignored).

All database errors are raised as ``HistoryError`` with a short message,
so the UI can show a warning instead of crashing.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend.storage.records import SessionRecord

LOCAL_OWNER = "local"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    role         TEXT NOT NULL,
    candidate    TEXT,
    questions    INTEGER NOT NULL,
    scored       INTEGER NOT NULL,
    average      REAL,
    record_json  TEXT NOT NULL,
    owner        TEXT
)
"""


class HistoryError(RuntimeError):
    """The history database could not be read or written."""


@dataclass(frozen=True)
class HistorySummary:
    session_id: str
    created_at: str
    role: str
    candidate: Optional[str]
    questions: int
    scored: int
    average: Optional[float]


class HistoryStore:
    def __init__(self, path: Path, owner: str) -> None:
        if not owner:
            raise ValueError("A history owner is required.")
        self.path = Path(path)
        self.owner = owner

    def _connect(self) -> sqlite3.Connection:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=5)
            conn.execute(SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            if "owner" not in columns:
                # Databases created before sign-in existed: rows keep owner NULL
                # and are therefore visible to nobody.
                conn.execute("ALTER TABLE sessions ADD COLUMN owner TEXT")
                conn.commit()
            return conn
        except (sqlite3.Error, OSError) as exc:
            raise HistoryError(f"The history database could not be opened ({exc}).") from exc

    def save(self, record: SessionRecord) -> None:
        """Insert or replace one of the owner's sessions (saving twice keeps one copy)."""
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        try:
            with closing(self._connect()) as conn, conn:
                row = conn.execute("SELECT owner FROM sessions WHERE session_id = ?",
                                   (record.session_id,)).fetchone()
                if row is not None and row[0] != self.owner:
                    raise HistoryError("This session belongs to another account.")
                conn.execute(
                    "INSERT OR REPLACE INTO sessions (session_id, created_at, role, candidate, "
                    "questions, scored, average, record_json, owner) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (record.session_id, record.created_at, record.role, record.candidate,
                     len(record.answers), len(record.scored), record.average, payload, self.owner),
                )
        except sqlite3.Error as exc:
            raise HistoryError(f"The session could not be saved ({exc}).") from exc

    def list(self, limit: int = 50) -> list[HistorySummary]:
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT session_id, created_at, role, candidate, questions, scored, average "
                    "FROM sessions WHERE owner = ? ORDER BY created_at DESC LIMIT ?",
                    (self.owner, limit),
                ).fetchall()
        except sqlite3.Error as exc:
            raise HistoryError(f"History could not be read ({exc}).") from exc
        return [HistorySummary(*row) for row in rows]

    def get(self, session_id: str) -> Optional[SessionRecord]:
        try:
            with closing(self._connect()) as conn:
                row = conn.execute("SELECT record_json FROM sessions "
                                   "WHERE session_id = ? AND owner = ?",
                                   (session_id, self.owner)).fetchone()
        except sqlite3.Error as exc:
            raise HistoryError(f"History could not be read ({exc}).") from exc
        if row is None:
            return None
        try:
            return SessionRecord.from_dict(json.loads(row[0]))
        except (ValueError, KeyError, TypeError) as exc:
            raise HistoryError("This saved session is damaged and cannot be shown.") from exc

    def delete(self, session_id: str) -> bool:
        try:
            with closing(self._connect()) as conn, conn:
                cur = conn.execute("DELETE FROM sessions WHERE session_id = ? AND owner = ?",
                                   (session_id, self.owner))
                return cur.rowcount > 0
        except sqlite3.Error as exc:
            raise HistoryError(f"The session could not be deleted ({exc}).") from exc

    def clear(self) -> int:
        """Delete all of this owner's sessions (other users' are untouched)."""
        try:
            with closing(self._connect()) as conn, conn:
                return conn.execute("DELETE FROM sessions WHERE owner = ?", (self.owner,)).rowcount
        except sqlite3.Error as exc:
            raise HistoryError(f"History could not be cleared ({exc}).") from exc
