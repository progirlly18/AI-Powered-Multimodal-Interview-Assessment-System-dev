"""Session records, SQLite history and report export."""

import csv
import io
import sqlite3
from types import SimpleNamespace

import pytest

from backend.core.results import (
    AnswerResult,
    CriterionScore,
    Metric,
    QuestionResult,
    SpeechResult,
    Status,
    VisionResult,
)
from backend.core.states import RunState
from backend.scoring.scoring_engine import score_question
from backend.storage.history import HistoryError, HistoryStore
from backend.storage.records import SessionRecord, build_session_record, new_session_id
from backend.storage.report import to_csv, to_html


def result(transcript="Secret transcript <b>bold</b>"):
    crit = [CriterionScore("relevance", "Relevance", Metric.available("Relevance", 60), 0.5, "why"),
            CriterionScore("coverage", "Concept coverage", Metric.available("Concept coverage", 80),
                           0.5, "found 2 of 3")]
    answer = AnswerResult(status=Status.AVAILABLE, relevance=Metric.available("r", 60),
                          coverage=Metric.available("c", 80),
                          answer_score=Metric.available("Answer content", 70), criteria=crit,
                          strengths=["Covered the key concepts."], improvements=["Add an example."])
    speech = SpeechResult(status=Status.AVAILABLE, transcript=transcript,
                          wpm=Metric.available("Speaking pace", 120),
                          delivery_score=Metric.available("Speaking delivery", 94), filler_count=2)
    reason = "Camera not used (audio-only mode)."
    vision = VisionResult(status=Status.UNAVAILABLE, reason=reason,
                          eye_contact=Metric.unavailable("Looking at camera", reason, "%"),
                          head_facing=Metric.unavailable("Facing camera", reason, "%"))
    score = score_question(vision, speech, answer)
    return QuestionResult("q1", "What is X?", 30, vision, speech, answer, score, ["tip"])


def record(include=False):
    items = [("q1", "What is X?", SimpleNamespace(state=RunState.COMPLETED, message="ok",
                                                  result=result())),
             ("q2", "What is Y?", SimpleNamespace(state=RunState.CANCELLED,
                                                  message="Cancelled.", result=None)),
             ("q3", "What is Z?", None)]
    return build_session_record("s-1", "2026-10-23 10:00", "AI Engineer", "  Ada <script>  ",
                                items, include_transcripts=include)


def test_record_reflects_results():
    r = record()
    a1, a2, a3 = r.answers
    assert a1.overall == pytest.approx((50 * 70 + 20 * 94) / 70)  # partial: no camera
    assert a1.is_partial and not a1.complete
    assert a1.components["Looking at the camera"] is None
    assert "audio-only" in a1.component_notes["Looking at the camera"]
    assert [c.score for c in a1.criteria] == [60, 80]
    assert a1.wpm == 120 and a1.filler_count == 2
    assert a2.overall is None and a2.state == "cancelled"
    assert a3.state == "not answered"
    assert r.average == pytest.approx(a1.overall) and len(r.scored) == 1
    assert r.candidate == "Ada <script>"


def test_transcripts_excluded_unless_requested():
    assert record(include=False).answers[0].transcript is None
    assert record(include=True).answers[0].transcript.startswith("Secret")


def test_round_trip_dict():
    r = record(include=True)
    assert SessionRecord.from_dict(r.to_dict()) == r


def test_session_ids_are_unique():
    assert len({new_session_id() for _ in range(50)}) == 50


# ---------------- history ----------------

def test_history_save_list_get_delete_clear(tmp_path):
    store = HistoryStore(tmp_path / "sub" / "history.db", "ada@uni.edu")
    r = record()
    store.save(r)
    store.save(r)  # idempotent
    summaries = store.list()
    assert len(summaries) == 1 and summaries[0].scored == 1 and summaries[0].questions == 3
    assert store.get("s-1") == r
    assert store.get("missing") is None
    assert store.delete("s-1") is True and store.delete("s-1") is False
    store.save(r)
    assert store.clear() == 1 and store.list() == []


def test_history_never_stores_transcript_by_default(tmp_path):
    path = tmp_path / "h.db"
    HistoryStore(path, "ada@uni.edu").save(record(include=False))
    raw = sqlite3.connect(path).execute("SELECT record_json FROM sessions").fetchone()[0]
    assert "Secret transcript" not in raw


def test_history_errors_are_friendly(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    with pytest.raises(HistoryError, match="could not be opened"):
        HistoryStore(blocker / "h.db", "ada@uni.edu").list()


def test_damaged_record_reported(tmp_path):
    path = tmp_path / "h.db"
    store = HistoryStore(path, "ada@uni.edu")
    store.save(record())
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE sessions SET record_json = '{broken'")
    with pytest.raises(HistoryError, match="damaged"):
        store.get("s-1")


# ---------------- export ----------------

def test_csv_matches_record():
    text = to_csv(record(include=True))
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["question_id"] for r in rows] == ["q1", "q2", "q3"]
    assert rows[0]["practice_score"] == str(round(((50 * 70 + 20 * 94) / 70)))
    assert rows[0]["partial"] == "yes" and rows[0]["complete"] == "no"
    assert rows[0]["Looking at the camera"] == ""          # not measured -> blank, not 0
    assert rows[0]["rubric: Relevance"] == "60"
    assert rows[1]["practice_score"] == "" and rows[1]["status"] == "cancelled"
    assert rows[0]["transcript"].startswith("Secret")


def test_html_report_escapes_and_labels():
    html = to_html(record(include=True))
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<b>bold</b>" not in html and "&lt;b&gt;bold&lt;/b&gt;" in html
    assert "not measured" in html and "(partial)" in html
    assert "must not be used to make hiring decisions" in html
    assert "No overall score for this question." in html


def test_history_is_isolated_per_owner(tmp_path):
    path = tmp_path / "h.db"
    ada, bob = HistoryStore(path, "ada@uni.edu"), HistoryStore(path, "bob@uni.edu")
    ada.save(record())
    assert len(ada.list()) == 1
    assert bob.list() == [] and bob.get("s-1") is None
    assert bob.delete("s-1") is False and bob.clear() == 0
    assert len(ada.list()) == 1                      # untouched by bob
    with pytest.raises(HistoryError, match="another account"):
        bob.save(record())                           # cannot overwrite ada's session id


def test_old_database_is_migrated_and_unowned_rows_hidden(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
                     "role TEXT NOT NULL, candidate TEXT, questions INTEGER NOT NULL, "
                     "scored INTEGER NOT NULL, average REAL, record_json TEXT NOT NULL)")
        conn.execute("INSERT INTO sessions VALUES ('old', '2026-01-01', 'R', NULL, 1, 0, NULL, '{}')")
    store = HistoryStore(path, "ada@uni.edu")
    assert store.list() == []
    store.save(record())
    assert [s.session_id for s in store.list()] == ["s-1"]


def test_owner_required(tmp_path):
    with pytest.raises(ValueError):
        HistoryStore(tmp_path / "h.db", "")
