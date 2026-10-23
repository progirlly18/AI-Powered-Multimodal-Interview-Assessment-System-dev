"""
A compact, serialisable summary of one interview session.

The same record is shown on the results page, saved to history and
exported, so all three always agree. It holds scores, explanations and
feedback; it never holds audio or video. Transcripts are included only
when ``include_transcripts`` is set.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

from backend.core.results import Metric, QuestionResult, Status

RECORD_VERSION = 1


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _value(metric: Metric) -> Optional[float]:
    return float(metric.value) if metric.is_available and metric.value is not None else None


@dataclass
class CriterionRecord:
    key: str
    label: str
    score: Optional[float]
    weight: float
    explanation: str


@dataclass
class AnswerRecord:
    position: int
    question_id: str
    question_text: str
    state: str                       # RunState value
    message: str
    overall: Optional[float] = None
    is_partial: bool = False
    complete: bool = False
    components: dict[str, Optional[float]] = field(default_factory=dict)
    component_notes: dict[str, str] = field(default_factory=dict)
    criteria: list[CriterionRecord] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
    wpm: Optional[float] = None
    filler_count: Optional[int] = None
    transcript: Optional[str] = None


@dataclass
class SessionRecord:
    session_id: str
    created_at: str
    role: str
    candidate: Optional[str]
    answers: list[AnswerRecord]
    version: int = RECORD_VERSION

    @property
    def scored(self) -> list[float]:
        return [a.overall for a in self.answers if a.overall is not None]

    @property
    def average(self) -> Optional[float]:
        s = self.scored
        return sum(s) / len(s) if s else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRecord":
        answers = []
        for a in data.get("answers", []):
            a = dict(a)
            a["criteria"] = [CriterionRecord(**c) for c in a.get("criteria", [])]
            answers.append(AnswerRecord(**a))
        return cls(
            session_id=data["session_id"], created_at=data["created_at"], role=data["role"],
            candidate=data.get("candidate"), answers=answers,
            version=data.get("version", RECORD_VERSION),
        )


def answer_record(position: int, question_id: str, question_text: str, state: str,
                  message: str, result: Optional[QuestionResult],
                  include_transcript: bool) -> AnswerRecord:
    rec = AnswerRecord(position, question_id, question_text, state, message)
    if result is None:
        return rec
    rec.overall = _value(result.score.overall)
    rec.is_partial = result.score.is_partial
    rec.complete = result.is_complete
    rec.components = {c.label: _value(c.metric) for c in result.score.components}
    rec.component_notes = {c.label: c.metric.reason or "" for c in result.score.components
                           if not c.metric.is_available}
    rec.criteria = [CriterionRecord(c.key, c.label, _value(c.metric), c.weight, c.explanation)
                    for c in result.answer.criteria]
    rec.strengths = list(result.answer.strengths)
    rec.improvements = list(result.answer.improvements)
    rec.suggestions = list(result.suggestions)
    rec.explanation = list(result.score.explanation)
    if result.speech.status is Status.AVAILABLE:
        rec.wpm = _value(result.speech.wpm)
        rec.filler_count = result.speech.filler_count
    if include_transcript:
        rec.transcript = result.speech.transcript
    return rec


def build_session_record(session_id: str, created_at: str, role: str,
                         candidate: Optional[str],
                         items: Iterable[tuple[str, str, Any]],
                         include_transcripts: bool = False) -> SessionRecord:
    """``items`` are (question_id, question_text, RunOutcome or None) in order."""
    answers = []
    for position, (qid, text, outcome) in enumerate(items, start=1):
        if outcome is None:
            answers.append(AnswerRecord(position, qid, text, "not answered", "Not answered."))
            continue
        answers.append(answer_record(position, qid, text, outcome.state.value, outcome.message,
                                     outcome.result, include_transcripts))
    name = (candidate or "").strip() or None
    return SessionRecord(session_id, created_at, role, name, answers)
