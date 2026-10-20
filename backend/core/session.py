"""
Interview session: which page the user is on, which questions are asked,
and the outcome of each. Pure Python so the whole flow is unit-testable.

    WELCOME -> SETUP -> READINESS -> CONSENT -> QUESTION <-> REVIEW -> SUMMARY
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from backend.core.states import InvalidTransitionError, RunState


class Stage(str, Enum):
    WELCOME = "welcome"
    SETUP = "setup"
    READINESS = "readiness"
    CONSENT = "consent"
    QUESTION = "question"
    REVIEW = "review"
    SUMMARY = "summary"


STAGE_TRANSITIONS: dict[Stage, frozenset[Stage]] = {
    Stage.WELCOME: frozenset({Stage.SETUP}),
    Stage.SETUP: frozenset({Stage.READINESS, Stage.WELCOME}),
    Stage.READINESS: frozenset({Stage.CONSENT, Stage.SETUP}),
    Stage.CONSENT: frozenset({Stage.QUESTION, Stage.SETUP}),
    Stage.QUESTION: frozenset({Stage.REVIEW, Stage.SETUP}),
    Stage.REVIEW: frozenset({Stage.QUESTION, Stage.SUMMARY, Stage.SETUP}),
    Stage.SUMMARY: frozenset({Stage.SETUP, Stage.WELCOME}),
}

MAX_QUESTIONS = 10


@dataclass
class InterviewSession:
    stage: Stage = Stage.WELCOME
    role: Optional[str] = None
    question_ids: list[str] = field(default_factory=list)
    duration_s: int = 30
    use_camera: bool = True
    consent_given: bool = False
    index: int = 0
    # question_id -> RunOutcome (kept as Any to avoid importing the service)
    outcomes: dict[str, Any] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)

    # ---------------- transitions ----------------
    def _go(self, target: Stage) -> None:
        if target not in STAGE_TRANSITIONS[self.stage]:
            raise InvalidTransitionError(
                f"Cannot go from '{self.stage.value}' to '{target.value}'."
            )
        self.stage = target

    def begin_setup(self) -> None:
        self._go(Stage.SETUP)

    def back_to_welcome(self) -> None:
        self._go(Stage.WELCOME)

    def configure(self, role: str, question_ids: list[str], duration_s: int,
                  use_camera: bool) -> None:
        if self.stage is not Stage.SETUP:
            raise InvalidTransitionError("The interview can only be configured on the setup page.")
        if not role:
            raise ValueError("Choose a role.")
        if not question_ids:
            raise ValueError("Choose at least one question.")
        if len(question_ids) > MAX_QUESTIONS:
            raise ValueError(f"Choose at most {MAX_QUESTIONS} questions.")
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("Each question can be chosen only once.")
        self.role = role
        self.question_ids = list(question_ids)
        self.duration_s = int(duration_s)
        self.use_camera = bool(use_camera)
        self.index = 0
        self.outcomes = {}
        self.attempts = {}
        self.consent_given = False
        self._go(Stage.READINESS)

    def readiness_confirmed(self) -> None:
        self._go(Stage.CONSENT)

    def give_consent(self) -> None:
        if self.stage is not Stage.CONSENT:
            raise InvalidTransitionError("Consent is given on the consent page.")
        self.consent_given = True
        self._go(Stage.QUESTION)

    def restart(self) -> None:
        """Back to setup from anywhere after the welcome page; clears results."""
        if self.stage is Stage.WELCOME:
            self._go(Stage.SETUP)
            return
        self.outcomes = {}
        self.attempts = {}
        self.index = 0
        self.consent_given = False
        self.stage = Stage.SETUP

    # ---------------- questions ----------------
    @property
    def current_question_id(self) -> Optional[str]:
        if 0 <= self.index < len(self.question_ids):
            return self.question_ids[self.index]
        return None

    @property
    def is_last_question(self) -> bool:
        return self.index == len(self.question_ids) - 1

    def record_outcome(self, outcome: Any) -> None:
        """Store the outcome of the current question exactly once."""
        if self.stage is not Stage.QUESTION:
            raise InvalidTransitionError("Outcomes can only be recorded while answering a question.")
        if not self.consent_given:
            raise InvalidTransitionError("Recording requires consent.")
        qid = self.current_question_id
        assert qid is not None
        if qid in self.outcomes:
            raise InvalidTransitionError(f"Question '{qid}' already has a result.")
        state = getattr(outcome, "state", None)
        if not isinstance(state, RunState) or not state.is_terminal:
            raise ValueError("Only finished runs can be recorded.")
        self.outcomes[qid] = outcome
        self.attempts[qid] = self.attempts.get(qid, 0) + 1
        self._go(Stage.REVIEW)

    def retry_question(self) -> None:
        """Discard the current question's outcome and answer it again."""
        if self.stage is not Stage.REVIEW:
            raise InvalidTransitionError("Retry is available after a question has finished.")
        qid = self.current_question_id
        if qid is not None:
            self.outcomes.pop(qid, None)
        self._go(Stage.QUESTION)

    def next_question(self) -> None:
        if self.stage is not Stage.REVIEW:
            raise InvalidTransitionError("Move on after reviewing the current question.")
        if self.is_last_question:
            self._go(Stage.SUMMARY)
            return
        self.index += 1
        self._go(Stage.QUESTION)

    def finish(self) -> None:
        self._go(Stage.SUMMARY)

    @property
    def progress_text(self) -> str:
        return f"Question {self.index + 1} of {len(self.question_ids)}"
