"""Run-state machine and interview-session flow."""

from types import SimpleNamespace

import pytest

from backend.core.session import InterviewSession, MAX_QUESTIONS, Stage
from backend.core.states import (
    RUN_TRANSITIONS,
    InvalidTransitionError,
    RunState,
    new_run_state_machine,
)


# ---------------- run states ----------------

def test_happy_path_transitions():
    seen = []
    sm = new_run_state_machine(lambda old, new: seen.append((old, new)))
    for s in (RunState.INITIALIZING, RunState.COUNTDOWN, RunState.RECORDING,
              RunState.PROCESSING, RunState.COMPLETED):
        sm.transition(s)
    assert sm.state is RunState.COMPLETED
    assert seen[0] == (RunState.IDLE, RunState.INITIALIZING)
    assert [s for _, s in sm.history][-1] is RunState.COMPLETED


@pytest.mark.parametrize("target", [RunState.RECORDING, RunState.PROCESSING, RunState.COMPLETED])
def test_cannot_skip_initialisation(target):
    sm = new_run_state_machine()
    with pytest.raises(InvalidTransitionError):
        sm.transition(target)


def test_recording_cannot_start_without_countdown():
    sm = new_run_state_machine()
    sm.transition(RunState.INITIALIZING)
    with pytest.raises(InvalidTransitionError):
        sm.transition(RunState.RECORDING)


def test_terminal_states_are_final():
    for terminal in (RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED, RunState.UNAVAILABLE):
        assert terminal.is_terminal
        assert RUN_TRANSITIONS[terminal] == frozenset()


def test_processing_cannot_be_cancelled_but_can_fail():
    sm = new_run_state_machine()
    for s in (RunState.INITIALIZING, RunState.COUNTDOWN, RunState.RECORDING, RunState.PROCESSING):
        sm.transition(s)
    assert not sm.can_transition(RunState.CANCELLED)
    assert sm.can_transition(RunState.FAILED)


def test_unavailable_only_during_initialisation():
    assert RunState.UNAVAILABLE in RUN_TRANSITIONS[RunState.INITIALIZING]
    assert RunState.UNAVAILABLE not in RUN_TRANSITIONS[RunState.RECORDING]


# ---------------- session ----------------

def outcome(state=RunState.COMPLETED):
    return SimpleNamespace(state=state, result=None)


def configured(n=2) -> InterviewSession:
    s = InterviewSession()
    s.begin_setup()
    s.configure("Role", [f"q{i}" for i in range(1, n + 1)], 20, True)
    s.readiness_confirmed()
    s.give_consent()
    return s


def test_full_session_flow():
    s = configured(2)
    assert s.stage is Stage.QUESTION and s.current_question_id == "q1"
    assert s.progress_text == "Question 1 of 2"
    s.record_outcome(outcome())
    assert s.stage is Stage.REVIEW
    s.next_question()
    assert s.current_question_id == "q2" and s.stage is Stage.QUESTION
    s.record_outcome(outcome())
    assert s.is_last_question
    s.next_question()
    assert s.stage is Stage.SUMMARY
    assert set(s.outcomes) == {"q1", "q2"}


def test_first_question_is_index_zero():
    """Regression: the original UI always asked the 2nd question."""
    s = configured(3)
    assert s.index == 0 and s.current_question_id == "q1"


def test_outcome_recorded_once():
    s = configured(1)
    s.record_outcome(outcome())
    with pytest.raises(InvalidTransitionError):
        s.record_outcome(outcome())


def test_outcome_must_be_terminal():
    s = configured(1)
    with pytest.raises(ValueError):
        s.record_outcome(outcome(RunState.RECORDING))


def test_retry_discards_previous_outcome():
    s = configured(1)
    s.record_outcome(outcome(RunState.FAILED))
    s.retry_question()
    assert s.stage is Stage.QUESTION and "q1" not in s.outcomes
    s.record_outcome(outcome())
    assert s.attempts["q1"] == 2


def test_consent_required_before_questions():
    s = InterviewSession()
    s.begin_setup()
    s.configure("Role", ["q1"], 20, True)
    with pytest.raises(InvalidTransitionError):
        s.give_consent()  # still on the readiness page
    s.readiness_confirmed()
    assert not s.consent_given
    s.give_consent()
    assert s.consent_given


def test_cannot_record_without_consent():
    s = InterviewSession(stage=Stage.QUESTION, question_ids=["q1"])
    with pytest.raises(InvalidTransitionError):
        s.record_outcome(outcome())


@pytest.mark.parametrize("ids,error", [
    ([], "at least one"),
    (["a", "a"], "only once"),
    ([f"q{i}" for i in range(MAX_QUESTIONS + 1)], "at most"),
])
def test_configure_validation(ids, error):
    s = InterviewSession()
    s.begin_setup()
    with pytest.raises(ValueError, match=error):
        s.configure("Role", ids, 20, True)
    assert s.stage is Stage.SETUP


def test_configure_only_on_setup_page():
    s = InterviewSession()
    with pytest.raises(InvalidTransitionError):
        s.configure("Role", ["q1"], 20, True)


def test_restart_clears_results_and_consent():
    s = configured(2)
    s.record_outcome(outcome())
    s.restart()
    assert s.stage is Stage.SETUP and s.outcomes == {} and not s.consent_given and s.index == 0


def test_repeated_sessions_are_independent():
    s = configured(1)
    s.record_outcome(outcome())
    s.next_question()
    s.restart()
    s.configure("Role", ["q9"], 30, False)
    s.readiness_confirmed()
    s.give_consent()
    assert s.current_question_id == "q9" and s.outcomes == {} and s.use_camera is False


def test_finish_early_from_review():
    s = configured(3)
    s.record_outcome(outcome())
    s.finish()
    assert s.stage is Stage.SUMMARY


def test_invalid_page_jump():
    s = InterviewSession()
    with pytest.raises(InvalidTransitionError):
        s.finish()
