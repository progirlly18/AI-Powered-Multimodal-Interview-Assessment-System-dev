"""
Explicit state machines for one recording run and for device readiness.

Transitions are validated: an illegal transition raises
``InvalidTransitionError`` instead of silently corrupting state.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Callable, Generic, Optional, TypeVar


class InvalidTransitionError(RuntimeError):
    """Raised when a state change is not allowed."""


class RunState(str, Enum):
    """Lifecycle of recording and evaluating a single answer."""

    IDLE = "idle"
    INITIALIZING = "initializing"     # opening camera and microphone
    COUNTDOWN = "countdown"           # devices open, nothing is recorded yet
    RECORDING = "recording"           # audio and video are being captured
    PROCESSING = "processing"         # devices released, models running
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"       # a required device could not be used
    FAILED = "failed"                 # an unexpected error stopped the run
    CANCELLED = "cancelled"           # the user stopped the run

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_RUN_STATES

    @property
    def is_active(self) -> bool:
        return self in (
            RunState.INITIALIZING,
            RunState.COUNTDOWN,
            RunState.RECORDING,
            RunState.PROCESSING,
        )


TERMINAL_RUN_STATES = frozenset(
    {RunState.COMPLETED, RunState.UNAVAILABLE, RunState.FAILED, RunState.CANCELLED}
)

_STOPPED = {RunState.FAILED, RunState.CANCELLED}

RUN_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.IDLE: frozenset({RunState.INITIALIZING}),
    RunState.INITIALIZING: frozenset({RunState.COUNTDOWN, RunState.UNAVAILABLE, *_STOPPED}),
    RunState.COUNTDOWN: frozenset({RunState.RECORDING, *_STOPPED}),
    RunState.RECORDING: frozenset({RunState.PROCESSING, *_STOPPED}),
    RunState.PROCESSING: frozenset({RunState.COMPLETED, RunState.FAILED}),
    RunState.COMPLETED: frozenset(),
    RunState.UNAVAILABLE: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
}


class DeviceStatus(str, Enum):
    """Result of checking a camera, microphone or model."""

    NOT_CHECKED = "not checked"
    INITIALIZING = "initializing"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


S = TypeVar("S", bound=Enum)


class StateMachine(Generic[S]):
    """Small thread-safe state machine with a transition table."""

    def __init__(
        self,
        initial: S,
        transitions: dict[S, frozenset[S]],
        on_change: Optional[Callable[[S, S], None]] = None,
    ) -> None:
        self._state = initial
        self._transitions = transitions
        self._on_change = on_change
        self._lock = threading.Lock()
        self._history: list[tuple[float, S]] = [(time.monotonic(), initial)]

    @property
    def state(self) -> S:
        with self._lock:
            return self._state

    @property
    def history(self) -> list[tuple[float, S]]:
        with self._lock:
            return list(self._history)

    def can_transition(self, new_state: S) -> bool:
        with self._lock:
            return new_state in self._transitions.get(self._state, frozenset())

    def transition(self, new_state: S) -> None:
        with self._lock:
            old = self._state
            if new_state not in self._transitions.get(old, frozenset()):
                raise InvalidTransitionError(
                    f"Cannot change state from '{old.value}' to '{new_state.value}'."
                )
            self._state = new_state
            self._history.append((time.monotonic(), new_state))
        if self._on_change is not None:
            self._on_change(old, new_state)


def new_run_state_machine(
    on_change: Optional[Callable[[RunState, RunState], None]] = None,
) -> StateMachine[RunState]:
    return StateMachine(RunState.IDLE, RUN_TRANSITIONS, on_change)
