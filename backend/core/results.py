"""
Result types that make missing data explicit.

Every measurement carries a status. A value is only present when the status
is ``AVAILABLE``; an unavailable or failed measurement never carries a
substitute number, so it cannot be mistaken for zero, a perfect result or a
normal result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Status(str, Enum):
    """Availability of a measurement or of a whole modality."""

    AVAILABLE = "available"
    # The input needed for the measurement was not present
    # (e.g. no face in view, no speech detected, device missing).
    UNAVAILABLE = "unavailable"
    # Something went wrong while producing the measurement
    # (e.g. a model failed to load or raised an exception).
    FAILED = "failed"


@dataclass(frozen=True)
class Metric:
    """A single named measurement."""

    name: str
    status: Status
    value: Optional[float] = None
    unit: str = ""
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is Status.AVAILABLE:
            if self.value is None:
                raise ValueError(f"Available metric '{self.name}' needs a value.")
            if isinstance(self.value, bool) or not math.isfinite(float(self.value)):
                raise ValueError(
                    f"Metric '{self.name}' has a non-finite value: {self.value!r}"
                )
        elif self.value is not None:
            raise ValueError(
                f"Metric '{self.name}' is {self.status.value} and must not carry a value."
            )

    @property
    def is_available(self) -> bool:
        return self.status is Status.AVAILABLE

    @classmethod
    def available(cls, name: str, value: float, unit: str = "") -> "Metric":
        return cls(name=name, status=Status.AVAILABLE, value=float(value), unit=unit)

    @classmethod
    def unavailable(cls, name: str, reason: str, unit: str = "") -> "Metric":
        return cls(name=name, status=Status.UNAVAILABLE, unit=unit, reason=reason)

    @classmethod
    def failed(cls, name: str, reason: str, unit: str = "") -> "Metric":
        return cls(name=name, status=Status.FAILED, unit=unit, reason=reason)

    def display(self, digits: int = 0) -> str:
        """Human-readable value, or a label explaining why there is none."""
        if not self.is_available:
            return "Not available" if self.status is Status.UNAVAILABLE else "Failed"
        assert self.value is not None
        text = f"{self.value:.{digits}f}" if digits else f"{round_half_up(self.value)}"
        return f"{text}{self.unit}"


def round_half_up(value: float) -> int:
    """Round to the nearest integer, halves away from zero (2.5 -> 3).

    Python's built-in ``round`` uses banker's rounding (2.5 -> 2), which
    surprises people checking scores by hand.
    """
    if not math.isfinite(value):
        raise ValueError(f"Cannot round non-finite value {value!r}")
    return int(math.floor(abs(value) + 0.5)) * (1 if value >= 0 else -1)


# --------------------------------------------------
# Modality results
# --------------------------------------------------

@dataclass
class VisionResult:
    """Aggregated computer-vision measurements for one answer."""

    status: Status
    reason: Optional[str] = None
    frames_captured: int = 0
    frames_analysed: int = 0
    frames_with_face: int = 0
    face_presence: Metric = field(
        default_factory=lambda: Metric.unavailable("Face in view", "Not measured", "%")
    )
    eye_contact: Metric = field(
        default_factory=lambda: Metric.unavailable("Looking at camera", "Not measured", "%")
    )
    head_facing: Metric = field(
        default_factory=lambda: Metric.unavailable("Facing camera", "Not measured", "%")
    )
    # Informational only: the emotion model is low-accuracy (see docs/SCORING.md).
    dominant_emotion: Optional[str] = None
    emotion_counts: dict[str, int] = field(default_factory=dict)
    eye_counts: dict[str, int] = field(default_factory=dict)
    head_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class SpeechResult:
    """Recording, transcription and delivery measurements for one answer."""

    status: Status
    reason: Optional[str] = None
    transcript: Optional[str] = None
    recording_seconds: float = 0.0
    speaking_seconds: Optional[float] = None
    speaking_time_is_estimate: bool = False
    word_count: int = 0
    wpm: Metric = field(
        default_factory=lambda: Metric.unavailable("Speaking pace", "Not measured", " wpm")
    )
    filler_count: int = 0
    filler_counts: dict[str, int] = field(default_factory=dict)
    # Discourse markers such as "like" or "so": reported, never penalised.
    possible_filler_counts: dict[str, int] = field(default_factory=dict)
    delivery_score: Metric = field(
        default_factory=lambda: Metric.unavailable("Delivery", "Not measured")
    )
    # Informational only.
    background_label: Optional[str] = None
    background_confidence: Optional[float] = None
    background_note: Optional[str] = None


@dataclass
class AnswerResult:
    """Content evaluation of one transcribed answer."""

    status: Status
    reason: Optional[str] = None
    relevance: Metric = field(
        default_factory=lambda: Metric.unavailable("Topic relevance", "Not measured")
    )
    coverage: Metric = field(
        default_factory=lambda: Metric.unavailable("Concept coverage", "Not measured")
    )
    reference_similarity: Metric = field(
        default_factory=lambda: Metric.unavailable(
            "Similarity to reference answer", "No reference answer for this question"
        )
    )
    answer_score: Metric = field(
        default_factory=lambda: Metric.unavailable("Answer content", "Not measured")
    )
    concept_scores: dict[str, int] = field(default_factory=dict)
    feedback: Optional[str] = None


@dataclass
class ScoreComponent:
    """One weighted input to the overall practice score."""

    key: str
    label: str
    metric: Metric
    nominal_weight: float
    effective_weight: float = 0.0
    contribution: Optional[float] = None


@dataclass
class ScoreBreakdown:
    """The overall practice score and exactly how it was computed."""

    components: list[ScoreComponent]
    overall: Metric
    is_partial: bool
    explanation: list[str]

    @property
    def used(self) -> list[ScoreComponent]:
        return [c for c in self.components if c.metric.is_available]

    @property
    def missing(self) -> list[ScoreComponent]:
        return [c for c in self.components if not c.metric.is_available]


@dataclass
class QuestionResult:
    """Everything measured for one question."""

    question_id: str
    question_text: str
    duration_seconds: float
    vision: VisionResult
    speech: SpeechResult
    answer: AnswerResult
    score: ScoreBreakdown
    suggestions: list[str]
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    # Measured gap between the first kept audio sample and the first
    # analysed video frame (positive = video started later).
    av_start_offset_ms: Optional[float] = None

    @property
    def is_complete(self) -> bool:
        """True only when every modality and every score component was measured."""
        return (
            self.vision.status is Status.AVAILABLE
            and self.speech.status is Status.AVAILABLE
            and self.answer.status is Status.AVAILABLE
            and self.score.overall.is_available
            and not self.score.is_partial
        )
