"""
Practice-score calculation.

The full specification, with worked examples, is in docs/SCORING.md.
Summary:

* Every component is a 0-100 ``Metric`` that is either available or not.
* Unavailable components are never treated as 0 or as a perfect result;
  they are left out and the remaining weights are rescaled.
* Without the answer-content component there is no overall score.
* Emotion and background-sound labels are informational and not scored.

The score is a coaching aid for interview practice. It has not been
validated against human ratings and is not a hiring recommendation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from backend.core.results import (
    AnswerResult,
    Metric,
    ScoreBreakdown,
    ScoreComponent,
    SpeechResult,
    Status,
    VisionResult,
    round_half_up,
)


class ScoringInputError(ValueError):
    """Raised when a component value is outside its valid 0-100 range."""


@dataclass(frozen=True)
class ComponentSpec:
    key: str
    label: str
    weight: float
    required: bool


# Provisional weights (see docs/SCORING.md for the rationale). They are not
# validated against human ratings; change them only together with the docs
# and the tests in tests/unit/test_scoring_engine.py.
COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec("answer", "Answer content", 50.0, required=True),
    ComponentSpec("delivery", "Speaking delivery", 20.0, required=False),
    ComponentSpec("eye_contact", "Looking at the camera", 15.0, required=False),
    ComponentSpec("head_facing", "Facing the camera", 15.0, required=False),
)

# Delivery rule (unchanged from the original speech analyser).
FILLER_PENALTY = 3.0
PACE_PENALTY = 10.0
PACE_MIN_WPM = 90.0
PACE_MAX_WPM = 170.0

# Answer-content rule (unchanged from the original answer evaluator).
RELEVANCE_WEIGHT = 0.40
COVERAGE_WEIGHT = 0.60


def clamp_0_100(value: float) -> float:
    """Clamp to [0, 100]; non-finite input is an error, never a score."""
    if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ScoringInputError(f"Score input must be a finite number, got {value!r}")
    return min(100.0, max(0.0, float(value)))


def _check_range(metric: Metric) -> None:
    if metric.is_available:
        assert metric.value is not None
        if not 0.0 <= metric.value <= 100.0:
            raise ScoringInputError(
                f"'{metric.name}' must be between 0 and 100, got {metric.value}"
            )


# --------------------------------------------------
# Component rules
# --------------------------------------------------

def delivery_score(wpm: Metric, filler_count: int) -> Metric:
    """Delivery = 100 - 3 per filler - 10 if pace is outside 90-170 wpm."""
    name = "Speaking delivery"
    if not wpm.is_available:
        return Metric.unavailable(name, wpm.reason or "Speaking pace not available")
    if filler_count < 0:
        raise ScoringInputError("filler_count cannot be negative")
    assert wpm.value is not None
    score = 100.0 - FILLER_PENALTY * filler_count
    if wpm.value < PACE_MIN_WPM or wpm.value > PACE_MAX_WPM:
        score -= PACE_PENALTY
    return Metric.available(name, clamp_0_100(score))


def answer_score(relevance: Metric, coverage: Metric) -> Metric:
    """Answer content = 0.4 x topic relevance + 0.6 x concept coverage."""
    name = "Answer content"
    for metric in (relevance, coverage):
        _check_range(metric)
        if not metric.is_available:
            return Metric(
                name=name,
                status=metric.status,
                reason=f"{metric.name}: {metric.reason}",
            )
    assert relevance.value is not None and coverage.value is not None
    value = RELEVANCE_WEIGHT * relevance.value + COVERAGE_WEIGHT * coverage.value
    return Metric.available(name, clamp_0_100(value))


# --------------------------------------------------
# Overall score
# --------------------------------------------------

def compute_overall(component_metrics: dict[str, Metric]) -> ScoreBreakdown:
    """Combine component metrics into an overall practice score.

    ``component_metrics`` maps each key in ``COMPONENTS`` to a Metric.
    Missing keys are treated as unavailable.
    """
    unknown = set(component_metrics) - {spec.key for spec in COMPONENTS}
    if unknown:
        raise ScoringInputError(f"Unknown score components: {sorted(unknown)}")

    components: list[ScoreComponent] = []
    for spec in COMPONENTS:
        metric = component_metrics.get(spec.key) or Metric.unavailable(
            spec.label, "Not measured"
        )
        _check_range(metric)
        components.append(
            ScoreComponent(
                key=spec.key,
                label=spec.label,
                metric=metric,
                nominal_weight=spec.weight,
            )
        )

    explanation: list[str] = []
    missing_required = [
        c for c, spec in zip(components, COMPONENTS)
        if spec.required and not c.metric.is_available
    ]
    if missing_required:
        names = ", ".join(c.label for c in missing_required)
        reasons = "; ".join(
            f"{c.label}: {c.metric.reason or c.metric.status.value}" for c in missing_required
        )
        explanation.append(
            f"No overall score: {names} could not be measured ({reasons})."
        )
        return ScoreBreakdown(
            components=components,
            overall=Metric.unavailable("Overall practice score", f"{names} not available"),
            is_partial=True,
            explanation=explanation,
        )

    available = [c for c in components if c.metric.is_available]
    total_weight = sum(c.nominal_weight for c in available)
    overall_value = 0.0
    for c in available:
        assert c.metric.value is not None
        c.effective_weight = c.nominal_weight / total_weight
        c.contribution = c.effective_weight * c.metric.value
        overall_value += c.contribution

    missing = [c for c in components if not c.metric.is_available]
    is_partial = bool(missing)
    used_text = ", ".join(
        f"{c.label} {round_half_up(c.metric.value)} x {c.effective_weight:.0%}"  # type: ignore[arg-type]
        for c in available
    )
    explanation.append(f"Weighted average of: {used_text}.")
    if is_partial:
        explanation.append(
            "Partial score: "
            + "; ".join(f"{c.label} not included ({c.metric.reason})" for c in missing)
            + ". Weights were rescaled over the measured components, so this "
            "score is not directly comparable with a complete score."
        )

    return ScoreBreakdown(
        components=components,
        overall=Metric.available("Overall practice score", clamp_0_100(overall_value)),
        is_partial=is_partial,
        explanation=explanation,
    )


def score_question(
    vision: VisionResult, speech: SpeechResult, answer: AnswerResult
) -> ScoreBreakdown:
    """Build the score breakdown for one answer from its modality results."""
    return compute_overall(
        {
            "answer": answer.answer_score,
            "delivery": speech.delivery_score,
            "eye_contact": vision.eye_contact,
            "head_facing": vision.head_facing,
        }
    )


def session_summary(overalls: Iterable[Metric]) -> tuple[Optional[float], int, int]:
    """Mean overall score across questions that have one.

    Returns (mean or None, questions scored, questions total).
    """
    items = list(overalls)
    scored = [m.value for m in items if m.is_available and m.value is not None]
    if not scored:
        return None, 0, len(items)
    return sum(scored) / len(scored), len(scored), len(items)


# --------------------------------------------------
# Coaching suggestions
# --------------------------------------------------

def build_suggestions(
    vision: VisionResult, speech: SpeechResult, answer: AnswerResult
) -> list[str]:
    """Practical suggestions based only on metrics that were measured."""
    tips: list[str] = []

    if vision.eye_contact.is_available and vision.eye_contact.value is not None:
        if vision.eye_contact.value < 60:
            tips.append(
                "Try to look towards the camera more often while you speak "
                f"(estimated {round_half_up(vision.eye_contact.value)}% of the time)."
            )
    if vision.head_facing.is_available and vision.head_facing.value is not None:
        if vision.head_facing.value < 60:
            tips.append("Keep your head turned towards the camera.")

    if speech.wpm.is_available and speech.wpm.value is not None:
        if speech.wpm.value < PACE_MIN_WPM:
            tips.append("You spoke quite slowly; a little more pace can sound more confident.")
        elif speech.wpm.value > PACE_MAX_WPM:
            tips.append("You spoke quickly; slowing down slightly helps the listener follow.")
    if speech.filler_count > 3:
        tips.append("Try pausing silently instead of saying 'um' or 'uh'.")

    if answer.answer_score.is_available and answer.answer_score.value is not None:
        if answer.answer_score.value < 60 and answer.feedback:
            tips.append(answer.feedback)
        weak = [name for name, value in answer.concept_scores.items() if value == 0]
        if weak:
            tips.append("Points you could cover next time: " + ", ".join(weak) + ".")

    if not tips:
        tips.append("No specific issues were detected in the areas that were measured.")
    return tips
