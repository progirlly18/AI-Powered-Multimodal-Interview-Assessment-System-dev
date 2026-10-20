"""Scoring rules, with the hand-calculated examples from docs/SCORING.md."""

import math

import pytest

from backend.core.results import (
    AnswerResult,
    Metric,
    SpeechResult,
    Status,
    VisionResult,
    round_half_up,
)
from backend.scoring.scoring_engine import (
    COMPONENTS,
    ScoringInputError,
    answer_score,
    build_suggestions,
    clamp_0_100,
    compute_overall,
    delivery_score,
    score_question,
    session_summary,
)


def A(v):
    return Metric.available("x", v)


def U(reason="missing"):
    return Metric.unavailable("x", reason)


# ---------------- weights ----------------

def test_weights_sum_to_100_and_answer_is_required():
    assert sum(c.weight for c in COMPONENTS) == 100
    required = [c.key for c in COMPONENTS if c.required]
    assert required == ["answer"]


# ---------------- clamp / rounding ----------------

@pytest.mark.parametrize("value,expected", [(-5, 0), (0, 0), (55.5, 55.5), (100, 100), (250, 100)])
def test_clamp(value, expected):
    assert clamp_0_100(value) == expected


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None, True])
def test_clamp_rejects_non_finite(bad):
    with pytest.raises(ScoringInputError):
        clamp_0_100(bad)


@pytest.mark.parametrize("value,expected", [(83.5, 84), (82.4999, 82), (2.5, 3), (0.5, 1), (0, 0), (100, 100)])
def test_round_half_up(value, expected):
    assert round_half_up(value) == expected


# ---------------- delivery ----------------

@pytest.mark.parametrize(
    "wpm,fillers,expected",
    [
        (120, 0, 100),   # good pace, no fillers
        (120, 2, 94),    # 100 - 2*3
        (80, 0, 90),     # slow: -10
        (180, 0, 90),    # fast: -10
        (90, 0, 100),    # boundary is inside the range
        (170, 0, 100),
        (89.9, 0, 90),
        (170.1, 0, 90),
        (60, 40, 0),     # clamps at 0
    ],
)
def test_delivery_score(wpm, fillers, expected):
    m = delivery_score(Metric.available("Speaking pace", wpm), fillers)
    assert m.is_available and m.value == expected


def test_delivery_unavailable_when_pace_unknown():
    m = delivery_score(Metric.unavailable("Speaking pace", "Less than 3 s of speech"), 0)
    assert not m.is_available and "3 s" in m.reason


def test_delivery_rejects_negative_fillers():
    with pytest.raises(ScoringInputError):
        delivery_score(A(120), -1)


def test_speech_contribution_no_longer_saturates():
    """Regression: the old engine capped a 0-100 speech score at 20 points,
    so 50 and 100 both scored 20. Now different deliveries give different scores."""
    low = compute_overall({"answer": A(80), "delivery": A(50)}).overall.value
    high = compute_overall({"answer": A(80), "delivery": A(100)}).overall.value
    assert high > low


# ---------------- answer content ----------------

def test_answer_score_formula():
    assert answer_score(A(50), A(75)).value == pytest.approx(65.0)  # 0.4*50 + 0.6*75


def test_answer_score_unavailable_propagates_reason():
    m = answer_score(U("no transcript"), A(50))
    assert m.status is Status.UNAVAILABLE and "no transcript" in m.reason


def test_answer_score_failed_status_propagates():
    m = answer_score(Metric.failed("Topic relevance", "model error"), A(50))
    assert m.status is Status.FAILED


def test_answer_score_rejects_out_of_range():
    with pytest.raises(ScoringInputError):
        answer_score(A(120), A(50))


# ---------------- overall ----------------

def test_example_complete_score():
    """docs/SCORING.md example 1."""
    b = compute_overall({"answer": A(80), "delivery": A(90), "eye_contact": A(70), "head_facing": A(100)})
    assert b.overall.value == pytest.approx(83.5)
    assert round_half_up(b.overall.value) == 84
    assert not b.is_partial
    assert sum(c.contribution for c in b.components) == pytest.approx(83.5)
    assert sum(c.effective_weight for c in b.components) == pytest.approx(1.0)


def test_example_audio_only_partial_score():
    """docs/SCORING.md example 2: camera off -> weights rescaled over 70."""
    b = compute_overall({"answer": A(80), "delivery": A(90),
                         "eye_contact": U("Camera not used"), "head_facing": U("Camera not used")})
    assert b.overall.value == pytest.approx((50 * 80 + 20 * 90) / 70)
    assert round_half_up(b.overall.value) == 83
    assert b.is_partial
    assert any("not directly comparable" in line for line in b.explanation)
    assert [c.key for c in b.missing] == ["eye_contact", "head_facing"]


def test_example_no_answer_withholds_score():
    """docs/SCORING.md example 3."""
    b = compute_overall({"answer": U("No words were recognised"), "delivery": A(90),
                         "eye_contact": A(100), "head_facing": A(100)})
    assert not b.overall.is_available
    assert b.overall.value is None
    assert "No overall score" in b.explanation[0]


def test_answer_only_is_partial():
    b = compute_overall({"answer": A(80)})
    assert b.overall.value == pytest.approx(80)
    assert b.is_partial


def test_missing_keys_are_unavailable_not_zero():
    b = compute_overall({"answer": A(60)})
    assert all(not c.metric.is_available for c in b.components if c.key != "answer")
    assert b.overall.value == pytest.approx(60)  # not dragged down by zeros


def test_perfect_unavailable_inputs_cannot_produce_100():
    """Regression for the teammate branch: no camera + no microphone gave 100/100."""
    b = compute_overall({"answer": U("No transcript"), "delivery": U("No transcript"),
                         "eye_contact": U("No camera"), "head_facing": U("No camera")})
    assert not b.overall.is_available


def test_unknown_component_rejected():
    with pytest.raises(ScoringInputError):
        compute_overall({"answer": A(50), "emotion": A(100)})


def test_out_of_range_component_rejected():
    with pytest.raises(ScoringInputError):
        compute_overall({"answer": A(101)})


@pytest.mark.parametrize("values", [(0, 0, 0, 0), (100, 100, 100, 100), (0, 100, 0, 100)])
def test_overall_stays_in_range(values):
    keys = [c.key for c in COMPONENTS]
    b = compute_overall({k: A(v) for k, v in zip(keys, values)})
    assert 0 <= b.overall.value <= 100


def test_metric_rejects_nan_and_values_on_unavailable():
    with pytest.raises(ValueError):
        Metric.available("x", math.nan)
    with pytest.raises(ValueError):
        Metric(name="x", status=Status.UNAVAILABLE, value=0.0)


# ---------------- question-level / session-level ----------------

def _vision(eye=80.0, head=90.0):
    return VisionResult(status=Status.AVAILABLE,
                        face_presence=Metric.available("Face in view", 100, "%"),
                        eye_contact=Metric.available("Looking at camera", eye, "%"),
                        head_facing=Metric.available("Facing camera", head, "%"))


def _speech(wpm=120.0, fillers=0):
    w = Metric.available("Speaking pace", wpm, " wpm")
    return SpeechResult(status=Status.AVAILABLE, transcript="text", wpm=w,
                        filler_count=fillers, delivery_score=delivery_score(w, fillers))


def _answer(score=70.0, concepts=None):
    return AnswerResult(status=Status.AVAILABLE, relevance=A(score), coverage=A(score),
                        answer_score=A(score),
                        concept_scores={"A": 100, "B": 0} if concepts is None else concepts,
                        feedback="Focus on the specific concepts.")


def test_score_question_uses_answer_score():
    """Regression: the answer score used to be excluded from the overall score."""
    low = score_question(_vision(), _speech(), _answer(20)).overall.value
    high = score_question(_vision(), _speech(), _answer(90)).overall.value
    assert high - low == pytest.approx(0.5 * 70)


def test_emotion_and_background_do_not_affect_score():
    v1, v2 = _vision(), _vision()
    v1.dominant_emotion, v2.dominant_emotion = "Happy", "Sad"
    s1, s2 = _speech(), _speech()
    s1.background_label, s2.background_label = "Speech", "Rain"
    a = _answer()
    assert score_question(v1, s1, a).overall.value == score_question(v2, s2, a).overall.value


def test_session_summary():
    mean, scored, total = session_summary([A(80), U(), A(60)])
    assert (mean, scored, total) == (70, 2, 3)
    assert session_summary([U(), U()]) == (None, 0, 2)
    assert session_summary([]) == (None, 0, 0)


def test_suggestions_only_use_measured_metrics():
    unavailable_vision = VisionResult(status=Status.UNAVAILABLE, reason="no camera")
    tips = build_suggestions(unavailable_vision, _speech(wpm=60, fillers=5), _answer(40))
    text = " ".join(tips)
    assert "camera" not in text.lower()
    assert "slowly" in text and "um" in text
    assert "Points you could cover next time: B." in tips


def test_suggestions_default_message():
    tips = build_suggestions(_vision(100, 100), _speech(), _answer(95, {"A": 100}))
    assert tips == ["No specific issues were detected in the areas that were measured."]
