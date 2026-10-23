"""Answer evaluation with a deterministic fake encoder."""

import numpy as np
import pytest

from backend.ai.answer_evaluator import (
    AnswerEvaluator,
    contains_phrase,
    feedback_for,
    similarity_to_score,
    split_sentences,
)
from backend.ai.question_bank import Question
from backend.core.results import Status
from tests.conftest import FakeEncoder


@pytest.mark.parametrize("text,phrase,expected", [
    ("We use L1 and L2 penalties", "L1", True),
    ("We tried L10 models", "L1", False),          # whole words only
    ("k-fold cross-validation helps", "cross validation", True),  # hyphens = spaces
    ("CROSS VALIDATION", "cross validation", True),
    ("generalization is poor", "generalize", False),
    ("", "x", False),
    ("anything", "", False),
])
def test_contains_phrase(text, phrase, expected):
    assert contains_phrase(text, phrase) is expected


def test_similarity_to_score_clamps():
    assert similarity_to_score(-0.3) == 0
    assert similarity_to_score(0.42) == pytest.approx(42)
    assert similarity_to_score(1.7) == 100


def test_split_sentences():
    assert split_sentences("One. Two!  Three?") == ["One", "Two", "Three"]
    assert split_sentences("...") == []


def test_feedback_tiers():
    assert feedback_for(90).startswith("Clear")
    assert feedback_for(75).startswith("You addressed")
    assert feedback_for(55).startswith("Partly")
    assert feedback_for(10).startswith("Focus")


def test_good_answer_scores_higher_than_off_topic(overfitting_question):
    ev = AnswerEvaluator(lambda: FakeEncoder())
    good = ev.evaluate(overfitting_question,
                       "Overfitting means the model does not generalize. Use regularization "
                       "and cross validation to prevent overfitting.")
    off = ev.evaluate(overfitting_question, "Python is a programming language for applications.")
    assert good.status is Status.AVAILABLE and off.status is Status.AVAILABLE
    assert good.coverage.value == 100  # every concept has explicit evidence
    assert good.answer_score.value > off.answer_score.value
    assert off.concept_scores == {"Generalization": 0, "Regularization": 0, "Cross validation": 0}


def test_answer_score_is_weighted_rubric(overfitting_question):
    ev = AnswerEvaluator(lambda: FakeEncoder())
    r = ev.evaluate(overfitting_question, "Regularization reduces variance.")
    assert [c.key for c in r.criteria] == ["relevance", "coverage", "elaboration",
                                           "structure", "examples"]
    assert sum(c.weight for c in r.criteria) == pytest.approx(1.0)
    expected = sum(c.weight * c.metric.value for c in r.criteria)
    assert r.answer_score.value == pytest.approx(expected)
    assert 0 <= r.answer_score.value <= 100
    assert all(c.explanation for c in r.criteria)
    # Relevance and coverage in the rubric are the same numbers shown separately.
    assert r.criteria[0].metric.value == r.relevance.value
    assert r.criteria[1].metric.value == r.coverage.value


def test_strengths_and_improvements_follow_scores(overfitting_question):
    ev = AnswerEvaluator(lambda: FakeEncoder())
    rich = ev.evaluate(overfitting_question,
                       "First, overfitting means the model fails to generalize because it "
                       "memorises noise. For example, in my project a model scored 99% on "
                       "training data but 70% on test data. However, regularization and cross "
                       "validation fixed it. Overall, simpler models generalize better, and "
                       "we used early stopping as well to keep the gap small on new data.")
    thin = ev.evaluate(overfitting_question, "It is bad.")
    assert "Backed the answer with concrete detail." in rich.strengths
    assert "Signposted the reasoning clearly." in rich.strengths
    assert any("concrete example" in tip for tip in thin.improvements)
    assert any("Develop the answer" in tip for tip in thin.improvements)
    assert rich.answer_score.value > thin.answer_score.value


def test_behavioural_questions_weight_examples_more(overfitting_question):
    from dataclasses import replace

    from backend.ai.rubric import default_rubric

    tech = default_rubric().weights_for("technical")["examples"]
    beh = default_rubric().weights_for("behavioural")["examples"]
    assert beh > tech
    q = replace(overfitting_question, category="behavioural")
    r = AnswerEvaluator(lambda: FakeEncoder()).evaluate(q, "For example, I built a tool.")
    assert next(c for c in r.criteria if c.key == "examples").weight == beh


def test_invalid_rubric_fails_cleanly(overfitting_question):
    from backend.ai.rubric import RubricError

    def broken():
        raise RubricError("weights must sum to 1")
    r = AnswerEvaluator(lambda: FakeEncoder(), broken).evaluate(overfitting_question, "Answer.")
    assert r.status is Status.FAILED and "rubric" in r.reason.lower()
    assert r.answer_score.value is None


@pytest.mark.parametrize("transcript", [None, "", "   ", "...", "?!"])
def test_empty_transcript_is_unavailable_not_zero(overfitting_question, transcript):
    encoder = FakeEncoder()
    r = AnswerEvaluator(lambda: encoder).evaluate(overfitting_question, transcript)
    assert r.status is Status.UNAVAILABLE
    assert r.answer_score.value is None and not r.answer_score.is_available
    assert encoder.calls == 0


def test_encoder_failure_is_reported_not_replaced(overfitting_question):
    r = AnswerEvaluator(lambda: FakeEncoder(fail=True)).evaluate(overfitting_question, "Some answer.")
    assert r.status is Status.FAILED
    assert r.answer_score.status is Status.FAILED and r.answer_score.value is None
    assert "failed" in r.reason.lower()
    assert r.feedback is None


def test_model_load_failure_is_reported(overfitting_question):
    def provider():
        raise RuntimeError("download failed")
    r = AnswerEvaluator(provider).evaluate(overfitting_question, "Some answer.")
    assert r.status is Status.FAILED


def test_bad_encoder_output_is_rejected(overfitting_question):
    class NaNEncoder:
        def encode(self, sentences, normalize_embeddings=True):
            return np.full((len(sentences), 4), np.nan)
    r = AnswerEvaluator(lambda: NaNEncoder()).evaluate(overfitting_question, "Answer.")
    assert r.status is Status.FAILED


def test_reference_answer_reported_separately(overfitting_question):
    q = Question(**{**overfitting_question.__dict__,
                    "reference_answer": "Overfitting is poor generalization to new data."})
    r = AnswerEvaluator(lambda: FakeEncoder()).evaluate(q, "Overfitting is poor generalization.")
    assert r.reference_similarity.is_available
    no_ref = AnswerEvaluator(lambda: FakeEncoder()).evaluate(overfitting_question, "Overfitting.")
    assert not no_ref.reference_similarity.is_available
    # The reference similarity is not part of the rubric.
    assert r.answer_score.value == pytest.approx(sum(c.weight * c.metric.value for c in r.criteria))
    assert "reference" not in [c.key for c in r.criteria]


def test_semantic_only_matches_are_capped(overfitting_question):
    class SameEncoder:
        def encode(self, sentences, normalize_embeddings=True):
            return np.ones((len(sentences), 3)) / np.sqrt(3)  # every pair has similarity 1
    r = AnswerEvaluator(lambda: SameEncoder()).evaluate(overfitting_question, "Unrelated words here.")
    assert set(r.concept_scores.values()) == {70}
