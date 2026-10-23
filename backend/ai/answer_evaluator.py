"""
Content evaluation of a transcribed answer.

Signals (all 0-100, see docs/SCORING.md):

* Topic relevance: cosine similarity between the question and the answer
  embeddings (all-MiniLM-L6-v2). It measures how on-topic the answer is,
  not whether it is correct. It is uncalibrated: there is no human-rated
  data mapping similarity values to answer quality.
* Concept coverage: per expected concept, 100 if an evidence phrase occurs
  as whole words, otherwise min(similarity x 100, 70) if the best
  alias-to-sentence similarity is at least 0.50, else 0. Averaged over
  concepts.
* Elaboration, structure, examples: transparent text heuristics
  (backend/ai/text_features.py).
* Similarity to a reference answer: only when the question has one
  (none do yet). Reported separately and not part of the answer score.

The answer score is the weighted rubric in data/rubric.json.
These are heuristics. Semantic similarity is not evidence of technical
correctness, completeness or sound reasoning.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from backend.ai.question_bank import Question
from backend.ai.rubric import CRITERIA, Rubric, RubricError, default_rubric
from backend.ai.text_features import elaboration, examples, structure
from backend.core.results import AnswerResult, CriterionScore, Metric, Status, round_half_up
from backend.scoring.scoring_engine import clamp_0_100, rubric_answer_score

log = logging.getLogger(__name__)

SEMANTIC_THRESHOLD = 0.50
SEMANTIC_ONLY_CAP = 70


class Encoder(Protocol):
    def encode(self, sentences: Sequence[str], normalize_embeddings: bool = ...) -> np.ndarray:
        ...


def similarity_to_score(similarity: float) -> float:
    """Map a cosine similarity to 0-100 by clamping to [0, 1] and scaling."""
    return clamp_0_100(max(0.0, min(1.0, float(similarity))) * 100.0)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _normalise(text: str) -> str:
    """Lower-case and treat hyphens/underscores as spaces."""
    text = text.lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word, case-insensitive phrase match ("L1" does not match "L10")."""
    norm_phrase = _normalise(phrase)
    if not norm_phrase:
        return False
    pattern = r"(?<!\w)" + re.escape(norm_phrase) + r"(?!\w)"
    return re.search(pattern, _normalise(text)) is not None


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a @ b.T


def feedback_for(score: float) -> str:
    if score >= 85:
        return ("Clear answer that addressed the question and covered the "
                "expected concepts.")
    if score >= 70:
        return ("You addressed the main idea; some expected concepts could be "
                "explained in more detail.")
    if score >= 50:
        return "Partly relevant answer. Try to explain the key concepts more directly."
    return "Focus on the specific concepts the question asks about."


class AnswerEvaluator:
    """Evaluates answers with a sentence-embedding model."""

    def __init__(self, encoder_provider: Callable[[], Encoder],
                 rubric_provider: Callable[[], Rubric] = default_rubric) -> None:
        # Providers (not the objects) so the model loads lazily and load
        # failures surface as an evaluation failure.
        self._encoder_provider = encoder_provider
        self._rubric_provider = rubric_provider

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        encoder = self._encoder_provider()
        vectors = np.asarray(encoder.encode(list(texts), normalize_embeddings=True))
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError(f"Encoder returned shape {vectors.shape} for {len(texts)} texts")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Encoder returned non-finite values")
        return vectors

    def concept_coverage(self, answer: str, question: Question) -> tuple[float, dict[str, int]]:
        sentences = split_sentences(answer)
        if not sentences:
            return 0.0, {c.name: 0 for c in question.concepts}

        need_semantic = [
            c for c in question.concepts
            if not any(contains_phrase(answer, p) for p in c.evidence)
        ]
        phrase_vectors: dict[str, np.ndarray] = {}
        if need_semantic:
            phrases = [p for c in need_semantic for p in (c.name, *c.aliases)]
            vectors = self._encode(list(sentences) + phrases)
            sentence_vecs, phrase_vecs = vectors[: len(sentences)], vectors[len(sentences):]
            offset = 0
            for c in need_semantic:
                n = 1 + len(c.aliases)
                phrase_vectors[c.name] = phrase_vecs[offset: offset + n]
                offset += n
        scores: dict[str, int] = {}
        for c in question.concepts:
            if c.name not in phrase_vectors:
                scores[c.name] = 100
                continue
            best = float(_cosine_matrix(phrase_vectors[c.name], sentence_vecs).max())
            if best >= SEMANTIC_THRESHOLD:
                scores[c.name] = min(round_half_up(similarity_to_score(best)), SEMANTIC_ONLY_CAP)
            else:
                scores[c.name] = 0
        coverage = sum(scores.values()) / len(scores) if scores else 0.0
        return clamp_0_100(coverage), scores

    def evaluate(self, question: Question, transcript: Optional[str]) -> AnswerResult:
        text = (transcript or "").strip()
        if not re.search(r"\w", text):
            reason = "No words were recognised in the recording"
            return AnswerResult(
                status=Status.UNAVAILABLE,
                reason=reason,
                relevance=Metric.unavailable("Topic relevance", reason),
                coverage=Metric.unavailable("Concept coverage", reason),
                answer_score=Metric.unavailable("Answer content", reason),
            )
        try:
            q_vec, a_vec = self._encode([question.text, text])
            relevance_value = similarity_to_score(float(_cosine_matrix(q_vec[None], a_vec[None])[0, 0]))
            coverage_value, concept_scores = self.concept_coverage(text, question)
            reference = Metric.unavailable(
                "Similarity to reference answer", "No reference answer for this question"
            )
            if question.reference_answer:
                r_vec, a2 = self._encode([question.reference_answer, text])
                reference = Metric.available(
                    "Similarity to reference answer",
                    similarity_to_score(float(_cosine_matrix(r_vec[None], a2[None])[0, 0])),
                )
        except Exception as exc:  # noqa: BLE001 - reported as a failed metric
            log.exception("Answer evaluation failed")
            reason = f"Answer evaluation failed ({type(exc).__name__})"
            return AnswerResult(
                status=Status.FAILED,
                reason=reason,
                relevance=Metric.failed("Topic relevance", reason),
                coverage=Metric.failed("Concept coverage", reason),
                answer_score=Metric.failed("Answer content", reason),
            )

        try:
            rubric = self._rubric_provider()
            weights = rubric.weights_for(question.category)
        except RubricError as exc:
            log.exception("Rubric could not be loaded")
            reason = f"The answer rubric is invalid: {exc}"
            return AnswerResult(
                status=Status.FAILED, reason=reason,
                relevance=Metric.failed("Topic relevance", reason),
                coverage=Metric.failed("Concept coverage", reason),
                answer_score=Metric.failed("Answer content", reason),
            )

        relevance = Metric.available("Topic relevance", relevance_value)
        coverage = Metric.available("Concept coverage", coverage_value)
        criteria = self._criteria(text, question, relevance_value, coverage_value,
                                  concept_scores, weights, rubric.target_words)
        score = rubric_answer_score(criteria)
        assert score.value is not None
        strengths, improvements = strengths_and_improvements(criteria, concept_scores,
                                                             rubric.target_words)
        return AnswerResult(
            status=Status.AVAILABLE,
            relevance=relevance,
            coverage=coverage,
            reference_similarity=reference,
            answer_score=score,
            concept_scores=concept_scores,
            feedback=feedback_for(score.value),
            criteria=criteria,
            strengths=strengths,
            improvements=improvements,
        )

    @staticmethod
    def _criteria(text: str, question: Question, relevance: float, coverage: float,
                  concept_scores: dict[str, int], weights: dict[str, float],
                  target_words: int) -> list[CriterionScore]:
        full = sum(1 for v in concept_scores.values() if v == 100)
        partial = sum(1 for v in concept_scores.values() if 0 < v < 100)
        total = len(concept_scores)
        features = {
            "relevance": (relevance,
                          f"Semantic similarity to the question: {relevance:.0f}/100 "
                          "(uncalibrated; on-topic answers typically land around 40-70)."),
            "coverage": (coverage,
                         f"{full} of {total} expected concepts stated explicitly"
                         + (f", {partial} by paraphrase" if partial else "") + "."),
        }
        for key, fn in (("elaboration", lambda t: elaboration(t, target_words)),
                        ("structure", structure), ("examples", examples)):
            feature = fn(text)
            features[key] = (feature.score, feature.explanation)
        return [
            CriterionScore(
                key=key,
                label=CRITERIA[key],
                metric=Metric.available(CRITERIA[key], clamp_0_100(features[key][0])),
                weight=weights[key],
                explanation=features[key][1],
            )
            for key in CRITERIA
        ]


STRENGTH_THRESHOLD = 75.0
IMPROVE_THRESHOLD = 50.0

_STRENGTH_TEXT = {
    "relevance": "Stayed on topic and answered the question asked.",
    "coverage": "Covered the key concepts the question expects.",
    "elaboration": "Developed the answer fully.",
    "structure": "Signposted the reasoning clearly.",
    "examples": "Backed the answer with concrete detail.",
}


def strengths_and_improvements(criteria: list[CriterionScore], concept_scores: dict[str, int],
                               target_words: int) -> tuple[list[str], list[str]]:
    """Plain-language strengths (score >= 75) and improvements (score < 50)."""
    strengths: list[str] = []
    improvements: list[str] = []
    missing = [name for name, v in concept_scores.items() if v == 0]
    tips = {
        "relevance": "Answer the question directly in your first sentence, then expand.",
        "coverage": ("Cover the expected points" + (f" — e.g. {', '.join(missing[:3])}" if missing else "")
                     + "."),
        "elaboration": f"Develop the answer further: about {target_words} words "
                       "(roughly 30 seconds of speech) gives room to explain.",
        "structure": "Signpost your reasoning with words like 'first', 'because', "
                     "'however' and 'overall'.",
        "examples": "Add a concrete example or a specific figure from your own experience.",
    }
    for c in criteria:
        if not c.metric.is_available or c.metric.value is None:
            continue
        if c.metric.value >= STRENGTH_THRESHOLD:
            strengths.append(_STRENGTH_TEXT[c.key])
        elif c.metric.value < IMPROVE_THRESHOLD:
            improvements.append(tips[c.key])
    return strengths, improvements


def evaluate_answer(question: Question, transcript: Optional[str]) -> AnswerResult:
    """Evaluate with the application's shared sentence model."""
    from backend.core.models import get_registry

    return AnswerEvaluator(lambda: get_registry().get("sentence")).evaluate(question, transcript)
