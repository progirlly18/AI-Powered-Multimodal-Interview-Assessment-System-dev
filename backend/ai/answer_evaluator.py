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
* Similarity to a reference answer: only when the question has one
  (none do yet). Reported separately and not part of the answer score.

These are heuristics. Semantic similarity is not evidence of technical
correctness, completeness or sound reasoning.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from backend.ai.question_bank import Question
from backend.core.results import AnswerResult, Metric, Status, round_half_up
from backend.scoring.scoring_engine import answer_score, clamp_0_100

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

    def __init__(self, encoder_provider: Callable[[], Encoder]) -> None:
        # A provider (not the encoder itself) so the model loads lazily and
        # load failures surface as an evaluation failure.
        self._encoder_provider = encoder_provider

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

        relevance = Metric.available("Topic relevance", relevance_value)
        coverage = Metric.available("Concept coverage", coverage_value)
        score = answer_score(relevance, coverage)
        assert score.value is not None
        return AnswerResult(
            status=Status.AVAILABLE,
            relevance=relevance,
            coverage=coverage,
            reference_similarity=reference,
            answer_score=score,
            concept_scores=concept_scores,
            feedback=feedback_for(score.value),
        )


def evaluate_answer(question: Question, transcript: Optional[str]) -> AnswerResult:
    """Evaluate with the application's shared sentence model."""
    from backend.core.models import get_registry

    return AnswerEvaluator(lambda: get_registry().get("sentence")).evaluate(question, transcript)
