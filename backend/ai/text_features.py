"""
Transparent text heuristics for answer quality (English only).

Each function returns a 0-100 score and a short explanation of exactly what
was found, so the user can see why they got the score. These are pattern
counts, not language understanding: they reward visible structure and
concrete detail, and they can be fooled by a speaker who uses the words
without the substance. They are documented as heuristics in docs/SCORING.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Connectives grouped by the kind of reasoning they signal.
STRUCTURE_MARKERS: dict[str, tuple[str, ...]] = {
    "sequence": ("first", "firstly", "second", "secondly", "then", "next", "finally", "lastly"),
    "cause": ("because", "therefore", "so that", "as a result", "this means", "which means", "since"),
    "contrast": ("however", "but", "whereas", "on the other hand", "although", "instead"),
    "summary": ("in summary", "overall", "to conclude", "in short", "to sum up"),
}
STRUCTURE_POINTS = {0: 0.0, 1: 50.0, 2: 80.0}  # 3 or more kinds -> 100

EXAMPLE_MARKERS: tuple[str, ...] = (
    "for example", "for instance", "such as", "e.g", "in my project", "in my internship",
    "i worked on", "i built", "we built", "i used", "we used", "a real example",
)


@dataclass(frozen=True)
class FeatureScore:
    score: float
    explanation: str


def _find(text: str, phrase: str) -> int:
    pattern = r"(?<![\w'])" + re.escape(phrase) + r"(?![\w'])"
    return len(re.findall(pattern, text))


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?", text))


def elaboration(text: str, target_words: int) -> FeatureScore:
    """How fully the answer is developed: words relative to a target length."""
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    n = word_count(text)
    score = min(100.0, 100.0 * n / target_words)
    if n >= target_words:
        why = f"{n} words — a fully developed answer (target about {target_words})."
    else:
        why = f"{n} words — short of the ~{target_words} words of a fully developed answer."
    return FeatureScore(score, why)


def structure(text: str) -> FeatureScore:
    """Kinds of connectives used (sequence, cause, contrast, summary)."""
    lowered = text.lower()
    found = {
        kind: [m for m in markers if _find(lowered, m)]
        for kind, markers in STRUCTURE_MARKERS.items()
    }
    kinds = [kind for kind, hits in found.items() if hits]
    score = STRUCTURE_POINTS.get(len(kinds), 100.0)
    if not kinds:
        why = ("No connecting words found (e.g. 'first', 'because', 'however', "
               "'overall'); the reasoning may be hard to follow.")
    else:
        detail = "; ".join(f"{k}: '{found[k][0]}'" for k in kinds)
        why = f"Uses {len(kinds)} kind(s) of connective — {detail}."
    return FeatureScore(score, why)


def examples(text: str) -> FeatureScore:
    """Signals of concrete evidence: example phrases and specific numbers."""
    lowered = text.lower()
    markers = [m for m in EXAMPLE_MARKERS if _find(lowered, m)]
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?", text)
    signals = len(markers) + (1 if numbers else 0)
    score = 0.0 if signals == 0 else 70.0 if signals == 1 else 100.0
    parts = []
    if markers:
        parts.append("example phrases: " + ", ".join(f"'{m}'" for m in markers[:3]))
    if numbers:
        parts.append("specific figures: " + ", ".join(numbers[:3]))
    why = ("No concrete example or specific detail found." if not parts
           else "Concrete detail found — " + "; ".join(parts) + ".")
    return FeatureScore(score, why)
