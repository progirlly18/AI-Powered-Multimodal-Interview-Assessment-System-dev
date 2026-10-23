"""
Answer-quality rubric loaded from data/rubric.json.

Criteria (all 0-100):

* relevance   - semantic similarity between question and answer (uncalibrated)
* coverage    - expected concepts the answer demonstrates
* elaboration - how fully developed the answer is (length vs. a target)
* structure   - connectives that show sequence, cause, contrast or summary
* examples    - concrete examples or specific figures

Weights are configured per question category and must sum to 1.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.config import PROJECT_ROOT

DEFAULT_RUBRIC_FILE = PROJECT_ROOT / "data" / "rubric.json"

CRITERIA: dict[str, str] = {
    "relevance": "Relevance",
    "coverage": "Concept coverage",
    "elaboration": "Elaboration",
    "structure": "Structure",
    "examples": "Examples & evidence",
}


class RubricError(ValueError):
    """The rubric file is missing or malformed."""


@dataclass(frozen=True)
class Rubric:
    weights: dict[str, dict[str, float]]
    target_words: int

    def weights_for(self, category: str) -> dict[str, float]:
        if category not in self.weights:
            raise RubricError(f"No rubric weights for question category '{category}'.")
        return self.weights[category]


def parse_rubric(data: Any) -> Rubric:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise RubricError("rubric: schema_version must be 1")
    raw_weights = data.get("weights")
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise RubricError("rubric: 'weights' must be a non-empty object")
    weights: dict[str, dict[str, float]] = {}
    for category, table in raw_weights.items():
        if not isinstance(table, dict) or set(table) != set(CRITERIA):
            raise RubricError(
                f"rubric '{category}': needs exactly these criteria: {', '.join(CRITERIA)}"
            )
        values = {}
        for key, value in table.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise RubricError(f"rubric '{category}.{key}': weight must be a number >= 0")
            values[key] = float(value)
        if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6):
            raise RubricError(f"rubric '{category}': weights must sum to 1, got {sum(values.values()):.3f}")
        weights[category] = values
    target = (data.get("elaboration") or {}).get("target_words")
    if not isinstance(target, int) or isinstance(target, bool) or not 5 <= target <= 1000:
        raise RubricError("rubric: elaboration.target_words must be an integer between 5 and 1000")
    return Rubric(weights=weights, target_words=target)


def load_rubric(path: Path = DEFAULT_RUBRIC_FILE) -> Rubric:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise RubricError(f"Rubric file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RubricError(f"Rubric file is not valid JSON: {exc}") from exc
    return parse_rubric(data)


@lru_cache(maxsize=1)
def default_rubric() -> Rubric:
    return load_rubric(DEFAULT_RUBRIC_FILE)
