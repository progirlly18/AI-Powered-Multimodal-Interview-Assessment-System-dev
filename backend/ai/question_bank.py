"""
Question bank loaded from a validated JSON file (data/questions.json).

Each question has a stable ID. The UI should show candidates only the
``CandidateQuestion`` view, which contains no concepts, evidence phrases or
reference answers, so the grading rubric is never revealed before the
answer is recorded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from backend.config import PROJECT_ROOT

DEFAULT_QUESTION_FILE = PROJECT_ROOT / "data" / "questions.json"
DIFFICULTIES = ("Easy", "Medium", "Hard")
CATEGORIES = ("technical", "behavioural")
SCHEMA_VERSION = 1


class QuestionBankError(ValueError):
    """The question file is missing or malformed."""


@dataclass(frozen=True)
class Concept:
    name: str
    aliases: tuple[str, ...]
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Question:
    id: str
    role: str
    category: str
    difficulty: str
    text: str
    concepts: tuple[Concept, ...]
    reference_answer: Optional[str] = None

    def candidate_view(self) -> "CandidateQuestion":
        return CandidateQuestion(self.id, self.role, self.difficulty, self.text)


@dataclass(frozen=True)
class CandidateQuestion:
    """What may be shown to the candidate before they answer."""

    id: str
    role: str
    difficulty: str
    text: str


def _require(cond: bool, where: str, message: str) -> None:
    if not cond:
        raise QuestionBankError(f"{where}: {message}")


def _str_list(value: Any, where: str, field: str, allow_empty: bool) -> tuple[str, ...]:
    _require(isinstance(value, list), where, f"'{field}' must be a list")
    items = tuple(value)
    _require(allow_empty or len(items) > 0, where, f"'{field}' must not be empty")
    for item in items:
        _require(isinstance(item, str) and item.strip() != "", where,
                 f"'{field}' entries must be non-empty strings")
    return items


def parse_question_bank(data: Any) -> tuple[Question, ...]:
    """Validate parsed JSON and convert it to ``Question`` objects."""
    _require(isinstance(data, dict), "question bank", "top level must be an object")
    _require(data.get("schema_version") == SCHEMA_VERSION, "question bank",
             f"schema_version must be {SCHEMA_VERSION}")
    raw_questions = data.get("questions")
    _require(isinstance(raw_questions, list) and len(raw_questions) > 0,
             "question bank", "'questions' must be a non-empty list")

    seen: set[str] = set()
    questions: list[Question] = []
    for index, raw in enumerate(raw_questions):
        where = f"question #{index + 1}"
        _require(isinstance(raw, dict), where, "must be an object")
        qid = raw.get("id")
        _require(isinstance(qid, str) and qid.strip() != "", where, "'id' is required")
        where = f"question '{qid}'"
        _require(qid not in seen, where, "duplicate id")
        seen.add(qid)

        for field in ("role", "question"):
            _require(isinstance(raw.get(field), str) and raw[field].strip() != "",
                     where, f"'{field}' must be a non-empty string")
        _require(raw.get("difficulty") in DIFFICULTIES, where,
                 f"'difficulty' must be one of {DIFFICULTIES}")
        _require(raw.get("category") in CATEGORIES, where,
                 f"'category' must be one of {CATEGORIES}")
        reference = raw.get("reference_answer")
        _require(reference is None or (isinstance(reference, str) and reference.strip()),
                 where, "'reference_answer' must be null or a non-empty string")

        raw_concepts = raw.get("concepts")
        _require(isinstance(raw_concepts, list) and len(raw_concepts) > 0,
                 where, "'concepts' must be a non-empty list")
        concepts = []
        names: set[str] = set()
        for c in raw_concepts:
            _require(isinstance(c, dict), where, "each concept must be an object")
            name = c.get("name")
            _require(isinstance(name, str) and name.strip() != "", where,
                     "each concept needs a 'name'")
            _require(name not in names, where, f"duplicate concept '{name}'")
            names.add(name)
            concepts.append(
                Concept(
                    name=name,
                    aliases=_str_list(c.get("aliases"), where, "aliases", allow_empty=False),
                    evidence=_str_list(c.get("evidence", []), where, "evidence", allow_empty=True),
                )
            )

        questions.append(
            Question(
                id=qid,
                role=raw["role"],
                category=raw["category"],
                difficulty=raw["difficulty"],
                text=raw["question"].strip(),
                concepts=tuple(concepts),
                reference_answer=reference,
            )
        )
    return tuple(questions)


def load_question_bank(path: Path = DEFAULT_QUESTION_FILE) -> tuple[Question, ...]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise QuestionBankError(f"Question file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise QuestionBankError(f"Question file is not valid JSON: {exc}") from exc
    return parse_question_bank(data)


@lru_cache(maxsize=1)
def _default_bank() -> tuple[Question, ...]:
    return load_question_bank(DEFAULT_QUESTION_FILE)


def all_questions() -> tuple[Question, ...]:
    return _default_bank()


def roles() -> list[str]:
    """Roles in file order."""
    return list(dict.fromkeys(q.role for q in all_questions()))


def get_questions(role: str) -> list[Question]:
    """Questions for ``role`` in file order (empty list if the role is unknown)."""
    return [q for q in all_questions() if q.role == role]


def get_question(role: str, index: int) -> Optional[Question]:
    """The question at 0-based ``index`` for ``role``, or None if out of range."""
    questions = get_questions(role)
    if not 0 <= index < len(questions):
        return None
    return questions[index]


def get_question_by_id(question_id: str) -> Optional[Question]:
    return next((q for q in all_questions() if q.id == question_id), None)


if __name__ == "__main__":
    for role in roles():
        print(f"\n{role}")
        for i, q in enumerate(get_questions(role), start=1):
            print(f"  {i}. [{q.id}] {q.text} ({q.difficulty}, {len(q.concepts)} concepts)")
