"""Question bank loading, validation and indexing."""

import copy
import json

import pytest

from backend.ai import question_bank as qb
from backend.ai.question_bank import QuestionBankError, load_question_bank, parse_question_bank

VALID = {
    "schema_version": 1,
    "questions": [
        {
            "id": "r-01", "role": "R", "category": "technical", "difficulty": "Easy",
            "question": "First?", "reference_answer": None,
            "concepts": [{"name": "A", "aliases": ["a"]}],
        },
        {
            "id": "r-02", "role": "R", "category": "behavioural", "difficulty": "Hard",
            "question": "Last?", "reference_answer": "Because.",
            "concepts": [{"name": "B", "aliases": ["b"], "evidence": ["bee"]}],
        },
    ],
}


def bank(**changes):
    data = copy.deepcopy(VALID)
    for path, value in changes.items():
        q = data["questions"][0]
        if value is _DELETE:
            q.pop(path)
        else:
            q[path] = value
    return data


_DELETE = object()


def test_parse_valid_bank():
    questions = parse_question_bank(VALID)
    assert [q.id for q in questions] == ["r-01", "r-02"]
    assert questions[1].concepts[0].evidence == ("bee",)
    assert questions[1].reference_answer == "Because."


@pytest.mark.parametrize("change,message", [
    ({"id": ""}, "'id' is required"),
    ({"question": "  "}, "'question'"),
    ({"difficulty": "Impossible"}, "'difficulty'"),
    ({"category": "trivia"}, "'category'"),
    ({"concepts": []}, "'concepts'"),
    ({"concepts": [{"name": "A", "aliases": []}]}, "'aliases' must not be empty"),
    ({"concepts": [{"name": "A", "aliases": ["a"]}, {"name": "A", "aliases": ["b"]}]}, "duplicate concept"),
    ({"reference_answer": ""}, "'reference_answer'"),
    ({"role": _DELETE}, "'role'"),
])
def test_invalid_entries_rejected(change, message):
    with pytest.raises(QuestionBankError, match=message):
        parse_question_bank(bank(**change))


def test_duplicate_ids_rejected():
    data = copy.deepcopy(VALID)
    data["questions"][1]["id"] = "r-01"
    with pytest.raises(QuestionBankError, match="duplicate id"):
        parse_question_bank(data)


@pytest.mark.parametrize("data", [[], {"schema_version": 2, "questions": []},
                                  {"schema_version": 1, "questions": []}])
def test_empty_or_wrong_schema_rejected(data):
    with pytest.raises(QuestionBankError):
        parse_question_bank(data)


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(QuestionBankError, match="not found"):
        load_question_bank(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(QuestionBankError, match="not valid JSON"):
        load_question_bank(bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(VALID), encoding="utf-8")
    assert len(load_question_bank(good)) == 2


# ---------------- the shipped bank ----------------

def test_shipped_bank_is_valid_and_complete():
    questions = qb.all_questions()
    assert len(questions) == 26
    assert len({q.id for q in questions}) == 26
    assert qb.roles() == ["Software Engineer", "AI Engineer", "Machine Learning Engineer",
                          "Data Analyst", "HR Interview"]
    # No reference answers were invented when the bank was moved to JSON.
    assert all(q.reference_answer is None for q in questions)


def test_first_and_last_question_indexing():
    role = "Machine Learning Engineer"
    questions = qb.get_questions(role)
    assert qb.get_question(role, 0).text == "What is supervised learning?"
    assert qb.get_question(role, len(questions) - 1).text == "What is cross validation?"
    assert qb.get_question(role, len(questions)) is None
    assert qb.get_question(role, -1) is None
    assert qb.get_question("Unknown role", 0) is None
    assert qb.get_questions("Unknown role") == []


def test_lookup_by_id():
    q = qb.get_question("Machine Learning Engineer", 2)
    assert qb.get_question_by_id(q.id) == q
    assert qb.get_question_by_id("does-not-exist") is None


def test_candidate_view_hides_rubric():
    q = qb.get_question("Machine Learning Engineer", 2)
    view = q.candidate_view()
    assert view.text == q.text
    assert not hasattr(view, "concepts")
    assert not hasattr(view, "reference_answer")


def test_json_preserves_original_rubric():
    """Spot-check that the export kept the hand-written evidence lists."""
    q = qb.get_question("Machine Learning Engineer", 2)
    reg = next(c for c in q.concepts if c.name == "Regularization")
    assert reg.evidence == ("regularization", "L1", "L2", "regularization penalty")
