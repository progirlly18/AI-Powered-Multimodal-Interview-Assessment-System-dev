"""Rubric configuration and the transparent text heuristics."""

import copy

import pytest

from backend.ai.rubric import CRITERIA, RubricError, default_rubric, load_rubric, parse_rubric
from backend.ai.text_features import elaboration, examples, structure, word_count

VALID = {
    "schema_version": 1,
    "weights": {"technical": {"relevance": 0.2, "coverage": 0.2, "elaboration": 0.2,
                              "structure": 0.2, "examples": 0.2}},
    "elaboration": {"target_words": 50},
}


def test_shipped_rubric_is_valid():
    r = default_rubric()
    assert set(r.weights) == {"technical", "behavioural"}
    for table in r.weights.values():
        assert set(table) == set(CRITERIA)
        assert sum(table.values()) == pytest.approx(1.0)
    assert r.target_words == 60


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d["weights"]["technical"].update(examples=0.5), "sum to 1"),
    (lambda d: d["weights"]["technical"].pop("examples"), "exactly these criteria"),
    (lambda d: d["weights"]["technical"].update(examples=-0.1, coverage=0.5), ">= 0"),
    (lambda d: d["weights"]["technical"].update(examples="0.2"), ">= 0"),
    (lambda d: d.update(schema_version=2), "schema_version"),
    (lambda d: d.update(weights={}), "non-empty"),
    (lambda d: d["elaboration"].update(target_words=0), "target_words"),
])
def test_invalid_rubrics_rejected(mutate, message):
    data = copy.deepcopy(VALID)
    mutate(data)
    with pytest.raises(RubricError, match=message):
        parse_rubric(data)


def test_unknown_category():
    with pytest.raises(RubricError, match="No rubric weights"):
        parse_rubric(VALID).weights_for("trivia")


def test_missing_file(tmp_path):
    with pytest.raises(RubricError, match="not found"):
        load_rubric(tmp_path / "x.json")


# ---------------- text features ----------------

def test_word_count():
    assert word_count("It's 2 models' outputs.") == 4


@pytest.mark.parametrize("n,expected", [(0, 0), (30, 50), (60, 100), (120, 100)])
def test_elaboration(n, expected):
    f = elaboration(" ".join(["word"] * n), target_words=60)
    assert f.score == pytest.approx(expected)
    assert f"{n} words" in f.explanation


def test_elaboration_rejects_bad_target():
    with pytest.raises(ValueError):
        elaboration("x", 0)


@pytest.mark.parametrize("text,expected", [
    ("Overfitting is when a model memorises data.", 0),
    ("First, the model memorises data.", 50),
    ("First it memorises, because it is too complex.", 80),
    ("First it memorises because it is complex; however, regularization helps. Overall it works.", 100),
])
def test_structure(text, expected):
    assert structure(text).score == expected


def test_structure_whole_words_only():
    # "butter" and "thenceforth" are not connectives.
    assert structure("Butter thenceforth.").score == 0


@pytest.mark.parametrize("text,expected", [
    ("Overfitting is bad.", 0),
    ("For example, a deep tree overfits.", 70),
    ("For example, accuracy fell from 99% to 70%.", 100),
    ("In my project we used dropout of 0.5.", 100),
])
def test_examples(text, expected):
    f = examples(text)
    assert f.score == expected
    assert f.explanation
