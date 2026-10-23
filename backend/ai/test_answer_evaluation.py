"""
Manual check of the answer evaluator with the real sentence model.

    python -m backend.ai.test_answer_evaluation

Not an automated test (pytest only collects tests/). Automated tests with a
fake encoder live in tests/unit/test_answer_evaluator.py.
"""

from backend.ai.answer_evaluator import evaluate_answer
from backend.ai.question_bank import get_question


def show(role: str, question_index: int, answer: str) -> None:
    question = get_question(role, question_index)
    if question is None:
        print("Question not found.")
        return
    result = evaluate_answer(question, answer)
    print("\n" + "=" * 60)
    print(f"Role: {role}\nQuestion: {question.text}\nAnswer: {answer}\n")
    print(f"Status:            {result.status.value}")
    print(f"Topic relevance:   {result.relevance.display()}")
    print(f"Concept coverage:  {result.coverage.display()}")
    print(f"Answer content:    {result.answer_score.display()}")
    for concept, score in result.concept_scores.items():
        print(f"  {score:>3}/100  {concept}")
    print(f"Feedback: {result.feedback}")


if __name__ == "__main__":
    show(
        "Machine Learning Engineer",
        2,
        "Overfitting happens when a model performs very well on the training data "
        "but does not generalize to unseen data. We can reduce it using "
        "regularization, cross validation, or more training data.",
    )
