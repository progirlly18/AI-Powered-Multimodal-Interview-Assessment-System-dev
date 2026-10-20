"""
Command-line interview practice (no browser).

    python -m backend.app --role "Machine Learning Engineer" --question 1 --seconds 20

``--question`` is 1-based, as shown by ``python -m backend.ai.question_bank``.
Uses the same service, states and scoring as the Streamlit app.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from backend.ai.question_bank import get_question, roles
from backend.config import load_settings
from backend.core.results import Metric, round_half_up
from backend.core.states import RunState
from backend.logging_setup import configure_logging
from backend.services.interview_service import InterviewService, LiveStatus


def _fmt(metric: Metric, suffix: str = "") -> str:
    if metric.is_available and metric.value is not None:
        return f"{round_half_up(metric.value)}{suffix}"
    return f"not available ({metric.reason})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--role", default="Machine Learning Engineer", choices=roles())
    parser.add_argument("--question", type=int, default=1, help="1-based question number")
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--no-camera", action="store_true", help="audio-only mode")
    args = parser.parse_args(argv)

    question = get_question(args.role, args.question - 1)
    if question is None:
        print(f"No question {args.question} for role {args.role!r}.", file=sys.stderr)
        return 2

    settings = load_settings()
    configure_logging(settings.log_dir)
    service = InterviewService(settings=settings)

    print("Loading models (the first run can take several minutes)…")
    for name, info in service.load_models(service.required_models(not args.no_camera)).items():
        print(f"  {info.label}: {info.status.value}")
        if info.status.value != "ready":
            print("A required model is not available; see logs/app.log.", file=sys.stderr)
            return 1

    print(f"\nQuestion: {question.text}")
    input("Press ENTER to start (Ctrl+C cancels)…")

    status = LiveStatus()
    cancel = threading.Event()
    holder: dict = {}
    worker = threading.Thread(
        target=lambda: holder.setdefault(
            "outcome",
            service.run_question(question, args.seconds, use_camera=not args.no_camera,
                                 status=status, cancel=cancel),
        )
    )
    worker.start()
    last = None
    try:
        while worker.is_alive():
            snap = status.snapshot()
            line = f"[{snap.state.value}] {snap.message}"
            if snap.seconds_left is not None:
                line += f" {int(snap.seconds_left) + 1}s"
            if line != last:
                print(line)
                last = line
            time.sleep(0.2)
    except KeyboardInterrupt:
        cancel.set()
    worker.join()
    outcome = holder["outcome"]
    print(f"\nResult: {outcome.state.value} — {outcome.message}")
    if outcome.state is not RunState.COMPLETED or outcome.result is None:
        return 1

    r = outcome.result
    print(f"\nTranscript: {r.speech.transcript or '(none)'}")
    print(f"Speaking pace: {_fmt(r.speech.wpm, ' wpm')}; fillers: {r.speech.filler_count}")
    print(f"Looking at camera: {_fmt(r.vision.eye_contact, '%')}; "
          f"facing camera: {_fmt(r.vision.head_facing, '%')}")
    print(f"Answer content: {_fmt(r.answer.answer_score)} "
          f"(relevance {_fmt(r.answer.relevance)}, coverage {_fmt(r.answer.coverage)})")
    label = "Practice score (partial)" if r.score.is_partial else "Practice score"
    print(f"{label}: {_fmt(r.score.overall)}")
    for line in r.score.explanation:
        print(f"  {line}")
    print("\nSuggestions:")
    for tip in r.suggestions:
        print(f"  - {tip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
