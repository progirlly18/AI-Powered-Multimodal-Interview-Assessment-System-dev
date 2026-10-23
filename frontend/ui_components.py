"""
Results views for the HUMAN NEXT app.

* ``render_result``: one answer, from a live ``QuestionResult``.
* ``render_dashboard``: a whole session, from a ``SessionRecord`` (used for
  both the end-of-interview summary and saved history, so they match).

Measured values, model estimates and unavailable data are labelled with
distinct chips. Text is escaped in ``frontend.components``.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import streamlit as st

from backend.ai.rubric import CRITERIA, default_rubric
from backend.core.results import Metric, QuestionResult, Status, round_half_up
from backend.scoring.scoring_engine import COMPONENTS
from backend.storage.records import SessionRecord
from backend.vision.emotion_detector import BBox
from frontend.components import (
    bar,
    bullet_list,
    card,
    chip,
    legend,
    para,
    score_ring,
    show,
)

MAX_DISPLAY_WIDTH = 640
DISCLAIMER = ("Prototype practice aid — scores are heuristic, not validated against human "
              "interviewers, and must not be used to make hiring decisions.")


def prepare_display_frame(frame_bgr: np.ndarray, bbox: Optional[BBox]) -> np.ndarray:
    """Copy, draw the face box, downscale and convert BGR -> RGB for display."""
    import cv2

    frame = frame_bgr.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 197, 54), 2)  # BGR of #36C5FF
    h, w = frame.shape[:2]
    if w > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / w
        frame = cv2.resize(frame, (MAX_DISPLAY_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def metric_text(metric: Metric, suffix: str = "") -> str:
    if metric.is_available and metric.value is not None:
        return f"{round_half_up(metric.value)}{suffix}"
    return "Not available"


def _value(metric: Metric) -> Optional[float]:
    return metric.value if metric.is_available else None


def _status_chip(complete: bool, has_score: bool, partial: bool) -> str:
    if not has_score:
        return chip("Score withheld", "err")
    if complete:
        return chip("Complete assessment", "ok")
    return chip("Partial assessment" if partial else "Incomplete", "warn")


# --------------------------------------------------
# One answer
# --------------------------------------------------

def render_result(result: QuestionResult) -> None:
    score = result.score
    show(legend())
    left, right = st.columns([1, 2], gap="large")
    with left:
        show(card(
            score_ring(_value(score.overall))
            + "<div style='text-align:center'>"
            + _status_chip(result.is_complete, score.overall.is_available, score.is_partial)
            + "</div>" + "".join(para(line) for line in score.explanation),
            label="Practice score", glow=True,
        ))
    with right:
        rows = []
        for c in score.components:
            if c.metric.is_available:
                note = f"Weight {c.effective_weight:.0%} · contributes {c.contribution:.1f} points"
            else:
                note = f"Not included — {c.metric.reason or c.metric.status.value}"
            rows.append(bar(c.label, _value(c.metric), note))
        show(card("".join(rows), label="Measured components",
                  title="How the score was built"))

    a = result.answer
    left, right = st.columns([3, 2], gap="large")
    with left:
        if a.criteria:
            rows = "".join(
                bar(f"{c.label}  ·  {c.weight:.0%}", _value(c.metric), c.explanation)
                for c in a.criteria
            )
            content = metric_text(a.answer_score)
            show(card(rows, label=f"Answer quality rubric · content score {content}/100",
                      title="Why the answer scored this way"))
        else:
            show(card(para(a.reason or "The answer content could not be evaluated."),
                      label="Answer quality rubric") )
    with right:
        show(card(bullet_list(a.strengths) or para("No criterion reached the strength threshold (75)."),
                  label="Strengths"))
        tips = result.suggestions or ["—"]
        show(card(bullet_list(tips), label="Recommendations"))
        for warning in result.warnings:
            st.info(warning)

    t1, t2, t3, t4, t5 = st.tabs(["Transcript & concepts", "Speaking", "Camera",
                                  "Expression (model estimate)", "Technical"])
    with t1:
        st.caption("Transcript (Whisper, measured)")
        st.write(result.speech.transcript or "_No transcript._")
        if a.concept_scores:
            st.caption("Expected concepts — shown only after answering")
            show("".join(bar(name, float(value)) for name, value in a.concept_scores.items()))
        st.caption("Topic relevance is semantic similarity to the question (uncalibrated); "
                   "it shows how on-topic an answer is, not whether it is correct.")
    with t2:
        s = result.speech
        if s.status is not Status.AVAILABLE:
            st.warning(s.reason or "Speech could not be analysed.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Speaking pace", metric_text(s.wpm, " wpm"))
        c2.metric("Hesitation fillers", str(s.filler_count) if s.status is Status.AVAILABLE else "—")
        c3.metric("Delivery", metric_text(s.delivery_score))
        if s.speaking_time_is_estimate:
            st.caption("Pace uses the whole recording length because speech timing was unavailable.")
        if s.possible_filler_counts:
            st.caption("Sometimes-filler words (not penalised): "
                       + ", ".join(f"'{k}' ×{v}" for k, v in s.possible_filler_counts.items()))
        if s.background_label:
            st.caption(f"Most prominent sound (informational): {s.background_label}")
    with t3:
        v = result.vision
        if v.status is not Status.AVAILABLE:
            st.warning(v.reason or "Camera measurements are not available.")
        show(bar("Face in view", _value(v.face_presence), style="blue")
             + bar("Looking at the camera", _value(v.eye_contact), style="blue")
             + bar("Facing the camera", _value(v.head_facing), style="blue"))
        st.caption("Rough webcam estimates for coaching only; they do not indicate honesty or ability.")
    with t4:
        if v.emotion_counts:
            total = sum(v.emotion_counts.values())
            show("".join(bar(label, 100 * n / total, right=f"{n} of {total}")
                         for label, n in Counter(v.emotion_counts).most_common()))
            st.caption(f"Share of the {total} expression predictions made during the answer. "
                       "Uncertain model output (about 51% accuracy on its test set), culturally "
                       "variable, and **not scored**.")
        else:
            st.caption("No expression predictions for this answer.")
    with t5:
        st.write(f"Recording length: {result.duration_seconds:.0f} s · frames captured "
                 f"{v.frames_captured}, analysed {v.frames_analysed}")
        if result.av_start_offset_ms is not None:
            st.write(f"Measured audio/video start offset: {result.av_start_offset_ms:+.0f} ms "
                     "(positive means video started later).")
        if result.timings:
            st.dataframe([{"Step": k, "Value": f"{val:.2f}"} for k, val in sorted(result.timings.items())],
                         hide_index=True, width="stretch")


# --------------------------------------------------
# Whole session
# --------------------------------------------------

def render_dashboard(record: SessionRecord) -> None:
    answered = [a for a in record.answers if a.state != "not answered"]
    complete = [a for a in record.answers if a.complete]
    left, mid, right = st.columns([1.1, 1, 1], gap="large")
    with left:
        caption = "AVERAGE SCORE"
        show(card(score_ring(record.average, caption)
                  + para(f"Average over {len(record.scored)} of {len(record.answers)} questions "
                         "with a practice score.")
                  + (para("Some answers are unscored or partial, so this average is not "
                          "comparable with a complete interview.")
                     if len(record.scored) < len(record.answers)
                     or any(a.is_partial for a in record.answers if a.overall is not None) else ""),
                  label="Interview summary", glow=True))
    with mid:
        show(card(
            "".join(bar(f"Q{a.position}", a.overall,
                        note=(a.question_text[:70] + ("…" if len(a.question_text) > 70 else ""))
                        + (" · partial" if a.is_partial and a.overall is not None else "")
                        + ("" if a.overall is not None else f" · {a.state}"))
                    for a in record.answers),
            label="Question scores"))
    with right:
        strengths = Counter(s for a in record.answers for s in a.strengths)
        improvements = Counter(i for a in record.answers for i in a.improvements)
        show(card(bullet_list([s for s, _ in strengths.most_common(4)])
                  or para("No criterion reached the strength threshold yet."),
                  label="Strengths"))
        show(card(bullet_list([i for i, _ in improvements.most_common(4)])
                  or para("No recurring weaknesses were detected."),
                  label="Top recommendations"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Questions answered", f"{len(answered)} / {len(record.answers)}")
    c2.metric("With a practice score", str(len(record.scored)))
    c3.metric("Complete assessments", str(len(complete)))

    st.markdown("#### Question-by-question feedback")
    for a in record.answers:
        title = f"Q{a.position} · {a.question_text}"
        with st.expander(title, expanded=False):
            if a.overall is None:
                st.warning(f"No practice score — {a.message}")
            else:
                st.markdown(f"**Practice score:** {round_half_up(a.overall)}/100"
                            + (" (partial)" if a.is_partial else ""))
            for line in a.explanation:
                st.caption(line)
            if a.components:
                show("".join(bar(label, value, note=a.component_notes.get(label, ""))
                             for label, value in a.components.items()))
            if a.criteria:
                st.caption("Answer quality rubric")
                show("".join(bar(f"{c.label} · {c.weight:.0%}", c.score, c.explanation)
                             for c in a.criteria))
            if a.strengths:
                show(card(bullet_list(a.strengths), label="Strengths"))
            if a.suggestions:
                show(card(bullet_list(a.suggestions), label="Recommendations"))
            if a.transcript:
                st.caption("Transcript")
                st.write(a.transcript)

    with st.expander("How scores are calculated"):
        st.markdown("**Practice score per answer** — weighted average of the measured components; "
                    "missing components are left out and the weights rescaled (labelled *partial*). "
                    "Without answer content the score is withheld.")
        st.dataframe([{"Component": c.label, "Weight": f"{c.weight:.0f}%",
                       "Required": "yes" if c.required else "no"} for c in COMPONENTS],
                     hide_index=True, width="stretch")
        rubric = default_rubric()
        st.markdown("**Answer content** — weighted rubric (data/rubric.json):")
        st.dataframe([{"Criterion": CRITERIA[k],
                       **{cat.capitalize(): f"{w[k]:.0%}" for cat, w in rubric.weights.items()}}
                      for k in CRITERIA], hide_index=True, width="stretch")
        st.caption("Facial-expression estimates and background sound are informational and never "
                   "scored. Full specification: docs/SCORING.md.")
    st.caption(DISCLAIMER)
