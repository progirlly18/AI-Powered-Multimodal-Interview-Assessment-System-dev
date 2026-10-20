"""
Rendering helpers for the Streamlit app.

Only native Streamlit components are used (no raw HTML), so user-provided
text such as names and transcripts is always escaped by Streamlit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st

from backend.core.results import Metric, QuestionResult, Status, round_half_up
from backend.core.states import DeviceStatus, RunState
from backend.vision.emotion_detector import BBox

MAX_DISPLAY_WIDTH = 640

STATE_LABELS: dict[RunState, tuple[str, str]] = {
    RunState.IDLE: ("info", "Ready to record"),
    RunState.INITIALIZING: ("info", "Starting camera and microphone…"),
    RunState.COUNTDOWN: ("warning", "Get ready — recording starts soon"),
    RunState.RECORDING: ("error", "🔴 Recording — camera and microphone are on"),
    RunState.PROCESSING: ("info", "Processing — camera and microphone are off"),
    RunState.COMPLETED: ("success", "Completed"),
    RunState.UNAVAILABLE: ("error", "Device unavailable"),
    RunState.FAILED: ("error", "Failed"),
    RunState.CANCELLED: ("warning", "Cancelled"),
}

DEVICE_LABELS: dict[DeviceStatus, tuple[str, str]] = {
    DeviceStatus.NOT_CHECKED: ("⚪", "Not checked"),
    DeviceStatus.INITIALIZING: ("🟡", "Checking…"),
    DeviceStatus.READY: ("🟢", "Ready"),
    DeviceStatus.UNAVAILABLE: ("🔴", "Unavailable"),
    DeviceStatus.FAILED: ("🔴", "Failed"),
}


def show_state(container, state: RunState, message: str) -> None:
    kind, label = STATE_LABELS[state]
    text = f"**{label}**  \n{message}"
    getattr(container, kind)(text)


def device_line(name: str, status: DeviceStatus, message: str = "") -> str:
    icon, label = DEVICE_LABELS[status]
    return f"{icon} **{name}:** {label}" + (f" — {message}" if message else "")


def prepare_display_frame(frame_bgr: np.ndarray, bbox: Optional[BBox]) -> np.ndarray:
    """Copy, draw the face box, downscale and convert BGR -> RGB for display."""
    import cv2

    frame = frame_bgr.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    h, w = frame.shape[:2]
    if w > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / w
        frame = cv2.resize(frame, (MAX_DISPLAY_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def metric_text(metric: Metric, suffix: str = "") -> str:
    if metric.is_available and metric.value is not None:
        return f"{round_half_up(metric.value)}{suffix}"
    return "Not available"


def metric_help(metric: Metric) -> Optional[str]:
    if metric.is_available:
        return None
    return metric.reason or metric.status.value


def render_result(result: QuestionResult) -> None:
    """Full breakdown of one answer."""
    score = result.score
    missing = score.missing

    if result.is_complete and not missing:
        st.success("Assessment complete — every component was measured.")
    else:
        names = ", ".join(c.label for c in missing) or "some measurements"
        st.warning(f"Incomplete assessment — not measured: {names}.")

    left, right = st.columns([1, 2])
    with left:
        if score.overall.is_available:
            st.metric(
                "Practice score" + (" (partial)" if score.is_partial else ""),
                f"{metric_text(score.overall)}/100",
            )
        else:
            st.metric("Practice score", "Withheld")
        st.caption(
            "A coaching aid for interview practice, not a hiring assessment. "
            "The weights are provisional and have not been validated."
        )
    with right:
        for line in score.explanation:
            st.write(line)

    rows = []
    for c in score.components:
        rows.append({
            "Component": c.label,
            "Score": metric_text(c.metric),
            "Weight": f"{c.effective_weight:.0%}" if c.metric.is_available else "—",
            "Contribution": (f"{c.contribution:.1f}" if c.contribution is not None else "—"),
            "Note": c.metric.reason or "",
        })
    st.dataframe(rows, hide_index=True, width="stretch")

    st.subheader("Suggestions")
    for tip in result.suggestions:
        st.write(f"• {tip}")
    for warning in result.warnings:
        st.info(warning)

    tab_answer, tab_speech, tab_vision, tab_tech = st.tabs(
        ["Answer content", "Speaking", "Camera", "Technical details"]
    )
    with tab_answer:
        a = result.answer
        if a.status is not Status.AVAILABLE:
            st.warning(a.reason or "Answer content could not be evaluated.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Topic relevance", metric_text(a.relevance), help=metric_help(a.relevance))
        c2.metric("Concept coverage", metric_text(a.coverage), help=metric_help(a.coverage))
        c3.metric("Answer content", metric_text(a.answer_score), help=metric_help(a.answer_score))
        st.caption(
            "Topic relevance is the semantic similarity between question and answer "
            "(uncalibrated). It shows how on-topic the answer is, not whether it is correct."
        )
        if a.concept_scores:
            st.write("**Expected concepts** (shown after answering)")
            st.dataframe(
                [{"Concept": k, "Coverage": v} for k, v in a.concept_scores.items()],
                hide_index=True, width="stretch",
            )
        if a.feedback:
            st.write(a.feedback)
        st.write("**Transcript**")
        st.write(result.speech.transcript or "_No transcript._")

    with tab_speech:
        s = result.speech
        if s.status is not Status.AVAILABLE:
            st.warning(s.reason or "Speech could not be analysed.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Speaking pace", metric_text(s.wpm, " wpm"), help=metric_help(s.wpm))
        c2.metric("Hesitation fillers", str(s.filler_count) if s.status is Status.AVAILABLE else "—")
        c3.metric("Delivery", metric_text(s.delivery_score), help=metric_help(s.delivery_score))
        if s.speaking_time_is_estimate:
            st.caption("Pace uses the whole recording length because speech timing was unavailable.")
        if s.possible_filler_counts:
            st.caption(
                "Words that are sometimes fillers (not penalised): "
                + ", ".join(f"'{k}' ×{v}" for k, v in s.possible_filler_counts.items())
            )
        if s.background_label:
            st.caption(f"Most prominent sound (informational): {s.background_label}")

    with tab_vision:
        v = result.vision
        if v.status is not Status.AVAILABLE:
            st.warning(v.reason or "Camera measurements are not available.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Face in view", metric_text(v.face_presence, "%"), help=metric_help(v.face_presence))
        c2.metric("Looking at camera", metric_text(v.eye_contact, "%"), help=metric_help(v.eye_contact))
        c3.metric("Facing camera", metric_text(v.head_facing, "%"), help=metric_help(v.head_facing))
        st.caption(
            "Gaze and head direction are rough webcam estimates for coaching only. "
            "They do not indicate honesty or ability."
        )
        if v.dominant_emotion:
            st.caption(
                f"Most frequent facial expression estimate (informational, not scored; "
                f"the model is about 51% accurate on its test set): {v.dominant_emotion}"
            )

    with tab_tech:
        st.write(f"Recording length: {result.duration_seconds:.0f} s")
        if result.av_start_offset_ms is not None:
            st.write(
                f"Measured audio/video start offset: {result.av_start_offset_ms:+.0f} ms "
                "(positive means video started later)."
            )
        st.write(f"Video frames captured: {result.vision.frames_captured}, "
                 f"analysed: {result.vision.frames_analysed}")
        if result.timings:
            st.dataframe(
                [{"Step": k, "Value": f"{v:.2f}"} for k, v in sorted(result.timings.items())],
                hide_index=True, width="stretch",
            )
