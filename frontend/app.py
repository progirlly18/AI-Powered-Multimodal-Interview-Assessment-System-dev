"""
AI Interview Practice — Streamlit front end.

Run from the project folder:

    streamlit run frontend/app.py

Pages follow ``backend.core.session.Stage``. Recording runs in a worker
thread; this script only polls its ``LiveStatus`` and never receives
callbacks from other threads.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="AI Interview Practice", page_icon="🎤", layout="wide")

# Light imports only: heavy models load on the readiness page.
from backend.ai.question_bank import (  # noqa: E402
    Question,
    QuestionBankError,
    get_question_by_id,
    get_questions,
    roles,
)
from backend.audio.recorder import check_microphone  # noqa: E402
from backend.config import ConfigError, load_settings  # noqa: E402
from backend.core.results import round_half_up  # noqa: E402
from backend.core.session import MAX_QUESTIONS, InterviewSession, Stage  # noqa: E402
from backend.core.states import DeviceStatus, InvalidTransitionError, RunState  # noqa: E402
from backend.logging_setup import configure_logging  # noqa: E402
from backend.scoring.scoring_engine import session_summary  # noqa: E402
from backend.services.interview_service import (  # noqa: E402
    InterviewService,
    LiveStatus,
    RunOutcome,
)
from backend.vision.capture import OpenCVCamera, check_camera  # noqa: E402
from backend.vision.guidance import guidance, latest_labels  # noqa: E402
from frontend.ui_components import (  # noqa: E402
    device_line,
    metric_text,
    prepare_display_frame,
    render_result,
    show_state,
)


# --------------------------------------------------
# Process-wide resources
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_service() -> InterviewService:
    settings = load_settings()
    configure_logging(settings.log_dir)
    return InterviewService(settings=settings)


@dataclass
class RunHandle:
    question_id: str
    thread: threading.Thread
    status: LiveStatus
    cancel: threading.Event
    outcome: Optional[RunOutcome] = None


def start_run(service: InterviewService, question: Question, duration_s: int,
              use_camera: bool) -> RunHandle:
    status = LiveStatus()
    cancel = threading.Event()
    handle = RunHandle(question.id, None, status, cancel)  # type: ignore[arg-type]

    def work() -> None:
        try:
            handle.outcome = service.run_question(
                question, duration_s, use_camera=use_camera, status=status, cancel=cancel
            )
        except Exception:  # noqa: BLE001 - logged by the service's logger
            import logging

            logging.getLogger("backend.app").exception("Run thread failed")
            handle.outcome = RunOutcome(RunState.FAILED, "Something went wrong. Please try again.")

    handle.thread = threading.Thread(target=work, name="interview-run", daemon=True)
    handle.thread.start()
    return handle


# --------------------------------------------------
# State
# --------------------------------------------------

ss = st.session_state
if "session" not in ss:
    ss.session = InterviewSession()
if "run" not in ss:
    ss.run = None
if "readiness" not in ss:
    ss.readiness = {}
session: InterviewSession = ss.session

try:
    settings = load_settings()
except ConfigError as exc:
    st.error(f"Configuration error: {exc}. Fix the environment variable and restart the app.")
    st.stop()


def go(action) -> None:
    """Apply a session transition and rerun; show a message if it is invalid."""
    try:
        action()
    except (InvalidTransitionError, ValueError) as exc:
        st.error(str(exc))
        return
    st.rerun()


# --------------------------------------------------
# Pages
# --------------------------------------------------

def page_welcome() -> None:
    st.title("🎤 AI Interview Practice")
    st.write(
        "Practise answering interview questions out loud. The app records your "
        "answer with your microphone and (optionally) your camera, then gives "
        "feedback on the content of your answer, your speaking pace and whether "
        "you faced the camera."
    )
    st.subheader("How it works")
    st.markdown(
        "1. Choose a role and questions.\n"
        "2. The app checks your camera, microphone and AI models.\n"
        "3. You agree to the recording, read the question and press **Start recording**.\n"
        "4. After a short countdown you answer for the time you chose.\n"
        "5. You get a transparent breakdown of every score and practical suggestions."
    )
    st.info(
        "This is a practice and coaching tool. Its scores are heuristic, have not "
        "been validated against human interviewers and must not be used to make "
        "hiring decisions."
    )
    if st.button("Get started", type="primary"):
        go(session.begin_setup)


def page_setup() -> None:
    st.title("Interview setup")
    try:
        role_list = roles()
    except QuestionBankError as exc:
        st.error(f"The question bank could not be loaded: {exc}")
        return
    role = st.selectbox("Role", role_list,
                        index=role_list.index(session.role) if session.role in role_list else 0)
    questions = get_questions(role)
    labels = {q.id: f"{q.text} ({q.difficulty})" for q in questions}
    default = [qid for qid in session.question_ids if qid in labels] or [questions[0].id]
    chosen = st.multiselect(
        "Questions (answered in this order)", list(labels), default=default,
        format_func=labels.get, max_selections=MAX_QUESTIONS,
    )
    duration = st.slider("Answer time per question (seconds)", 10, 120,
                         session.duration_s, step=5)
    use_camera = st.toggle(
        "Use the camera for face and gaze feedback", value=session.use_camera,
        help="Turn off to practise with audio only. Camera-based scores are then not calculated.",
    )
    c1, c2 = st.columns(2)
    if c1.button("Continue", type="primary"):
        ss.readiness = {}
        go(lambda: session.configure(role, chosen, duration, use_camera))
    if c2.button("Back"):
        go(session.back_to_welcome)


def _model_note(service: InterviewService, name: str) -> str:
    info = service.registry.info(name)
    if info.status is DeviceStatus.READY and info.load_seconds is not None:
        return f"loaded in {info.load_seconds:.1f} s"
    if info.status is DeviceStatus.FAILED:
        # Short, actionable text; the full error is in logs/app.log.
        spec_help = "Check your internet connection for the first download, then retry."
        return f"could not be loaded. {spec_help} Details are in logs/app.log."
    return ""


def _models_ready(service: InterviewService, use_camera: bool) -> bool:
    return all(service.registry.is_loaded(n) for n in service.required_models(use_camera))


def page_readiness(service: InterviewService) -> None:
    st.title("Readiness check")
    st.write("Checks run on this computer. Nothing is recorded during these checks.")
    names = service.required_models(session.use_camera)

    st.subheader("AI models")
    model_list = st.empty()
    model_list.markdown("\n\n".join(
        device_line(service.registry.info(n).label, service.registry.info(n).status,
                    _model_note(service, n))
        for n in names
    ))
    if not _models_ready(service, session.use_camera):
        st.caption("The first load can take a few minutes; models are downloaded on first use.")
        if st.button("Load models", type="primary"):
            model_list.empty()  # the progress box below replaces the static list
            with st.status("Loading models…", expanded=True) as box:
                for name in names:
                    info = service.registry.info(name)
                    if service.registry.is_loaded(name):
                        continue
                    box.write(f"Loading {info.label}…")
                    start = time.perf_counter()
                    service.load_models([name])
                    info = service.registry.info(name)
                    if info.status is DeviceStatus.READY:
                        box.write(f"✅ {info.label} ({time.perf_counter() - start:.1f} s)")
                    else:
                        box.write(f"❌ {info.label} could not be loaded.")
                box.update(label="Model loading finished", state="complete")
            st.rerun()

    st.subheader("Devices")
    devices = ["Microphone"] + (["Camera"] if session.use_camera else [])
    for device in devices:
        status, message = ss.readiness.get(device, (DeviceStatus.NOT_CHECKED, ""))
        st.markdown(device_line(device, status, message))
    if st.button("Check devices"):
        with st.spinner("Checking devices…"):
            ok, msg = check_microphone()
            ss.readiness["Microphone"] = (DeviceStatus.READY if ok else DeviceStatus.UNAVAILABLE, msg)
            if session.use_camera:
                ok, msg = check_camera(OpenCVCamera(settings.camera_index))
                ss.readiness["Camera"] = (DeviceStatus.READY if ok else DeviceStatus.UNAVAILABLE, msg)
        st.rerun()

    devices_ready = all(ss.readiness.get(d, (None,))[0] is DeviceStatus.READY for d in devices)
    models_ready = _models_ready(service, session.use_camera)
    camera_failed = session.use_camera and \
        ss.readiness.get("Camera", (None,))[0] is DeviceStatus.UNAVAILABLE

    if camera_failed:
        st.warning(
            "The camera is not available. You can fix it and check again, or continue "
            "with audio only (camera-based scores will not be calculated)."
        )
        if st.button("Continue with audio only"):
            session.use_camera = False
            ss.readiness.pop("Camera", None)
            st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("Continue", type="primary", disabled=not (devices_ready and models_ready)):
        go(session.readiness_confirmed)
    if c2.button("Back to setup"):
        go(session.restart)
    if not (devices_ready and models_ready):
        st.caption("Continue is enabled once every model is loaded and every device is ready.")


def page_consent() -> None:
    st.title("Before you start")
    camera_text = "your camera and microphone" if session.use_camera else "your microphone"
    st.markdown(
        f"- Each answer is recorded with {camera_text} **only while the red "
        "'Recording' indicator is shown**.\n"
        "- A countdown appears before every recording; nothing is recorded during it.\n"
        "- Recordings are processed on this computer. They are not uploaded.\n"
        "- Audio and video are kept in memory only and discarded after analysis"
        + (" — **except that INTERVIEW_AI_KEEP_RECORDINGS is on, so audio is saved to "
           "the recordings folder**" if settings.keep_recordings else "")
        + ".\n"
        "- Your transcript and results stay in this browser session and are cleared "
        "when you start a new interview or close the app.\n"
        "- Results are coaching feedback, not a hiring assessment."
    )
    agreed = st.checkbox("I understand and agree to be recorded for this practice session.")
    c1, c2 = st.columns(2)
    if c1.button("Start interview", type="primary", disabled=not agreed):
        go(session.give_consent)
    if c2.button("Back to setup"):
        go(session.restart)


def _live_view(handle: RunHandle, use_camera: bool) -> None:
    """Poll the worker until it finishes, updating the page at a limited rate."""
    state_box = st.empty()
    timer_box = st.empty()
    progress_box = st.empty()
    video_col, info_col = st.columns([3, 2]) if use_camera else (None, st.container())
    video_box = video_col.empty() if video_col is not None else None
    with info_col:
        guidance_box = st.empty()
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        face_box, eye_box, head_box, emotion_box = m1.empty(), m2.empty(), m3.empty(), m4.empty()
        st.caption("Live values are rough estimates. Final results are calculated after recording.")

    interval = 1.0 / max(1.0, settings.display_fps)
    duration = float(session.duration_s)
    last_frame_id = None
    last_header = None
    last_analysis_id = None
    audio_hint_shown = False
    while True:
        snap = handle.status.snapshot()
        whole_seconds = int(snap.seconds_left) + 1 if snap.seconds_left is not None else None
        header = (snap.state, snap.message, whole_seconds)
        if header != last_header:  # only send changes to the browser
            last_header = header
            show_state(state_box, snap.state, snap.message)
            if snap.state is RunState.COUNTDOWN and whole_seconds is not None:
                timer_box.markdown(f"## Starting in {whole_seconds}…")
                progress_box.empty()
            elif snap.state is RunState.RECORDING and snap.seconds_left is not None:
                timer_box.markdown(f"## ⏱ {whole_seconds} s left")
                progress_box.progress(min(1.0, max(0.0, 1 - snap.seconds_left / duration)))
            elif snap.state is RunState.PROCESSING:
                timer_box.markdown("## ⏳ Processing…")
                progress_box.empty()
            else:
                timer_box.empty()

        if video_box is not None and snap.frame is not None and id(snap.frame) != last_frame_id:
            last_frame_id = id(snap.frame)
            bbox = snap.analysis.bbox if snap.analysis is not None else None
            video_box.image(prepare_display_frame(snap.frame, bbox), channels="RGB",
                            output_format="JPEG", width="stretch")
        if use_camera and snap.state is RunState.RECORDING and id(snap.analysis) != last_analysis_id:
            last_analysis_id = id(snap.analysis)
            level, text = guidance(snap.recent)
            (guidance_box.success if level == "ok" else
             guidance_box.warning if level == "warning" else guidance_box.info)(text)
            labels = latest_labels(snap.analysis)
            face_box.metric("Face", labels["face"])
            eye_box.metric("Gaze (estimate)", labels["eye"])
            head_box.metric("Head (estimate)", labels["head"])
            emotion_box.metric("Expression (info only)", labels["emotion"])
        elif not use_camera and not audio_hint_shown:
            audio_hint_shown = True
            guidance_box.info("Audio-only mode: speak clearly towards your microphone.")

        if not handle.thread.is_alive():
            break
        time.sleep(interval)


def page_question(service: InterviewService) -> None:
    qid = session.current_question_id
    question = get_question_by_id(qid) if qid else None
    if question is None:
        st.error("This question is no longer in the question bank.")
        if st.button("Back to setup"):
            go(session.restart)
        return
    view = question.candidate_view()  # never shows concepts or reference answers

    st.caption(f"{session.progress_text} · {view.role} · {view.difficulty} · "
               f"{session.duration_s} s to answer"
               + ("" if session.use_camera else " · audio only"))
    st.subheader(view.text)

    handle: Optional[RunHandle] = ss.run
    if handle is not None and handle.question_id == qid:
        # A run exists for this question: keep showing it (also after a rerun
        # caused by a click), then record its outcome exactly once.
        if handle.thread.is_alive():
            if st.button("Stop recording", help="Cancels this answer. Nothing is saved."):
                handle.cancel.set()
            _live_view(handle, session.use_camera)
        handle.thread.join()
        outcome = handle.outcome or RunOutcome(RunState.FAILED, "The recording did not finish.")
        ss.run = None
        go(lambda: session.record_outcome(outcome))
        return

    st.write("Take a moment to plan your answer. When you press **Start recording**, "
             f"a {settings.countdown_s}-second countdown runs before recording begins.")
    c1, c2 = st.columns(2)
    if c1.button("Start recording", type="primary"):
        ss.run = start_run(service, question, session.duration_s, session.use_camera)
        st.rerun()
    if c2.button("End interview"):
        go(session.restart)


def page_review() -> None:
    qid = session.current_question_id
    outcome: Optional[RunOutcome] = session.outcomes.get(qid) if qid else None
    question = get_question_by_id(qid) if qid else None
    st.caption(session.progress_text)
    st.subheader(question.text if question else "Question")

    if outcome is None:
        st.error("No result for this question.")
    elif outcome.state is not RunState.COMPLETED or outcome.result is None:
        show_state(st, outcome.state, outcome.message)
        if outcome.device == "camera":
            st.write("You can fix the camera and retry, or switch to audio-only mode.")
            if st.button("Retry in audio-only mode"):
                session.use_camera = False
                go(session.retry_question)
    else:
        render_result(outcome.result)

    c1, c2, c3 = st.columns(3)
    if c1.button("Retry this question"):
        go(session.retry_question)
    if c2.button("Finish" if session.is_last_question else "Next question", type="primary"):
        go(session.next_question)
    if not session.is_last_question and c3.button("Finish now"):
        go(session.finish)


def _score_cell(result) -> str:
    if result is None:
        return "—"
    overall = result.score.overall
    if not overall.is_available:
        return "Withheld"
    return metric_text(overall) + (" (partial)" if result.score.is_partial else "")


def page_summary() -> None:
    st.title("Interview summary")
    rows = []
    overalls = []
    for i, qid in enumerate(session.question_ids, start=1):
        question = get_question_by_id(qid)
        outcome: Optional[RunOutcome] = session.outcomes.get(qid)
        result = outcome.result if outcome else None
        if result is not None:
            overalls.append(result.score.overall)
        rows.append({
            "#": i,
            "Question": question.text if question else qid,
            "Status": "Not answered" if outcome is None else outcome.state.value.capitalize(),
            "Complete": "Yes" if result is not None and result.is_complete else "No",
            "Practice score": _score_cell(result),
        })
    st.dataframe(rows, hide_index=True, width="stretch")
    mean, scored, total = session_summary(overalls)
    if mean is None:
        st.warning("No question produced an overall score.")
    else:
        st.metric(f"Average practice score ({scored} of {len(session.question_ids)} questions scored)",
                  f"{round_half_up(mean)}/100")
        if scored < len(session.question_ids) or any(
            o is not None and o.result is not None and o.result.score.is_partial
            for o in session.outcomes.values()
        ):
            st.caption("Some questions are unscored or partially scored, so this average "
                       "is not comparable with a complete interview.")
    st.info("Use these results to decide what to practise next. They are not a hiring assessment.")
    if st.button("Start a new interview", type="primary"):
        go(session.restart)


# --------------------------------------------------
# Router
# --------------------------------------------------

service = get_service()
stage = session.stage
if stage is Stage.WELCOME:
    page_welcome()
elif stage is Stage.SETUP:
    page_setup()
elif stage is Stage.READINESS:
    page_readiness(service)
elif stage is Stage.CONSENT:
    page_consent()
elif stage is Stage.QUESTION:
    page_question(service)
elif stage is Stage.REVIEW:
    page_review()
elif stage is Stage.SUMMARY:
    page_summary()
