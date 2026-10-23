"""
HUMAN NEXT — AI interview practice console (Streamlit front end).

Run from the project folder:

    streamlit run frontend/app.py

Pages follow ``backend.core.session.Stage``. Recording runs in a worker
thread; this script only polls its thread-safe ``LiveStatus`` and never
receives callbacks from other threads. Heavy models are cached once per
process (``ModelRegistry`` behind ``st.cache_resource``), so reruns never
reload them or reopen the camera.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="HUMAN NEXT · AI Interview Practice", page_icon="🔷",
                   layout="wide", initial_sidebar_state="collapsed")

# Light imports only: heavy models load on the system-check page.
from backend.ai.question_bank import (  # noqa: E402
    DIFFICULTIES,
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
from backend.services.interview_service import (  # noqa: E402
    InterviewService,
    LiveSnapshot,
    LiveStatus,
    RunOutcome,
)
from backend.auth.config import AuthSettings, load_auth_settings  # noqa: E402
from backend.auth.service import AuthService, build_auth_service  # noqa: E402
from backend.storage.history import LOCAL_OWNER, HistoryError, HistoryStore  # noqa: E402
from backend.storage.records import SessionRecord, build_session_record  # noqa: E402
from backend.storage.report import to_csv, to_html  # noqa: E402
from backend.vision.capture import OpenCVCamera, probe_camera  # noqa: E402
from backend.vision.emotion_smoothing import latest_raw, smooth_emotion  # noqa: E402
from backend.vision.eye_contact import LOOKING_CENTER  # noqa: E402
from backend.vision.guidance import OK, WARNING, guidance  # noqa: E402
from backend.vision.head_pose import CENTER  # noqa: E402
from frontend.components import (  # noqa: E402
    bar,
    brand_bar,
    bullet_list,
    card,
    chip,
    para,
    show,
    stage_placeholder,
    stepper,
)
from frontend.auth_view import render_sign_in, signed_in_chip  # noqa: E402
from frontend.theme import inject_theme  # noqa: E402
from frontend.ui_components import (  # noqa: E402
    prepare_display_frame,
    render_dashboard,
    render_result,
)

inject_theme()

STEPS = ["Configure", "System check", "Consent", "Interview", "Results"]


# --------------------------------------------------
# Process-wide resources
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_service() -> InterviewService:
    settings = load_settings()
    configure_logging(settings.log_dir)
    return InterviewService(settings=settings)


@st.cache_resource(show_spinner=False)
def get_auth_service(auth_settings: AuthSettings) -> AuthService:
    configure_logging(load_settings().log_dir)
    return build_auth_service(auth_settings)


def _auth_secrets() -> dict:
    """The [auth] section of .streamlit/secrets.toml, or {} if there is none."""
    try:
        return dict(st.secrets.get("auth", {}))
    except Exception:  # noqa: BLE001 - no secrets file is normal
        return {}


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
ss.setdefault("session", InterviewSession())
ss.setdefault("run", None)
ss.setdefault("readiness", {})
ss.setdefault("preview", None)
session: InterviewSession = ss.session

try:
    settings = load_settings()
    auth_settings = load_auth_settings(os.environ, _auth_secrets())
except ConfigError as exc:
    st.error(f"Configuration error: {exc}. Fix the setting and restart the app.")
    st.stop()


# --------------------------------------------------
# Sign-in gate: nothing below runs until the email is verified
# --------------------------------------------------

def sign_out(message: Optional[str] = None) -> None:
    for key in ("user", "auth_email", "run"):
        ss[key] = None
    ss.session = InterviewSession()
    ss.readiness = {}
    ss.preview = None
    if message:
        ss.auth_notice = message
    st.rerun()


if auth_settings.require_sign_in:
    auth = get_auth_service(auth_settings)
    ss.setdefault("user", None)
    # Expire old sign-ins, but never in the middle of a recording.
    if ss.user is not None and ss.get("run") is None and not auth.session_valid(ss.user):
        sign_out("Your sign-in expired. Please sign in again.")
    if ss.user is None:
        render_sign_in(auth)
        st.stop()
    owner = ss.user.email
else:
    owner = LOCAL_OWNER

history = HistoryStore(settings.history_path, owner)


def go(action) -> None:
    """Apply a session transition and rerun; show a message if it is invalid."""
    try:
        action()
    except (InvalidTransitionError, ValueError) as exc:
        st.error(str(exc))
        return
    ss.preview = None  # the camera preview never outlives the system-check page
    st.rerun()


def header(*extra_chips: str) -> None:
    chips = list(extra_chips)
    if session.session_id and session.stage not in (Stage.WELCOME, Stage.HISTORY):
        chips.insert(0, chip(f"Session {session.session_id}", "na"))
    user = ss.get("user")
    if user is None:
        show(brand_bar(chips))
        return
    chips.append(signed_in_chip(user.email))
    bar_col, button_col = st.columns([6, 1], vertical_alignment="center")
    with bar_col:
        show(brand_bar(chips))
    # Not offered mid-recording: signing out would orphan the running capture.
    if ss.get("run") is None and button_col.button("Sign out", width="stretch"):
        sign_out("You have been signed out.")


def record_for(include_transcripts: bool) -> SessionRecord:
    items = []
    for qid in session.question_ids:
        q = get_question_by_id(qid)
        items.append((qid, q.text if q else qid, session.outcomes.get(qid)))
    return build_session_record(session.session_id, session.created_at, session.role or "",
                                session.candidate, items, include_transcripts)


# --------------------------------------------------
# Landing
# --------------------------------------------------

def page_welcome() -> None:
    header(chip("Prototype", "na"), chip("Runs locally", "ok"))
    show(
        "<div class='hn-hero'><div class='hn-orb'></div>"
        "<div class='hn-label'>AI interview intelligence</div>"
        "<h1>Decode Your <span class='hn-grad-text'>Interview Potential</span></h1>"
        "<p>Answer real interview questions out loud. HUMAN NEXT listens, watches your on-camera "
        "presence and explains — criterion by criterion — what made the answer strong and what to "
        "practise next.</p></div>"
    )
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Start Interview", type="primary", width="stretch"):
        go(session.begin_setup)
    if c2.button("View history", width="stretch"):
        go(session.open_history)

    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        show(card(para("Gaze, head direction and face visibility are measured about three times a "
                       "second, with smoothed hints while you speak."),
                  label="01 · Vision", title="Live Behavioral Insights", cls="eq"))
    with f2:
        show(card(para("Whisper transcribes your answer; a semantic model and a transparent rubric "
                       "rate relevance, concept coverage, structure, elaboration and examples."),
                  label="02 · Language", title="AI Response Evaluation", cls="eq"))
    with f3:
        show(card(para("Every score shows its weight and its reason, with concrete strengths and "
                       "recommendations — exportable as a report."),
                  label="03 · Coaching", title="Personalized Feedback", cls="eq"))

    w, p = st.columns([3, 2], gap="medium")
    with w:
        show(card(bullet_list([
            "Configure — choose a role, difficulty and questions.",
            "System check — models load, camera and microphone are tested.",
            "Consent — see exactly when recording happens.",
            "Interview — countdown, then answer while live insights update.",
            "Results — explainable scores, feedback and a downloadable report.",
        ]), label="How it works", title="Five steps, about five minutes", cls="eq-lg"))
    with p:
        show(card(para("Camera and microphone are used only while the recording indicator is on. "
                       "Audio and video stay in memory on this computer and are discarded after "
                       "analysis; nothing is uploaded. History stores scores, not recordings.")
                  + para("Facial-expression estimates are shown as uncertain model output and are "
                         "never scored. This is a practice aid, not a hiring decision system."),
                  label="Privacy & limits", title="Your data stays here", cls="eq-lg"))


# --------------------------------------------------
# Setup (configure)
# --------------------------------------------------

def page_setup() -> None:
    header()
    show(stepper(STEPS, 0))
    try:
        role_list = roles()
    except QuestionBankError as exc:
        st.error(f"The question bank could not be loaded: {exc}")
        return
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Configure your interview")
        candidate = st.text_input("Name or session label (optional)", value=session.candidate or "",
                                  max_chars=60, help="Shown on your results and report only.")
        role = st.selectbox("Role", role_list,
                            index=role_list.index(session.role) if session.role in role_list else 0)
        all_questions = get_questions(role)
        present = [d for d in DIFFICULTIES if any(q.difficulty == d for q in all_questions)]
        difficulty = st.selectbox("Difficulty", ["All"] + present,
                                  help="Filters the question list below.")
        questions = [q for q in all_questions if difficulty == "All" or q.difficulty == difficulty]
        labels = {q.id: f"{q.text} · {q.difficulty}" for q in questions}
        default = [qid for qid in session.question_ids if qid in labels] or [questions[0].id]
        chosen = st.multiselect("Questions (answered in this order)", list(labels), default=default,
                                format_func=labels.get, max_selections=MAX_QUESTIONS)
        duration = st.slider("Answer time per question (seconds)", 10, 120, session.duration_s, step=5)
        use_camera = st.toggle("Use the camera for live behavioral insights", value=session.use_camera,
                               help="Off = audio-only practice; camera-based scores are not calculated.")
    with right:
        show(card(bullet_list([
            f"{len(chosen)} question(s) × {duration} s answers.",
            "Camera on: gaze, head direction and face visibility are measured." if use_camera
            else "Audio only: the score will be labelled partial.",
            "Answers are transcribed locally with Whisper.",
            "Questions are shown without their grading rubric.",
        ]), label="Session preview", title=role))
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Continue", type="primary", width="stretch"):
        ss.readiness = {}
        go(lambda: session.configure(role, chosen, duration, use_camera, candidate))
    if c2.button("Back", width="stretch"):
        go(session.back_to_welcome)


# --------------------------------------------------
# System check
# --------------------------------------------------

def _model_note(service: InterviewService, name: str) -> str:
    info = service.registry.info(name)
    if info.status is DeviceStatus.READY and info.load_seconds is not None:
        return f"loaded in {info.load_seconds:.1f} s"
    if info.status is DeviceStatus.FAILED:
        return "could not be loaded — see logs/app.log, fix, then retry"
    return "not loaded yet"


def _status_chip(status: DeviceStatus) -> str:
    return {
        DeviceStatus.READY: chip("Ready", "ok", dot=True),
        DeviceStatus.FAILED: chip("Failed", "err"),
        DeviceStatus.UNAVAILABLE: chip("Unavailable", "err"),
        DeviceStatus.INITIALIZING: chip("Loading", "live", dot=True),
    }.get(status, chip("Not checked", "na"))


def _models_ready(service: InterviewService, use_camera: bool) -> bool:
    return all(service.registry.is_loaded(n) for n in service.required_models(use_camera))


def page_readiness(service: InterviewService) -> None:
    header()
    show(stepper(STEPS, 1))
    names = service.required_models(session.use_camera)
    left, right = st.columns([3, 2], gap="large")
    with left:
        model_box = st.empty()
        rows = "".join(
            "<div class='hn-row'><div class='hn-row-head'><span>"
            + escape(service.registry.info(n).label)
            + "</span>" + _status_chip(service.registry.info(n).status) + "</div>"
            + f"<div class='hn-row-note'>{escape(_model_note(service, n))}</div></div>"
            for n in names
        )
        show(card(rows, label="AI models", title="Model status"), model_box)
        if not _models_ready(service, session.use_camera):
            if st.button("Load models", type="primary"):
                model_box.empty()
                with st.status("Loading models — the first run can take a few minutes…",
                               expanded=True) as box:
                    for name in names:
                        if service.registry.is_loaded(name):
                            continue
                        info = service.registry.info(name)
                        box.write(f"Loading {info.label}…")
                        start = time.perf_counter()
                        service.load_models([name])
                        info = service.registry.info(name)
                        box.write(("✅ " if info.status is DeviceStatus.READY else "❌ ")
                                  + f"{info.label} ({time.perf_counter() - start:.1f} s)")
                    box.update(label="Model loading finished", state="complete")
                st.rerun()

        devices = ["Microphone"] + (["Camera"] if session.use_camera else [])
        rows = ""
        for device in devices:
            status, message = ss.readiness.get(device, (DeviceStatus.NOT_CHECKED, ""))
            rows += ("<div class='hn-row'><div class='hn-row-head'><span>" + escape(device) + "</span>"
                     + _status_chip(status) + "</div>"
                     + (f"<div class='hn-row-note'>{escape(message)}</div>" if message else "")
                     + "</div>")
        show(card(rows, label="Devices", title="Camera & microphone"))
        if st.button("Check devices"):
            with st.spinner("Checking devices — nothing is recorded…"):
                ok, msg = check_microphone()
                ss.readiness["Microphone"] = (DeviceStatus.READY if ok else DeviceStatus.UNAVAILABLE, msg)
                if session.use_camera:
                    ok, msg, frame = probe_camera(OpenCVCamera(settings.camera_index))
                    ss.readiness["Camera"] = (DeviceStatus.READY if ok else DeviceStatus.UNAVAILABLE, msg)
                    ss.preview = prepare_display_frame(frame, None) if frame is not None else None
            st.rerun()

    with right:
        if session.use_camera and ss.preview is not None:
            st.image(ss.preview, channels="RGB", caption="Camera preview (one frame, not saved)",
                     width="stretch")
        else:
            show(stage_placeholder("Camera preview", "Press 'Check devices' to see a single preview "
                                   "frame." if session.use_camera else "Audio-only mode."))
        show(card(bullet_list([
            "Sit facing the camera with your face well lit.",
            "Use a quiet room; speak at a normal volume.",
            "Close other apps that use the camera (Teams, Zoom).",
        ]), label="Before you start"))

    devices_ready = all(ss.readiness.get(d, (None,))[0] is DeviceStatus.READY for d in devices)
    models_ready = _models_ready(service, session.use_camera)
    if session.use_camera and ss.readiness.get("Camera", (None,))[0] is DeviceStatus.UNAVAILABLE:
        st.warning("The camera is not available. Fix it and check again, or continue with audio "
                   "only (camera-based scores will not be calculated).")
        if st.button("Continue with audio only"):
            session.use_camera = False
            ss.readiness.pop("Camera", None)
            ss.preview = None
            st.rerun()

    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Continue", type="primary", disabled=not (devices_ready and models_ready),
                 width="stretch"):
        go(session.readiness_confirmed)
    if c2.button("Back to setup", width="stretch"):
        go(session.restart)
    if not (devices_ready and models_ready):
        st.caption("Continue unlocks when every model is loaded and every device is ready.")


# --------------------------------------------------
# Consent
# --------------------------------------------------

def page_consent() -> None:
    header()
    show(stepper(STEPS, 2))
    devices = "camera and microphone" if session.use_camera else "microphone"
    items = [
        f"Your {devices} record only while the magenta RECORDING indicator is on.",
        f"A {settings.countdown_s}-second countdown runs first; nothing is recorded during it.",
        "Everything is processed on this computer and nothing is uploaded.",
        "Audio and video stay in memory and are discarded after analysis"
        + (" — except that INTERVIEW_AI_KEEP_RECORDINGS is on, so audio is saved to the "
           "recordings folder." if settings.keep_recordings else "."),
        "Results stay in this browser session unless you save them to history.",
        "Facial-expression estimates are uncertain, culturally variable and never scored.",
        "Results are coaching feedback, not a hiring assessment.",
    ]
    show(card(bullet_list(items), label="Consent", title="Before you start"))
    agreed = st.checkbox("I understand and agree to be recorded for this practice session.")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Start interview", type="primary", disabled=not agreed, width="stretch"):
        go(session.give_consent)
    if c2.button("Back to setup", width="stretch"):
        go(session.restart)


# --------------------------------------------------
# Live interview console
# --------------------------------------------------

TIMELINE = [(RunState.INITIALIZING, "Connect"), (RunState.COUNTDOWN, "Countdown"),
            (RunState.RECORDING, "Recording"), (RunState.PROCESSING, "Processing"),
            (RunState.COMPLETED, "Results")]


def _state_chips(state: RunState, use_camera: bool) -> str:
    cam = use_camera and state in (RunState.INITIALIZING, RunState.COUNTDOWN, RunState.RECORDING)
    chips = [chip("Camera live", "live", dot=True) if cam
             else chip("Camera off" if use_camera else "Audio only", "na")]
    chips.append({
        RunState.INITIALIZING: chip("Connecting devices", "live", dot=True),
        RunState.COUNTDOWN: chip("Get ready · not recording", "warn"),
        RunState.RECORDING: chip("Recording", "rec", dot=True),
        RunState.PROCESSING: chip("Processing · devices off", "live", dot=True),
        RunState.COMPLETED: chip("Results ready", "ok"),
        RunState.FAILED: chip("Failed", "err"),
        RunState.UNAVAILABLE: chip("Device unavailable", "err"),
        RunState.CANCELLED: chip("Cancelled", "warn"),
    }.get(state, chip("Ready", "ok")))
    return "<div class='hn-legend'>" + "".join(chips) + "</div>"


def _timeline(state: RunState) -> str:
    order = [s for s, _ in TIMELINE]
    current = order.index(state) if state in order else -1
    return stepper([label for _, label in TIMELINE], current, compact=True)


def _signals_html(snap: LiveSnapshot) -> str:
    analyses = snap.recent
    if not analyses:
        return card(para("Waiting for the first analysed frame…"), label="Live · last ~10 s")
    faces = [a for a in analyses if a.face_detected]
    face_pct = 100 * len(faces) / len(analyses)
    eye = 100 * sum(a.eye == LOOKING_CENTER for a in faces) / len(faces) if faces else None
    head = 100 * sum(a.head == CENTER for a in faces) / len(faces) if faces else None
    body = (bar("Face in view", face_pct, style="blue")
            + bar("Looking at the camera", eye, style="blue")
            + bar("Facing the camera", head, style="blue")
            + f"<div class='hn-row-note'>Based on {len(analyses)} analysed frames "
              "(measured, ~3 per second).</div>")
    return card(body, label="Live · behavioral signals · last ~10 s")


def _emotion_html(snap: LiveSnapshot) -> str:
    smoothed = smooth_emotion(snap.recent, now=time.perf_counter())
    raw = latest_raw(snap.recent)
    head = "<div class='hn-legend'>" + chip("Model estimate", "model") + chip("Not scored", "na") + "</div>"
    if smoothed is None:
        body = para("Collecting predictions (the expression model runs about once per second)…")
    else:
        body = (f"<div class='hn-big'>{escape(smoothed.label)}</div>"
                + para(f"Average model probability {smoothed.probability:.0%} over "
                       f"{smoothed.samples} predictions in the last {smoothed.window_s:.0f} s."))
    if raw is not None and raw.emotion:
        body += para(f"Latest raw prediction: {raw.emotion} ({raw.emotion_confidence:.0%}).")
    body += ("<div class='hn-row-note'>Uncertain model output (~51% test accuracy); not a measure "
             "of personality, honesty or ability.</div>")
    return card(head + body, label="Facial expression")


def _live_view(handle: RunHandle, use_camera: bool, boxes: dict) -> None:
    """Poll the worker until it finishes, updating only what changed."""
    interval = 1.0 / max(1.0, settings.display_fps)
    duration = float(session.duration_s)
    last_header = last_frame_id = last_analysis_id = None
    while True:
        snap = handle.status.snapshot()
        whole = int(snap.seconds_left) + 1 if snap.seconds_left is not None else None
        hdr = (snap.state, snap.message, whole)
        if hdr != last_header:
            last_header = hdr
            show(_state_chips(snap.state, use_camera), boxes["chips"])
            show(_timeline(snap.state), boxes["timeline"])
            if snap.state is RunState.COUNTDOWN and whole is not None:
                show(f"<div class='hn-big'>Recording starts in {whole}…</div>", boxes["timer"])
                boxes["progress"].empty()
            elif snap.state is RunState.RECORDING and snap.seconds_left is not None:
                show(f"<div class='hn-big'>{whole}s left</div>", boxes["timer"])
                boxes["progress"].progress(min(1.0, max(0.0, 1 - snap.seconds_left / duration)))
            elif snap.state is RunState.PROCESSING:
                show(card(para(snap.message), label="Processing"), boxes["timer"])
                boxes["progress"].empty()
            else:
                show(card(para(snap.message), label="Status"), boxes["timer"])
        if use_camera and boxes["video"] is not None and snap.frame is not None \
                and id(snap.frame) != last_frame_id:
            last_frame_id = id(snap.frame)
            bbox = snap.analysis.bbox if snap.analysis is not None else None
            boxes["video"].image(prepare_display_frame(snap.frame, bbox), channels="RGB",
                                 output_format="JPEG", width="stretch")
        if use_camera and snap.state is RunState.RECORDING and id(snap.analysis) != last_analysis_id:
            last_analysis_id = id(snap.analysis)
            level, text = guidance(snap.recent)
            kind = {OK: "ok", WARNING: "warn"}.get(level, "na")
            show(card("<div class='hn-legend'>" + chip("Live hint", kind) + "</div>" + para(text),
                      label="Coaching"), boxes["guidance"])
            show(_signals_html(snap), boxes["signals"])
            show(_emotion_html(snap), boxes["emotion"])
        if not handle.thread.is_alive():
            break
        time.sleep(interval)


def page_question(service: InterviewService) -> None:
    qid = session.current_question_id
    question = get_question_by_id(qid) if qid else None
    header(chip(session.progress_text, "live"))
    if question is None:
        st.error("This question is no longer in the question bank.")
        if st.button("Back to setup"):
            go(session.restart)
        return
    view = question.candidate_view()  # never includes concepts or reference answers
    show(stepper(STEPS, 3))

    handle: Optional[RunHandle] = ss.run
    active = handle is not None and handle.question_id == qid

    main, side = st.columns([3, 2], gap="large")
    with main:
        show(card(f"<h3 style='margin:.2rem 0 .4rem'>{escape(view.text)}</h3>"
                  + "<div class='hn-legend'>" + chip(view.difficulty, "measured")
                  + chip(f"{session.duration_s} s answer", "na")
                  + chip("Audio only" if not session.use_camera else "Camera + mic", "na") + "</div>",
                  label=f"{session.progress_text} · {view.role}", glow=True))
        controls = st.container()
        boxes = {"chips": st.empty(), "timer": st.empty(), "progress": st.empty(),
                 "video": st.empty() if session.use_camera else None}
    with side:
        boxes.update(timeline=st.empty(), guidance=st.empty(), signals=st.empty(),
                     emotion=st.empty())

    if active:
        # A run exists for this question: keep showing it (also after a rerun
        # caused by a click), then record its outcome exactly once.
        if handle.thread.is_alive():
            with controls:
                if st.button("Stop recording", help="Cancels this answer. Nothing is saved."):
                    handle.cancel.set()
            _live_view(handle, session.use_camera, boxes)
        handle.thread.join()
        outcome = handle.outcome or RunOutcome(RunState.FAILED, "The recording did not finish.")
        ss.run = None
        go(lambda: session.record_outcome(outcome))
        return

    show(_state_chips(RunState.IDLE, session.use_camera), boxes["chips"])
    show(_timeline(RunState.IDLE), boxes["timeline"])
    if boxes["video"] is not None:
        show(stage_placeholder("Camera standby", "The camera turns on when you press Start "
                               "recording; a countdown runs before anything is recorded."),
             boxes["video"])
    show(card(para("Live insights appear once recording starts: behavioral signals (measured), "
                   "coaching hints and facial-expression estimates (model output, not scored)."),
              label="Insights"), boxes["guidance"])
    with controls:
        c1, c2, _ = st.columns([1, 1, 1])
        if c1.button("Start recording", type="primary", width="stretch"):
            ss.run = start_run(service, question, session.duration_s, session.use_camera)
            st.rerun()
        if c2.button("End interview", width="stretch"):
            go(session.restart)
        st.caption("Pause/resume is not supported: each answer is one continuous recording. "
                   "Use Stop recording to cancel and Retry on the next screen.")


# --------------------------------------------------
# Review and summary
# --------------------------------------------------

def page_review() -> None:
    qid = session.current_question_id
    outcome: Optional[RunOutcome] = session.outcomes.get(qid) if qid else None
    question = get_question_by_id(qid) if qid else None
    header(chip(session.progress_text, "live"))
    show(stepper(STEPS, 4))
    st.markdown(f"### {question.text if question else 'Question'}")

    if outcome is None:
        st.error("No result for this question.")
    elif outcome.state is not RunState.COMPLETED or outcome.result is None:
        kind = "warn" if outcome.state is RunState.CANCELLED else "err"
        show(card("<div class='hn-legend'>" + chip(outcome.state.value, kind) + "</div>"
                  + para(outcome.message), label="This answer was not assessed"))
        if outcome.device == "camera":
            st.write("Fix the camera and retry, or continue in audio-only mode.")
            if st.button("Retry in audio-only mode"):
                session.use_camera = False
                go(session.retry_question)
    else:
        render_result(outcome.result)

    c1, c2, c3, _ = st.columns([1, 1, 1, 2])
    if c1.button("Retry this question", width="stretch"):
        go(session.retry_question)
    if c2.button("Finish" if session.is_last_question else "Next question", type="primary",
                 width="stretch"):
        go(session.next_question)
    if not session.is_last_question and c3.button("Finish now", width="stretch"):
        go(session.finish)


def _export_buttons(record: SessionRecord, key: str) -> None:
    c1, c2 = st.columns(2)
    c1.download_button("Download report (HTML → print to PDF)", to_html(record),
                       file_name=f"human-next-{record.session_id}.html", mime="text/html",
                       key=f"html-{key}", width="stretch")
    c2.download_button("Download data (CSV)", to_csv(record),
                       file_name=f"human-next-{record.session_id}.csv", mime="text/csv",
                       key=f"csv-{key}", width="stretch")


def page_summary() -> None:
    header(chip("Interview complete", "ok"))
    st.markdown("## Interview results")
    if session.candidate:
        st.caption(f"{session.candidate} · {session.role} · {session.created_at}")
    render_dashboard(record_for(include_transcripts=True))

    st.markdown("#### Save & export")
    include = st.checkbox("Include transcripts in the saved history and downloads", value=False,
                          help="Off by default: only scores, explanations and feedback are kept.")
    export_record = record_for(include_transcripts=include)
    _export_buttons(export_record, "summary")
    c1, c2, c3, _ = st.columns([1.2, 1, 1, 1.5])
    if c1.button("Save to history", disabled=session.saved_to_history, width="stretch"):
        try:
            history.save(export_record)
            session.saved_to_history = True
            st.success("Saved. Only scores, explanations and feedback were stored"
                       + (" (with transcripts)." if include else "."))
        except HistoryError as exc:
            st.warning(str(exc))
    if c2.button("New interview", type="primary", width="stretch"):
        go(session.restart)
    if c3.button("History", width="stretch"):
        go(session.open_history)


def page_history() -> None:
    header(chip("History", "measured"))
    st.markdown("## Interview history")
    st.caption(f"Stored on this computer in {settings.history_path.name}: scores, explanations and "
               "feedback — plus transcripts only if you chose to include them. Never audio or video.")
    try:
        sessions = history.list()
    except HistoryError as exc:
        st.warning(str(exc))
        sessions = []
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Back", width="stretch"):
        go(session.back_to_welcome)
    if c2.button("New interview", type="primary", width="stretch"):
        go(session.restart)
    if not sessions:
        show(card(para("No saved sessions yet. Finish an interview and press 'Save to history'."),
                  label="Empty"))
        return

    labels = {s.session_id: (f"{s.created_at} · {s.role}"
                             + (f" · {s.candidate}" if s.candidate else "")
                             + (f" · avg {round_half_up(s.average)}" if s.average is not None
                                else " · unscored"))
              for s in sessions}
    chosen = st.selectbox("Saved sessions", list(labels), format_func=labels.get)
    try:
        record = history.get(chosen)
    except HistoryError as exc:
        st.warning(str(exc))
        record = None
    if record is None:
        st.info("This session is no longer available.")
    else:
        render_dashboard(record)
        _export_buttons(record, f"hist-{record.session_id}")
        if st.button("Delete this session"):
            try:
                history.delete(chosen)
            except HistoryError as exc:
                st.warning(str(exc))
            st.rerun()
    with st.expander("Clear all history"):
        confirm = st.checkbox("Yes, permanently delete every saved session")
        if st.button("Clear history", disabled=not confirm):
            try:
                n = history.clear()
                st.success(f"Deleted {n} session(s).")
            except HistoryError as exc:
                st.warning(str(exc))
            st.rerun()


# --------------------------------------------------
# Router
# --------------------------------------------------

service = get_service()
stage = session.stage
if stage is Stage.WELCOME:
    page_welcome()
elif stage is Stage.HISTORY:
    page_history()
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
