"""
Orchestrates one recorded answer:

    INITIALIZING -> COUNTDOWN -> RECORDING -> PROCESSING -> COMPLETED
                 \-> UNAVAILABLE (device)   \-> FAILED / CANCELLED

* Camera and microphone are opened and verified before the countdown.
* Audio samples and analysed video frames are only kept from the moment
  recording starts; both start from the same signal and the measured
  start offset is reported.
* Devices are released before processing starts and on every error path.
* Worker threads never call the UI. They update a thread-safe
  ``LiveStatus`` that the UI polls.
* Missing or failed measurements are reported as such; nothing is
  substituted.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from backend.ai.answer_evaluator import AnswerEvaluator
from backend.ai.question_bank import Question
from backend.audio.recorder import (
    SAMPLE_RATE,
    AudioSource,
    DeviceUnavailableError,
    MicrophoneRecorder,
)
from backend.audio.speech_analyzer import SpeechAnalyzer
from backend.config import Settings, load_settings
from backend.core.models import ModelInfo, ModelLoadError, ModelRegistry, get_registry
from backend.core.results import (
    AnswerResult,
    Metric,
    QuestionResult,
    SpeechResult,
    Status,
    VisionResult,
)
from backend.core.states import RunState, new_run_state_machine
from backend.scoring.scoring_engine import build_suggestions, score_question
from backend.vision.capture import CameraSource, OpenCVCamera, VisionSession
from backend.vision.face_analysis import FaceAnalyzer, FrameAnalysis

log = logging.getLogger(__name__)

# Only one recording at a time per process (there is one camera and microphone).
_RUN_LOCK = threading.Lock()

MIN_DURATION_S = 5
MAX_DURATION_S = 300
# Analyses kept in live snapshots: ~10 s at the default 0.33 s interval.
RECENT_WINDOW = 30


class CancelledError(Exception):
    """The user cancelled the run."""


# --------------------------------------------------
# Live status shared with the UI
# --------------------------------------------------

@dataclass
class LiveSnapshot:
    state: RunState
    message: str
    seconds_left: Optional[float]
    frame: Optional[np.ndarray]
    analysis: Optional[FrameAnalysis]
    stage: Optional[str]
    recent: list[FrameAnalysis] = field(default_factory=list)


class LiveStatus:
    """Thread-safe progress information for the UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = RunState.IDLE
        self._message = "Waiting to start."
        self._seconds_left: Optional[float] = None
        self._stage: Optional[str] = None
        self._vision: Optional[VisionSession] = None

    def update(self, *, state: Optional[RunState] = None, message: Optional[str] = None,
               seconds_left: Optional[float] = None, stage: Optional[str] = None,
               clear_seconds: bool = False) -> None:
        with self._lock:
            if state is not None:
                self._state = state
            if message is not None:
                self._message = message
            if seconds_left is not None or clear_seconds:
                self._seconds_left = seconds_left
            if stage is not None:
                self._stage = stage

    def attach_vision(self, vision: Optional[VisionSession]) -> None:
        with self._lock:
            self._vision = vision

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            vision = self._vision
            state, message = self._state, self._message
            seconds, stage = self._seconds_left, self._stage
        frame, analysis, recent = None, None, []
        if vision is not None and not vision.released:
            frame, analysis = vision.snapshot()
            recent = vision.recent_analyses(RECENT_WINDOW)
        return LiveSnapshot(state, message, seconds, frame, analysis, stage, recent)


@dataclass
class RunOutcome:
    state: RunState
    message: str
    result: Optional[QuestionResult] = None
    device: Optional[str] = None
    timings: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------
# Service
# --------------------------------------------------

class InterviewService:
    """Runs recordings. All collaborators can be replaced in tests."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        registry: Optional[ModelRegistry] = None,
        camera_factory: Optional[Callable[[], CameraSource]] = None,
        microphone_factory: Optional[Callable[[], AudioSource]] = None,
        face_analyzer: Optional[FaceAnalyzer] = None,
        speech_analyzer: Optional[SpeechAnalyzer] = None,
        evaluator: Optional[AnswerEvaluator] = None,
        # perf_counter: time.monotonic only ticks every ~16 ms on Windows.
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or load_settings()
        self.registry = registry or get_registry()
        reg = self.registry
        self._camera_factory = camera_factory or (lambda: OpenCVCamera(self.settings.camera_index))
        self._microphone_factory = microphone_factory or (lambda: MicrophoneRecorder())
        self._face_analyzer = face_analyzer
        self._speech_analyzer = speech_analyzer or SpeechAnalyzer(
            whisper_provider=lambda: reg.get("whisper"),
            yamnet_provider=lambda: reg.get("yamnet"),
            language=self.settings.language,
        )
        self._evaluator = evaluator or AnswerEvaluator(lambda: reg.get("sentence"))
        self._clock = clock
        self._sleep = sleep

    # ---------------- models ----------------
    @staticmethod
    def required_models(use_camera: bool) -> list[str]:
        names = ["whisper", "sentence", "yamnet"]
        return (["face_mesh", "emotion"] if use_camera else []) + names

    def load_models(self, names: list[str],
                    on_progress: Optional[Callable[[ModelInfo], None]] = None) -> dict[str, ModelInfo]:
        """Load models one by one; failures are recorded in the returned info."""
        infos: dict[str, ModelInfo] = {}
        for name in names:
            info = self.registry.info(name)
            if on_progress is not None:
                on_progress(info)
            try:
                self.registry.get(name)
            except ModelLoadError:
                pass
            infos[name] = self.registry.info(name)
            if on_progress is not None:
                on_progress(infos[name])
        return infos

    def _make_face_analyzer(self, warnings: list[str]) -> FaceAnalyzer:
        if self._face_analyzer is not None:
            return self._face_analyzer
        self.registry.get("face_mesh")  # required: raises ModelLoadError
        emotion_provider: Optional[Callable] = None
        try:
            self.registry.get("emotion")
            emotion_provider = lambda: self.registry.get("emotion")  # noqa: E731
        except ModelLoadError as exc:
            warnings.append(f"Facial expression estimates are off: {exc}")
        return FaceAnalyzer(lambda: self.registry.get("face_mesh"), emotion_provider,
                            emotion_interval_s=self.settings.emotion_interval_s)

    # ---------------- main entry point ----------------
    def run_question(
        self,
        question: Question,
        duration_s: float,
        *,
        use_camera: bool = True,
        status: Optional[LiveStatus] = None,
        cancel: Optional[threading.Event] = None,
    ) -> RunOutcome:
        status = status or LiveStatus()
        cancel = cancel or threading.Event()
        if not MIN_DURATION_S <= duration_s <= MAX_DURATION_S:
            raise ValueError(f"duration_s must be between {MIN_DURATION_S} and {MAX_DURATION_S}")
        if not _RUN_LOCK.acquire(blocking=False):
            message = "Another recording is already running on this computer."
            status.update(state=RunState.FAILED, message=message)
            return RunOutcome(RunState.FAILED, message)
        try:
            return self._run_locked(question, float(duration_s), use_camera, status, cancel)
        finally:
            _RUN_LOCK.release()

    def _run_locked(self, question: Question, duration_s: float, use_camera: bool,
                    status: LiveStatus, cancel: threading.Event) -> RunOutcome:
        sm = new_run_state_machine(lambda _old, new: status.update(state=new))
        timings: dict[str, float] = {}
        warnings: list[str] = []
        vision: Optional[VisionSession] = None
        mic: Optional[AudioSource] = None
        t_start = self._clock()

        def check_cancel() -> None:
            if cancel.is_set():
                raise CancelledError()

        try:
            sm.transition(RunState.INITIALIZING)

            # -- devices (verified before anything is recorded) --
            if use_camera:
                status.update(message="Starting the camera…")
                analyzer = self._make_face_analyzer(warnings)
                t = self._clock()
                vision = VisionSession(self._camera_factory(), analyzer,
                                       self.settings.vision_interval_s)
                vision.start()
                timings["camera_open_s"] = self._clock() - t
                status.attach_vision(vision)
            check_cancel()

            status.update(message="Starting the microphone…")
            t = self._clock()
            mic = self._microphone_factory()
            mic.open()
            timings["microphone_open_s"] = self._clock() - t
            check_cancel()

            # -- countdown (devices open, nothing kept) --
            sm.transition(RunState.COUNTDOWN)
            countdown_end = self._clock() + self.settings.countdown_s
            while (left := countdown_end - self._clock()) > 0:
                check_cancel()
                status.update(message="Recording starts in…", seconds_left=left)
                self._sleep(min(0.05, left))

            # -- recording --
            sm.transition(RunState.RECORDING)
            mic.start_capture()
            if vision is not None:
                vision.begin_recording()
            rec_start = self._clock()
            rec_end = rec_start + duration_s
            while (left := rec_end - self._clock()) > 0:
                check_cancel()
                if vision is not None and vision.error:
                    raise DeviceUnavailableError("camera", vision.error)
                if mic.error:
                    raise DeviceUnavailableError("microphone", mic.error)
                status.update(message="Recording — answer the question now.", seconds_left=left)
                self._sleep(min(0.05, left))
            mic.stop_capture()
            if vision is not None:
                vision.end_recording()
            timings["recording_s"] = self._clock() - rec_start

            av_offset_ms: Optional[float] = None
            if vision is not None and vision.first_frame_time and mic.first_sample_time:
                av_offset_ms = (vision.first_frame_time - mic.first_sample_time) * 1000.0
            audio = mic.audio()
            vision_result = self._finish_vision(vision, use_camera, timings)
            vision = None
            mic.close()
            mic = None
            status.attach_vision(None)

            # -- processing (devices already released) --
            sm.transition(RunState.PROCESSING)
            status.update(message="Processing your answer…", clear_seconds=True)
            result = self._process(question, duration_s, audio, vision_result, warnings,
                                   timings, status, av_offset_ms)
            timings["total_s"] = self._clock() - t_start
            result.timings["total_s"] = timings["total_s"]
            sm.transition(RunState.COMPLETED)
            status.update(message="Results are ready.")
            return RunOutcome(RunState.COMPLETED, "Results are ready.", result, timings=timings)

        except CancelledError:
            sm.transition(RunState.CANCELLED)
            message = "Recording cancelled. Nothing was saved."
            status.update(message=message, clear_seconds=True)
            return RunOutcome(RunState.CANCELLED, message, timings=timings)
        except DeviceUnavailableError as exc:
            target = RunState.UNAVAILABLE if sm.can_transition(RunState.UNAVAILABLE) else RunState.FAILED
            sm.transition(target)
            status.update(message=str(exc), clear_seconds=True)
            return RunOutcome(target, str(exc), device=exc.device, timings=timings)
        except ModelLoadError as exc:
            sm.transition(RunState.FAILED)
            status.update(message=str(exc), clear_seconds=True)
            return RunOutcome(RunState.FAILED, str(exc), timings=timings)
        except Exception:  # noqa: BLE001 - logged with traceback, generic message shown
            log.exception("Recording run failed")
            if sm.can_transition(RunState.FAILED):
                sm.transition(RunState.FAILED)
            message = "Something went wrong during the recording. Please try again."
            status.update(message=message, clear_seconds=True)
            return RunOutcome(RunState.FAILED, message, timings=timings)
        finally:
            if vision is not None:
                vision.stop()
            if mic is not None:
                mic.close()
            status.attach_vision(None)

    # ---------------- helpers ----------------
    def _finish_vision(self, vision: Optional[VisionSession], use_camera: bool,
                       timings: dict[str, float]) -> VisionResult:
        if vision is None:
            reason = "Camera not used (audio-only mode)." if not use_camera else "Camera not started."
            return VisionResult(
                status=Status.UNAVAILABLE, reason=reason,
                face_presence=Metric.unavailable("Face in view", reason, "%"),
                eye_contact=Metric.unavailable("Looking at camera", reason, "%"),
                head_facing=Metric.unavailable("Facing camera", reason, "%"),
            )
        vision.stop()
        timings.update(vision.timing_summary_ms())
        return vision.result()

    def _process(self, question: Question, duration_s: float, audio: np.ndarray,
                 vision_result: VisionResult, warnings: list[str], timings: dict[str, float],
                 status: LiveStatus, av_offset_ms: Optional[float]) -> QuestionResult:
        t_proc = self._clock()
        if self.settings.keep_recordings and audio.size:
            self._save_recording(audio, warnings)

        status.update(stage="transcribing", message="Transcribing your answer…")
        t = self._clock()
        speech = self._speech_analyzer.analyze(audio, SAMPLE_RATE)
        timings["speech_analysis_s"] = self._clock() - t
        timings.update(getattr(self._speech_analyzer, "timings", {}))

        status.update(stage="evaluating", message="Evaluating the answer content…")
        t = self._clock()
        answer = self._evaluate(question, speech)
        timings["answer_evaluation_s"] = self._clock() - t

        status.update(stage="scoring", message="Calculating scores…")
        score = score_question(vision_result, speech, answer)
        suggestions = build_suggestions(vision_result, speech, answer)
        if speech.background_note:
            warnings.append(speech.background_note)
        if vision_result.reason and vision_result.status is Status.AVAILABLE:
            warnings.append(vision_result.reason)
        if vision_result.frames_with_face and \
                vision_result.frames_multiple_faces / vision_result.frames_with_face > 0.2:
            warnings.append("More than one face was in view for part of the answer; "
                            "camera measurements used only one face.")
        timings["processing_s"] = self._clock() - t_proc

        return QuestionResult(
            question_id=question.id,
            question_text=question.text,
            duration_seconds=duration_s,
            vision=vision_result,
            speech=speech,
            answer=answer,
            score=score,
            suggestions=suggestions,
            warnings=warnings,
            timings=dict(timings),
            av_start_offset_ms=av_offset_ms,
        )

    def _evaluate(self, question: Question, speech: SpeechResult) -> AnswerResult:
        if speech.status is not Status.AVAILABLE:
            reason = f"No transcript: {speech.reason}"
            metric = Metric.unavailable if speech.status is Status.UNAVAILABLE else Metric.failed
            return AnswerResult(
                status=speech.status,
                reason=reason,
                relevance=metric("Topic relevance", reason),
                coverage=metric("Concept coverage", reason),
                answer_score=metric("Answer content", reason),
            )
        return self._evaluator.evaluate(question, speech.transcript)

    def _save_recording(self, audio: np.ndarray, warnings: list[str]) -> None:
        try:
            from scipy.io.wavfile import write

            folder = self.settings.recordings_dir
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"answer-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.wav"
            write(str(path), SAMPLE_RATE, audio)
            warnings.append(f"A copy of this recording was saved to {path.name} (INTERVIEW_AI_KEEP_RECORDINGS is on).")
        except Exception:  # noqa: BLE001
            log.exception("Saving the recording failed")
            warnings.append("The recording could not be saved (INTERVIEW_AI_KEEP_RECORDINGS is on).")
