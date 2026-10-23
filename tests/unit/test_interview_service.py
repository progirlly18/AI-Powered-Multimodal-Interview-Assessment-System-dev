"""End-to-end recording runs with fake devices and models."""

import threading
import time

import pytest

from backend.ai.answer_evaluator import AnswerEvaluator
from backend.audio.speech_analyzer import SpeechAnalyzer
from backend.core.models import ModelLoadError, ModelRegistry, ModelSpec
from backend.core.results import Status
from backend.core.states import RunState
from backend.services import interview_service as svc
from backend.services.interview_service import InterviewService, LiveStatus
from tests.conftest import (
    YAMNET_LABELS,
    FakeCamera,
    FakeEncoder,
    FakeFaceAnalyzer,
    FakeMicrophone,
    FakeWhisper,
    FakeYamnet,
)

DURATION = 1.3
ANSWER = ("Overfitting means the model does not generalize to new data. "
          "Use regularization and cross validation.")


@pytest.fixture(autouse=True)
def short_runs(monkeypatch):
    monkeypatch.setattr(svc, "MIN_DURATION_S", 0.5)


def make_service(settings, *, camera=None, mic=None, analyzer=None, whisper=None,
                 encoder=None, yamnet=None, registry=None):
    camera = camera or FakeCamera()
    mic = mic or FakeMicrophone()
    whisper = whisper or FakeWhisper(ANSWER, segments=[{"start": 0.0, "end": 1.2}])
    service = InterviewService(
        settings=settings,
        registry=registry or ModelRegistry({}),
        camera_factory=lambda: camera,
        microphone_factory=lambda: mic,
        face_analyzer=analyzer or FakeFaceAnalyzer(),
        speech_analyzer=SpeechAnalyzer(lambda: whisper,
                                       lambda: (yamnet or FakeYamnet(0), YAMNET_LABELS)),
        evaluator=AnswerEvaluator(lambda: encoder or FakeEncoder()),
    )
    return service, camera, mic


def assert_released(camera, mic):
    assert camera.release_count <= 1 and not camera.opened
    assert mic.closed >= 1 and not mic.opened


def test_complete_run(fast_settings, overfitting_question):
    # Needs > 3 s of speech, otherwise speaking pace (and delivery) is
    # correctly reported as unavailable.
    whisper = FakeWhisper(ANSWER, segments=[{"start": 0.0, "end": 3.4}])
    service, camera, mic = make_service(fast_settings, whisper=whisper)
    status = LiveStatus()
    outcome = service.run_question(overfitting_question, 3.5, status=status)

    assert outcome.state is RunState.COMPLETED, outcome.message
    r = outcome.result
    assert r.question_id == "test-01" and r.question_text == overfitting_question.text
    assert r.is_complete
    assert r.speech.transcript == ANSWER
    assert r.answer.coverage.value == 100
    assert r.vision.eye_contact.value == 100
    assert r.score.overall.is_available and not r.score.is_partial
    assert r.av_start_offset_ms is not None and abs(r.av_start_offset_ms) < 500
    for key in ("camera_open_s", "microphone_open_s", "recording_s", "speech_analysis_s",
                "answer_evaluation_s", "processing_s", "total_s", "transcription_s"):
        assert key in r.timings
    assert r.timings["recording_s"] == pytest.approx(3.5, abs=0.3)
    assert r.speech.delivery_score.is_available
    assert status.snapshot().state is RunState.COMPLETED
    assert_released(camera, mic)


def test_states_are_visited_in_order(fast_settings, overfitting_question):
    service, _, _ = make_service(fast_settings)
    seen = []
    status = LiveStatus()
    original = status.update

    def spy(**kw):
        if kw.get("state") is not None:
            seen.append(kw["state"])
        original(**kw)

    status.update = spy
    service.run_question(overfitting_question, DURATION, status=status)
    assert seen == [RunState.INITIALIZING, RunState.COUNTDOWN, RunState.RECORDING,
                    RunState.PROCESSING, RunState.COMPLETED]


def test_countdown_happens_before_recording(fast_settings, overfitting_question):
    from dataclasses import replace

    settings = replace(fast_settings, countdown_s=1)
    service, _, mic = make_service(settings)
    t0 = time.perf_counter()
    outcome = service.run_question(overfitting_question, DURATION)
    assert mic.first_sample_time - t0 >= 0.95       # capture began after the countdown
    assert outcome.result.speech.recording_seconds == pytest.approx(DURATION, abs=0.3)


def test_camera_unavailable_reports_and_fabricates_nothing(fast_settings, overfitting_question):
    service, camera, mic = make_service(fast_settings, camera=FakeCamera(fail_open=True))
    outcome = service.run_question(overfitting_question, DURATION)
    assert outcome.state is RunState.UNAVAILABLE and outcome.device == "camera"
    assert outcome.result is None
    assert "camera" in outcome.message.lower()
    assert not mic.opened  # the microphone was never opened


def test_microphone_unavailable(fast_settings, overfitting_question):
    service, camera, mic = make_service(fast_settings, mic=FakeMicrophone(fail_open=True))
    outcome = service.run_question(overfitting_question, DURATION)
    assert outcome.state is RunState.UNAVAILABLE and outcome.device == "microphone"
    assert camera.release_count == 1  # camera was released


def test_microphone_failure_during_recording(fast_settings, overfitting_question):
    service, camera, mic = make_service(fast_settings, mic=FakeMicrophone(fail_during=True))
    outcome = service.run_question(overfitting_question, DURATION)
    assert outcome.state is RunState.FAILED and outcome.device == "microphone"
    assert_released(camera, mic)


def test_camera_disconnect_during_recording(fast_settings, overfitting_question):
    camera = FakeCamera(stop_after=40, frame_delay=0.002)
    service, _, mic = make_service(fast_settings, camera=camera)
    outcome = service.run_question(overfitting_question, 3.0)
    assert outcome.state is RunState.FAILED and outcome.device == "camera"
    assert_released(camera, mic)


def test_audio_only_mode_gives_labelled_partial_score(fast_settings, overfitting_question):
    service, camera, mic = make_service(fast_settings)
    outcome = service.run_question(overfitting_question, DURATION, use_camera=False)
    r = outcome.result
    assert outcome.state is RunState.COMPLETED and camera.reads == 0
    assert r.vision.status is Status.UNAVAILABLE and "audio-only" in r.vision.reason
    assert r.score.is_partial and r.score.overall.is_available
    assert not r.is_complete


def test_partial_score_is_not_reported_complete(fast_settings, overfitting_question):
    # 1.3 s of speech -> pace unavailable -> delivery missing -> partial.
    service, _, _ = make_service(fast_settings)
    r = service.run_question(overfitting_question, DURATION).result
    assert r.speech.status is Status.AVAILABLE
    assert not r.speech.wpm.is_available
    assert r.score.is_partial and not r.is_complete


def test_transcription_failure_withholds_score(fast_settings, overfitting_question):
    service, _, _ = make_service(fast_settings, whisper=FakeWhisper(fail=True))
    r = service.run_question(overfitting_question, DURATION).result
    assert r.speech.status is Status.FAILED
    assert r.answer.status is Status.FAILED
    assert not r.score.overall.is_available        # no default score
    assert r.answer.feedback is None


def test_silent_microphone_withholds_score(fast_settings, overfitting_question):
    service, _, _ = make_service(fast_settings, mic=FakeMicrophone(silent=True),
                                 whisper=FakeWhisper(""))
    r = service.run_question(overfitting_question, DURATION).result
    assert r.speech.status is Status.UNAVAILABLE and "muted" in r.speech.reason
    assert not r.score.overall.is_available


def test_evaluation_failure_is_not_replaced(fast_settings, overfitting_question):
    service, _, _ = make_service(fast_settings, encoder=FakeEncoder(fail=True))
    r = service.run_question(overfitting_question, DURATION).result
    assert r.answer.status is Status.FAILED
    assert r.answer.answer_score.value is None
    assert "evaluated successfully" not in " ".join(r.suggestions)
    assert not r.score.overall.is_available


def test_no_hardware_cannot_score_100(fast_settings, overfitting_question):
    """Regression for the teammate branch, where this scored 100/100."""
    service, _, _ = make_service(fast_settings, mic=FakeMicrophone(silent=True),
                                 whisper=FakeWhisper(""), analyzer=FakeFaceAnalyzer(
                                     [dict(face_detected=False)]))
    r = service.run_question(overfitting_question, DURATION).result
    assert r.vision.status is Status.UNAVAILABLE
    assert not r.score.overall.is_available


@pytest.mark.parametrize("cancel_after,expected_reads", [(0.05, "countdown"), (0.6, "recording")])
def test_cancel_releases_devices(fast_settings, overfitting_question, cancel_after, expected_reads):
    from dataclasses import replace

    settings = replace(fast_settings, countdown_s=0 if expected_reads == "recording" else 2)
    service, camera, mic = make_service(settings)
    cancel = threading.Event()
    threading.Timer(cancel_after, cancel.set).start()
    outcome = service.run_question(overfitting_question, 3.0, cancel=cancel)
    assert outcome.state is RunState.CANCELLED and outcome.result is None
    assert "Nothing was saved" in outcome.message
    assert_released(camera, mic)


def test_only_one_run_at_a_time(fast_settings, overfitting_question):
    service, _, _ = make_service(fast_settings)
    results = {}

    def first():
        results["first"] = service.run_question(overfitting_question, 1.5)

    t = threading.Thread(target=first)
    t.start()
    time.sleep(0.3)
    second, _, _ = make_service(fast_settings)
    results["second"] = second.run_question(overfitting_question, 1.0)
    t.join()
    assert results["first"].state is RunState.COMPLETED
    assert results["second"].state is RunState.FAILED
    assert "already running" in results["second"].message


def test_required_face_model_failure_fails_before_recording(fast_settings, overfitting_question):
    def broken():
        raise OSError("missing file")
    registry = ModelRegistry({"face_mesh": ModelSpec("Face landmark detector", broken, "Fix it."),
                              "emotion": ModelSpec("Emotion", lambda: object(), "")})
    camera, mic = FakeCamera(), FakeMicrophone()
    service = InterviewService(settings=fast_settings, registry=registry,
                               camera_factory=lambda: camera, microphone_factory=lambda: mic,
                               speech_analyzer=SpeechAnalyzer(lambda: FakeWhisper("x")),
                               evaluator=AnswerEvaluator(lambda: FakeEncoder()))
    outcome = service.run_question(overfitting_question, DURATION)
    assert outcome.state is RunState.FAILED and "Face landmark detector" in outcome.message
    assert camera.reads == 0 and not mic.opened


def test_emotion_model_failure_only_disables_expressions(fast_settings, overfitting_question):
    from backend.vision.landmarks import FULL_LANDMARK_COUNT  # noqa: F401 (import check)

    class Landmarker:
        def detect(self, frame):
            from tests.unit.test_vision import make_face
            return make_face()

    def broken():
        raise OSError("bad model")

    registry = ModelRegistry({"face_mesh": ModelSpec("Faces", lambda: Landmarker(), ""),
                              "emotion": ModelSpec("Expressions", broken, "")})
    camera, mic = FakeCamera(shape=(300, 300, 3)), FakeMicrophone()
    service = InterviewService(
        settings=fast_settings, registry=registry,
        camera_factory=lambda: camera, microphone_factory=lambda: mic,
        speech_analyzer=SpeechAnalyzer(lambda: FakeWhisper(ANSWER, [{"start": 0, "end": 1.2}])),
        evaluator=AnswerEvaluator(lambda: FakeEncoder()))
    outcome = service.run_question(overfitting_question, DURATION)
    assert outcome.state is RunState.COMPLETED
    r = outcome.result
    assert r.vision.status is Status.AVAILABLE and r.vision.dominant_emotion is None
    assert any("expression estimates are off" in w for w in r.warnings)


def test_invalid_duration_rejected(fast_settings, overfitting_question):
    service, _, _ = make_service(fast_settings)
    with pytest.raises(ValueError):
        service.run_question(overfitting_question, 10_000)


def test_recordings_not_written_by_default(fast_settings, overfitting_question):
    service, _, _ = make_service(fast_settings)
    service.run_question(overfitting_question, DURATION)
    assert not fast_settings.recordings_dir.exists()


def test_keep_recordings_writes_unique_files(fast_settings, overfitting_question):
    from dataclasses import replace

    settings = replace(fast_settings, keep_recordings=True)
    for _ in range(2):
        service, _, _ = make_service(settings)
        r = service.run_question(overfitting_question, DURATION).result
        assert any("saved to" in w for w in r.warnings)
    files = list(settings.recordings_dir.glob("answer-*.wav"))
    assert len(files) == 2


def test_live_status_exposes_frames_only_while_devices_open(fast_settings, overfitting_question):
    service, _, _ = make_service(fast_settings)
    status = LiveStatus()
    frames_during = []

    def watch():
        while status.snapshot().state is not RunState.COMPLETED:
            snap = status.snapshot()
            if snap.state is RunState.RECORDING and snap.frame is not None:
                frames_during.append(snap.frame.shape)
            time.sleep(0.02)

    t = threading.Thread(target=watch)
    t.start()
    service.run_question(overfitting_question, DURATION, status=status)
    t.join(5)
    assert frames_during, "expected live frames during recording"
    assert status.snapshot().frame is None  # released after the run
