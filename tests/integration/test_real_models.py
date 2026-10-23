"""Integration tests with the real models (pytest -m integration).

Models are downloaded on first use. Audio is synthetic (Windows speech
synthesiser); no personal recording is used.
"""

import numpy as np
import pytest

from backend.ai import question_bank as qb
from backend.ai.answer_evaluator import AnswerEvaluator
from backend.audio.speech_analyzer import SpeechAnalyzer
from backend.core.models import get_registry
from backend.core.results import Status
from backend.vision.emotion_detector import EMOTION_LABELS, predict_emotion
from tests.integration.tts import synthesize, word_error_rate

pytestmark = pytest.mark.integration

SPOKEN = ("Overfitting happens when a model learns the training data too closely "
          "and does not generalize to new data. Regularization and cross validation help.")


@pytest.fixture(scope="module")
def registry():
    return get_registry()


@pytest.fixture(scope="module")
def synthetic_speech(tmp_path_factory):
    audio = synthesize(SPOKEN, tmp_path_factory.mktemp("tts") / "speech.wav")
    if audio is None:
        pytest.skip("Windows speech synthesis is not available")
    return audio


def test_sentence_model_ranks_answers(registry):
    q = qb.get_question("Machine Learning Engineer", 2)
    ev = AnswerEvaluator(lambda: registry.get("sentence"))
    good = ev.evaluate(q, SPOKEN)
    partial = ev.evaluate(q, "Overfitting means the model learns the training data very closely.")
    off = ev.evaluate(q, "Python is a programming language used for developing software.")
    assert good.status is partial.status is off.status is Status.AVAILABLE
    assert good.answer_score.value > partial.answer_score.value > off.answer_score.value
    for r in (good, partial, off):
        assert 0 <= r.relevance.value <= 100


def test_whisper_transcribes_synthetic_speech(registry, synthetic_speech):
    analyzer = SpeechAnalyzer(lambda: registry.get("whisper"),
                              lambda: registry.get("yamnet"))
    r = analyzer.analyze(synthetic_speech)
    assert r.status is Status.AVAILABLE, r.reason
    wer = word_error_rate(SPOKEN, r.transcript)
    assert wer < 0.25, f"WER {wer:.2f}: {r.transcript!r}"
    assert r.wpm.is_available and 60 < r.wpm.value < 250
    assert r.speaking_time_is_estimate is False
    assert r.background_label  # YAMNet ran
    assert "transcription_s" in analyzer.timings


def test_yamnet_labels_parse_quoted_names(registry):
    _model, labels = registry.get("yamnet")
    assert len(labels) == 521
    assert "Child speech, kid speaking" in labels  # contains a comma


def test_face_landmarker_on_blank_frame(registry):
    landmarker = registry.get("face_mesh")
    assert landmarker.detect(np.zeros((480, 640, 3), np.uint8)) is None
    assert landmarker.detect_all(np.zeros((480, 640, 3), np.uint8)) == []


def test_rubric_with_real_model_explains_every_criterion(registry):
    q = qb.get_question("Machine Learning Engineer", 2)
    r = AnswerEvaluator(lambda: registry.get("sentence")).evaluate(q, SPOKEN)
    assert [c.key for c in r.criteria] == ["relevance", "coverage", "elaboration",
                                           "structure", "examples"]
    assert all(c.explanation and c.metric.is_available for c in r.criteria)
    assert r.answer_score.value == pytest.approx(sum(c.weight * c.metric.value for c in r.criteria))


def test_emotion_model_output(registry):
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    label, confidence = predict_emotion(registry.get("emotion"), frame, (200, 100, 440, 380))
    assert label in EMOTION_LABELS and 0 <= confidence <= 1
