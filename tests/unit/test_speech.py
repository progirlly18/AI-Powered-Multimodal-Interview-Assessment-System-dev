"""Speech metrics and the speech analyser (fake Whisper/YAMNet)."""

import numpy as np
import pytest

from backend.audio.speech_analyzer import SpeechAnalyzer
from backend.audio.speech_metrics import (
    FILLERS,
    POSSIBLE_FILLERS,
    background_note,
    count_phrases,
    count_words,
    int16_to_float32,
    rms_dbfs,
    speaking_seconds_from_segments,
    words_per_minute,
)
from backend.core.results import Status
from tests.conftest import YAMNET_LABELS, FakeWhisper, FakeYamnet

SR = 16000


def tone(seconds=5.0, amplitude=8000):
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * 220 * t) * amplitude).astype(np.int16)


# ---------------- pure metrics ----------------

def test_ordinary_words_are_not_fillers():
    """Regression: 'like', 'so' and 'actually' were counted and penalised."""
    text = "So I actually like Python, so I use it."
    assert count_phrases(text, FILLERS) == {}
    assert count_phrases(text, POSSIBLE_FILLERS) == {"like": 1, "so": 2, "actually": 1}


def test_hesitation_fillers_counted_as_whole_words():
    text = "Um, I think, uh, the umbrella... erm, hmm. Uhm."
    counts = count_phrases(text, FILLERS)
    assert counts == {"um": 1, "uh": 1, "uhm": 1, "erm": 1, "hmm": 1}  # not 'umbrella'


def test_count_words():
    assert count_words("It's a model's output, 42 times.") == 6
    assert count_words("") == 0


def test_speaking_time_from_segments():
    segments = [{"start": 0.5, "end": 3.0}, {"start": 4.0, "end": 9.0}, {"start": 9.5, "end": 12.0}]
    assert speaking_seconds_from_segments(segments, recording_seconds=10.0) == pytest.approx(2.5 + 5 + 0.5)
    assert speaking_seconds_from_segments([], 10) is None
    assert speaking_seconds_from_segments([{"start": 5, "end": 5}], 10) is None


def test_wpm_uses_speaking_time():
    """Regression: WPM was divided by the whole recording window."""
    m = words_per_minute(60, 20.0)  # 60 words in 20 s of speech
    assert m.value == pytest.approx(180)


@pytest.mark.parametrize("words,seconds,reason", [
    (0, 20, "No words"), (10, 2.0, "Less than"), (10, None, "unknown"), (10, float("nan"), "unknown"),
])
def test_wpm_unavailable(words, seconds, reason):
    m = words_per_minute(words, seconds)
    assert not m.is_available and reason in m.reason


def test_rms_and_conversion():
    assert rms_dbfs(np.zeros(100, dtype=np.int16)) == float("-inf")
    assert -20 < rms_dbfs(tone()) < -10
    f = int16_to_float32(np.array([-32768, 0, 16384], dtype=np.int16))
    assert f.dtype == np.float32 and f.tolist() == [-1.0, 0.0, 0.5]
    with pytest.raises(ValueError):
        int16_to_float32(np.zeros(3, dtype=np.float32))


def test_background_note():
    assert background_note("Speech", 0.9) is None           # the candidate's voice is fine
    assert background_note("Silence", 0.9) is None          # a quiet room is fine
    assert "Music" in background_note("Music", 0.5)
    assert background_note("Music", 0.1) is None            # low confidence


# ---------------- analyser ----------------

def analyzer(whisper, yamnet=None):
    return SpeechAnalyzer(lambda: whisper, (lambda: (yamnet, YAMNET_LABELS)) if yamnet else None)


def test_successful_analysis_passes_float_audio_in_memory():
    whisper = FakeWhisper("Um, overfitting means poor generalization on new data, so we regularize.",
                          segments=[{"start": 0.0, "end": 4.0}])
    r = analyzer(whisper, FakeYamnet(0)).analyze(tone(5.0))
    audio, kwargs = whisper.received
    assert audio.dtype == np.float32 and audio.ndim == 1          # no file, no FFmpeg
    assert kwargs["fp16"] is False and kwargs["language"] == "en"
    assert r.status is Status.AVAILABLE
    assert r.word_count == 11 and r.filler_count == 1
    assert r.wpm.value == pytest.approx(11 / (4 / 60))
    assert r.speaking_time_is_estimate is False
    assert r.possible_filler_counts == {"so": 1}
    # 165 wpm is inside 90-170, so only the filler penalty applies.
    assert r.delivery_score.value == pytest.approx(100 - 3)
    assert r.background_label == "Speech" and r.background_note is None


def test_quiet_room_is_not_penalised():
    """Regression: anything other than rain/wind/birds scored lower."""
    whisper = FakeWhisper("A clear answer with enough words to measure pace properly today.",
                          segments=[{"start": 0, "end": 5}])
    speech_room = analyzer(whisper, FakeYamnet(0)).analyze(tone())
    silent_room = analyzer(whisper, FakeYamnet(2)).analyze(tone())
    assert speech_room.delivery_score.value == silent_room.delivery_score.value


def test_missing_segments_fall_back_to_recording_length_and_flag_it():
    whisper = FakeWhisper("one two three four five six seven eight nine ten", segments=None)
    r = analyzer(whisper).analyze(tone(6.0))
    assert r.speaking_time_is_estimate is True
    assert r.wpm.value == pytest.approx(10 / (6 / 60))


def test_transcription_failure_is_failed_not_empty_text():
    r = analyzer(FakeWhisper(fail=True)).analyze(tone())
    assert r.status is Status.FAILED and r.transcript is None
    assert not r.delivery_score.is_available


def test_silence_is_explained():
    r = analyzer(FakeWhisper("")).analyze(np.zeros(SR * 3, dtype=np.int16))
    assert r.status is Status.UNAVAILABLE and "muted" in r.reason


def test_unrecognised_sound_is_explained():
    r = analyzer(FakeWhisper("  ")).analyze(tone())
    assert r.status is Status.UNAVAILABLE and "no words" in r.reason.lower()


def test_too_short_recording():
    r = analyzer(FakeWhisper("hello")).analyze(np.zeros(100, dtype=np.int16))
    assert r.status is Status.UNAVAILABLE and r.reason == "No audio was recorded."


def test_background_failure_does_not_fail_speech():
    whisper = FakeWhisper("enough words here to count as an answer for sure", [{"start": 0, "end": 4}])
    r = analyzer(whisper, FakeYamnet(fail=True)).analyze(tone())
    assert r.status is Status.AVAILABLE
    assert r.background_label is None and "could not run" in r.background_note


def test_wrong_sample_rate_rejected():
    with pytest.raises(ValueError):
        analyzer(FakeWhisper("x")).analyze(tone(), sample_rate=44100)
