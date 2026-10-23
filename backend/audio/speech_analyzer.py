"""
Speech analysis of a recorded answer: transcription (Whisper), delivery
metrics and an informational background-sound label (YAMNet).

Audio is passed to the models as in-memory arrays. Nothing is written to
disk, so no FFmpeg install is needed and no recording is left behind in the
project folder (the original code overwrote the tracked sample.wav).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import numpy as np

from backend.audio.speech_metrics import (
    FILLERS,
    POSSIBLE_FILLERS,
    SILENCE_DBFS,
    background_note,
    count_phrases,
    count_words,
    int16_to_float32,
    rms_dbfs,
    speaking_seconds_from_segments,
    words_per_minute,
)
from backend.core.results import Metric, SpeechResult, Status
from backend.scoring.scoring_engine import delivery_score

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MIN_RECORDING_SECONDS = 1.0


class SpeechAnalyzer:
    """Turns recorded audio into a ``SpeechResult``."""

    def __init__(
        self,
        whisper_provider: Callable[[], Any],
        yamnet_provider: Optional[Callable[[], Any]] = None,
        language: Optional[str] = "en",
    ) -> None:
        self._whisper_provider = whisper_provider
        self._yamnet_provider = yamnet_provider
        self._language = language
        self.timings: dict[str, float] = {}

    def _classify_background(self, waveform: np.ndarray) -> tuple[Optional[str], Optional[float]]:
        if self._yamnet_provider is None:
            return None, None
        model, labels = self._yamnet_provider()
        scores, _embeddings, _spectrogram = model(waveform)
        mean_scores = np.mean(np.asarray(scores), axis=0)
        idx = int(np.argmax(mean_scores))
        return labels[idx], float(mean_scores[idx])

    def analyze(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> SpeechResult:
        self.timings = {}
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
        audio = np.asarray(audio).reshape(-1)
        recording_seconds = audio.size / sample_rate

        if recording_seconds < MIN_RECORDING_SECONDS:
            return SpeechResult(
                status=Status.UNAVAILABLE,
                reason="No audio was recorded.",
                recording_seconds=recording_seconds,
            )

        waveform = int16_to_float32(audio)
        level = rms_dbfs(audio)

        # ---- transcription ----
        start = time.perf_counter()
        try:
            model = self._whisper_provider()
            kwargs: dict[str, Any] = {"fp16": False}
            if self._language:
                kwargs["language"] = self._language
            output = model.transcribe(waveform, **kwargs)
            transcript = str(output.get("text", "")).strip()
            segments = output.get("segments") or []
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            log.exception("Transcription failed")
            self.timings["transcription_s"] = time.perf_counter() - start
            return SpeechResult(
                status=Status.FAILED,
                reason=f"Speech could not be transcribed ({type(exc).__name__}).",
                recording_seconds=recording_seconds,
            )
        self.timings["transcription_s"] = time.perf_counter() - start

        word_count = count_words(transcript)
        if word_count == 0:
            if level < SILENCE_DBFS:
                reason = ("No sound reached the microphone. Check that it is not muted "
                          "and is selected as the Windows input device.")
            else:
                reason = "Sound was recorded, but no words could be recognised."
            return SpeechResult(
                status=Status.UNAVAILABLE,
                reason=reason,
                transcript="",
                recording_seconds=recording_seconds,
            )

        # ---- delivery ----
        speaking = speaking_seconds_from_segments(segments, recording_seconds)
        estimate = speaking is None
        if estimate:
            speaking = recording_seconds
        wpm = words_per_minute(word_count, speaking)
        fillers = count_phrases(transcript, FILLERS)
        filler_total = sum(fillers.values())

        # ---- background (informational; failures do not fail the analysis) ----
        label: Optional[str] = None
        confidence: Optional[float] = None
        note: Optional[str] = None
        start = time.perf_counter()
        try:
            label, confidence = self._classify_background(waveform)
            note = background_note(label, confidence)
        except Exception:  # noqa: BLE001
            log.exception("Background classification failed")
            note = "The background-sound check could not run."
        self.timings["background_s"] = time.perf_counter() - start

        return SpeechResult(
            status=Status.AVAILABLE,
            transcript=transcript,
            recording_seconds=recording_seconds,
            speaking_seconds=speaking,
            speaking_time_is_estimate=estimate,
            word_count=word_count,
            wpm=wpm,
            filler_count=filler_total,
            filler_counts=fillers,
            possible_filler_counts=count_phrases(transcript, POSSIBLE_FILLERS),
            delivery_score=delivery_score(wpm, filler_total),
            background_label=label,
            background_confidence=confidence,
            background_note=note,
        )


def default_speech_analyzer() -> SpeechAnalyzer:
    from backend.core.models import get_registry

    registry = get_registry()
    return SpeechAnalyzer(
        whisper_provider=lambda: registry.get("whisper"),
        yamnet_provider=lambda: registry.get("yamnet"),
    )
