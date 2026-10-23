"""
Pure speech-delivery metrics (no models, no devices).

Kept separate from recording and transcription so every rule can be unit
tested with synthetic input.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from backend.core.results import Metric

# Hesitation sounds. Only these are counted (and penalised) as fillers.
# Note: Whisper often omits them from transcripts, so counts are a lower bound.
FILLERS: tuple[str, ...] = ("um", "umm", "uh", "uhm", "erm", "er", "ah", "hmm", "mm")

# Discourse markers that are often fillers but are also ordinary words
# ("I like Python", "so the model..."). Reported for coaching, never penalised.
POSSIBLE_FILLERS: tuple[str, ...] = (
    "like", "so", "actually", "basically", "you know", "kind of", "sort of", "i mean",
)

# Minimum speaking time before a words-per-minute figure is meaningful.
MIN_SPEAKING_SECONDS = 3.0

# Below this RMS level the microphone most likely recorded silence
# (heuristic; used only to explain an empty transcript, never to score).
SILENCE_DBFS = -50.0

# YAMNet classes that suggest competing audio. Informational only.
DISTRACTING_SOUNDS: tuple[str, ...] = (
    "Music", "Television", "Radio", "Conversation", "Crowd", "Babble",
    "Typing", "Computer keyboard", "Vehicle", "Traffic", "Siren", "Dog", "Vacuum cleaner",
)
BACKGROUND_NOTE_MIN_CONFIDENCE = 0.30


def count_phrases(text: str, phrases: Iterable[str]) -> dict[str, int]:
    """Whole-word, case-insensitive counts of each phrase (zeros omitted)."""
    lowered = text.lower()
    counts: dict[str, int] = {}
    for phrase in phrases:
        pattern = r"(?<![\w'])" + re.escape(phrase) + r"(?![\w'])"
        n = len(re.findall(pattern, lowered))
        if n:
            counts[phrase] = n
    return counts


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?", text))


def speaking_seconds_from_segments(
    segments: Optional[Sequence[Mapping[str, float]]], recording_seconds: float
) -> Optional[float]:
    """Total duration of transcribed speech segments, clipped to the recording."""
    if not segments:
        return None
    total = 0.0
    for seg in segments:
        start = max(0.0, float(seg.get("start", 0.0)))
        end = min(recording_seconds, float(seg.get("end", 0.0)))
        if math.isfinite(start) and math.isfinite(end) and end > start:
            total += end - start
    return total if total > 0 else None


def words_per_minute(word_count: int, speaking_seconds: Optional[float]) -> Metric:
    name = "Speaking pace"
    if word_count <= 0:
        return Metric.unavailable(name, "No words recognised", " wpm")
    if speaking_seconds is None or not math.isfinite(speaking_seconds):
        return Metric.unavailable(name, "Speaking time unknown", " wpm")
    if speaking_seconds < MIN_SPEAKING_SECONDS:
        return Metric.unavailable(
            name, f"Less than {MIN_SPEAKING_SECONDS:.0f} s of speech", " wpm"
        )
    return Metric.available(name, word_count / (speaking_seconds / 60.0), " wpm")


def rms_dbfs(audio: np.ndarray) -> float:
    """RMS level in dB relative to full scale. Accepts int16 or float in [-1, 1]."""
    if audio.size == 0:
        return float("-inf")
    data = audio.astype(np.float64)
    if np.issubdtype(audio.dtype, np.integer):
        data = data / 32768.0
    rms = float(np.sqrt(np.mean(np.square(data))))
    return 20.0 * math.log10(rms) if rms > 0 else float("-inf")


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Mono int16 PCM -> float32 in [-1, 1) as Whisper and YAMNet expect."""
    if audio.dtype != np.int16:
        raise ValueError(f"Expected int16 audio, got {audio.dtype}")
    return (audio.reshape(-1).astype(np.float32)) / 32768.0


def background_note(label: Optional[str], confidence: Optional[float]) -> Optional[str]:
    """A short note if the dominant sound suggests competing audio."""
    if not label or confidence is None or confidence < BACKGROUND_NOTE_MIN_CONFIDENCE:
        return None
    if any(word.lower() in label.lower() for word in DISTRACTING_SOUNDS):
        return f"Background sound detected: {label}. A quieter room may improve transcription."
    return None
