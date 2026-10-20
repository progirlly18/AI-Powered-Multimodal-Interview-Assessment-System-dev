"""Legacy wrapper kept for compatibility. New code should use SpeechAnalyzer."""

import numpy as np

from backend.audio.speech_analyzer import default_speech_analyzer
from backend.core.results import SpeechResult


class AudioPipeline:

    def __init__(self):
        self._analyzer = default_speech_analyzer()

    def analyze(self, audio: np.ndarray) -> SpeechResult:
        return self._analyzer.analyze(audio)
