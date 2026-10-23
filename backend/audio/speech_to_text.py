"""
Developer utility: transcribe a WAV file with Whisper.

    python -m backend.audio.speech_to_text path\\to\\clip.wav

Reads 16 kHz mono int16 WAV files directly (no FFmpeg needed).
"""

import argparse

import numpy as np
from scipy.io.wavfile import read

from backend.audio.speech_metrics import FILLERS, count_phrases
from backend.core.models import get_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    rate, audio = read(args.path)
    if rate != 16000 or audio.dtype != np.int16:
        raise SystemExit("Expected a 16 kHz int16 WAV file.")
    if audio.ndim > 1:
        audio = audio[:, 0]
    result = get_registry().get("whisper").transcribe(
        audio.astype(np.float32) / 32768.0, fp16=False
    )
    transcript = result["text"].strip()
    print("\nTranscript:\n" + transcript)
    print("\nFillers:", count_phrases(transcript, FILLERS) or "none")


if __name__ == "__main__":
    main()
