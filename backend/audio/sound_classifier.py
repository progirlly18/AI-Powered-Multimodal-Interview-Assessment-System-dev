"""
Developer utility: show the dominant YAMNet sound class of a WAV file.

    python -m backend.audio.sound_classifier path\\to\\clip.wav

The label is informational. It is not evidence of malpractice: the
original "possible external assistance" verdict for any speech-like sound
was removed because the candidate's own voice is speech.
"""

import argparse

import numpy as np
from scipy.io.wavfile import read

from backend.audio.speech_metrics import background_note
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
    model, labels = get_registry().get("yamnet")
    scores, _, _ = model(audio.astype(np.float32) / 32768.0)
    mean_scores = np.mean(scores.numpy(), axis=0)
    idx = int(np.argmax(mean_scores))
    print(f"Detected sound: {labels[idx]} (confidence {mean_scores[idx]:.3f})")
    print(background_note(labels[idx], float(mean_scores[idx])) or "No competing background sound flagged.")


if __name__ == "__main__":
    main()
