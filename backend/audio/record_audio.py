"""
Developer utility: record a short WAV clip from the default microphone.

    python -m backend.audio.record_audio --seconds 10

The clip is written to the git-ignored ``recordings/`` folder with a unique
name (it previously overwrote the tracked ``sample.wav``).
"""

import argparse
import time

import sounddevice as sd
from scipy.io.wavfile import write

from backend.config import load_settings

FS = 16000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()

    folder = load_settings().recordings_dir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"clip-{time.strftime('%Y%m%d-%H%M%S')}.wav"

    print("Recording will start in 3 seconds...")
    sd.sleep(3000)
    print("Recording... Speak now!")
    recording = sd.rec(int(args.seconds * FS), samplerate=FS, channels=1, dtype="int16")
    sd.wait()
    write(str(path), FS, recording)
    print(f"Recording saved as {path}")


if __name__ == "__main__":
    main()
