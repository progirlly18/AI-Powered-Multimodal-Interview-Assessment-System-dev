"""Synthetic speech for tests, generated with the Windows speech synthesiser.

The audio is computer-generated from fixed text, so it contains no personal
recording. Returns None when synthesis is not available (e.g. not Windows).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def synthesize(text: str, path: Path) -> Optional[np.ndarray]:
    if sys.platform != "win32":
        return None
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono); "
        f"$s.SetOutputToWaveFile('{path}', $f); "
        f"$s.Speak('{text.replace(chr(39), '')}'); $s.Dispose()"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    from scipy.io.wavfile import read

    rate, audio = read(str(path))
    if rate != 16000 or audio.dtype != np.int16:
        return None
    return audio.reshape(-1)


def word_error_rate(reference: str, hypothesis: str) -> float:
    import re

    ref = re.findall(r"[a-z0-9']+", reference.lower())
    hyp = re.findall(r"[a-z0-9']+", hypothesis.lower())
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(ref), len(hyp)] / max(1, len(ref))
