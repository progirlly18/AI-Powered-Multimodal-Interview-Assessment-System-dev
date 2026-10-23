"""
Temporal smoothing of facial-expression probabilities for display.

The raw model output flickers from frame to frame. For the live panel the
probability vectors from the last few seconds are averaged and the top
class of the average is shown, together with how many predictions it is
based on. Raw and smoothed values are kept separate; neither is scored.

Expression labels are uncertain model outputs. They are not measurements
of personality, honesty, confidence or ability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from backend.vision.emotion_detector import EMOTION_LABELS
from backend.vision.face_analysis import FrameAnalysis

DEFAULT_WINDOW_S = 3.0
MIN_SAMPLES = 2


@dataclass(frozen=True)
class SmoothedEmotion:
    label: str
    probability: float      # mean softmax probability of ``label`` over the window
    samples: int            # number of model predictions averaged
    window_s: float


def smooth_emotion(
    analyses: Sequence[FrameAnalysis],
    now: float,
    window_s: float = DEFAULT_WINDOW_S,
    min_samples: int = MIN_SAMPLES,
) -> Optional[SmoothedEmotion]:
    """Average the expression probabilities predicted in the last ``window_s``.

    Returns None when fewer than ``min_samples`` predictions are available,
    so the display says "not enough data" rather than showing a single
    noisy frame as if it were stable.
    """
    vectors = [
        np.asarray(a.emotion_probs, dtype=np.float64)
        for a in analyses
        if a.emotion_probs is not None and now - a.timestamp <= window_s
    ]
    if len(vectors) < min_samples:
        return None
    mean = np.mean(vectors, axis=0)
    idx = int(np.argmax(mean))
    return SmoothedEmotion(EMOTION_LABELS[idx], float(mean[idx]), len(vectors), window_s)


def latest_raw(analyses: Sequence[FrameAnalysis]) -> Optional[FrameAnalysis]:
    """Most recent analysis in which the expression model actually ran."""
    for a in reversed(analyses):
        if a.emotion_probs is not None:
            return a
    return None
