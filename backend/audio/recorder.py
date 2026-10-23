"""
Microphone capture.

The stream is opened (and so verified) before the countdown. Samples are
discarded until ``start_capture`` is called, so the countdown is never part
of the recording and the stream is already running when capture starts.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional, Protocol

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
BLOCK_SIZE = 1024  # 64 ms at 16 kHz


class DeviceUnavailableError(RuntimeError):
    """A camera or microphone could not be opened or stopped working."""

    def __init__(self, device: str, message: str) -> None:
        super().__init__(message)
        self.device = device


class AudioSource(Protocol):
    sample_rate: int

    def open(self) -> None: ...
    def start_capture(self) -> None: ...
    def stop_capture(self) -> None: ...
    def close(self) -> None: ...
    def audio(self) -> np.ndarray: ...
    @property
    def first_sample_time(self) -> Optional[float]: ...
    @property
    def error(self) -> Optional[str]: ...


class MicrophoneRecorder:
    """Records mono 16 kHz int16 audio from the default input device."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, device: Optional[int] = None) -> None:
        self.sample_rate = sample_rate
        self._device = device
        self._stream: Any = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._capturing = False
        self._first_sample_time: Optional[float] = None
        self._status_flags = 0
        self._error: Optional[str] = None

    # -- stream callback (runs on the PortAudio thread) --
    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self._status_flags += 1
        if not self._capturing:
            return
        with self._lock:
            if self._first_sample_time is None:
                self._first_sample_time = time.perf_counter()
            self._chunks.append(indata.copy().reshape(-1))

    def open(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001
            raise DeviceUnavailableError(
                "microphone", "The audio library (sounddevice/PortAudio) is not available."
            ) from exc
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype="int16",
                blocksize=BLOCK_SIZE,
                device=self._device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            self.close()
            raise DeviceUnavailableError(
                "microphone",
                "The microphone could not be opened. Check that one is connected, "
                "that Windows allows apps to use it (Settings > Privacy & security > "
                "Microphone) and that no other app is using it.",
            ) from exc

    def start_capture(self) -> None:
        with self._lock:
            self._chunks = []
            self._first_sample_time = None
        self._capturing = True

    def stop_capture(self) -> None:
        self._capturing = False

    def close(self) -> None:
        self._capturing = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:  # noqa: BLE001
                log.warning("Stopping the microphone stream failed", exc_info=True)
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                log.warning("Closing the microphone stream failed", exc_info=True)

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    @property
    def first_sample_time(self) -> Optional[float]:
        return self._first_sample_time

    @property
    def error(self) -> Optional[str]:
        if self._stream is not None and not getattr(self._stream, "active", True):
            return "The microphone stream stopped unexpectedly."
        return self._error

    @property
    def overflow_count(self) -> int:
        return self._status_flags

    def audio(self) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.int16)
            return np.concatenate(self._chunks).astype(np.int16, copy=False)


MicrophoneFactory = Callable[[], AudioSource]


def check_microphone(open_seconds: float = 0.3) -> tuple[bool, str]:
    """Open the default input briefly. No audio is kept."""
    recorder = MicrophoneRecorder()
    try:
        recorder.open()
        time.sleep(open_seconds)
        if recorder.error:
            return False, recorder.error
        try:
            import sounddevice as sd

            name = sd.query_devices(kind="input")["name"]
        except Exception:  # noqa: BLE001
            name = "default input"
        return True, f"Microphone ready ({name})."
    except DeviceUnavailableError as exc:
        return False, str(exc)
    finally:
        recorder.close()
