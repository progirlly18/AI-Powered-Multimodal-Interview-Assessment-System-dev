# Setup on Linux / WSL (untested)

**This configuration has not been tested.** The steps below are expected to work, but they have not been verified.

## Linux (native)

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-dev.txt
sudo apt install libportaudio2      # needed by sounddevice
.venv/bin/python -m streamlit run frontend/app.py
```

- `requirements-lock.txt` was captured on Windows. Use `requirements.txt` on Linux and let pip resolve the platform-specific dependencies.
- The camera is `/dev/video0` (set `INTERVIEW_AI_CAMERA_INDEX` if needed); your user must be in the `video` group.

## WSL 2

- WSL has no direct access to the Windows webcam or microphone by default:
  - USB webcams need `usbipd-win` passthrough;
  - audio input needs WSLg / PulseAudio.
- Running the app in WSL is therefore **not recommended**. Run it natively on Windows (see SETUP_WINDOWS.md).
- Audio-only mode and the unit tests do not need devices: `.venv/bin/python -m pytest`.
