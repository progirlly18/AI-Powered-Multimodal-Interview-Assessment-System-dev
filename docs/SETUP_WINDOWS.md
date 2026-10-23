# Setup on Windows (tested)

Tested on Windows 11 with Python 3.11.9.

## 1. Prerequisites

- **Python 3.11** (64-bit) from python.org. TensorFlow 2.21 and MediaPipe 1.0 wheels are built for 3.11.
- A webcam and microphone, for camera mode.
- About **4 GB** of free disk space for the virtual environment, plus about 300 MB for downloaded models.
- **FFmpeg is not required.** The app passes audio to Whisper as an in-memory array.

## 2. Create the environment

From the project folder, in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # tests and benchmarks
```

- `requirements-lock.txt` is the exact, tested package set.
- `requirements.txt` lists only the direct dependencies, pinned to the same versions.

Both `opencv-python` (required by cvzone) and `opencv-contrib-python` (required by mediapipe) get installed, and both provide `cv2`. Whichever installs last is the one that loads; the tested environment loaded `cv2 4.10.0`. Check yours with:

```powershell
.\.venv\Scripts\python.exe -c "import cv2; print(cv2.__version__)"
```

## 3. Model files

| File | Location | Notes |
|---|---|---|
| Emotion model | `backend/models/best_model.keras` | in the repository (28 MB) |
| Face landmarks | `face_landmarker.task` (project root) | in the repository (3.7 MB) |
| Whisper `base` | `%USERPROFILE%\.cache\whisper` | downloaded on first use |
| MiniLM | `%USERPROFILE%\.cache\huggingface` | downloaded on first use |
| YAMNet | TF Hub cache (`%TEMP%\tfhub_modules`, or `TFHUB_CACHE_DIR`) | downloaded on first use |

The first model load needs an internet connection and can take several minutes.

## 4. Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

- Open the address it prints (http://localhost:8501). The server only accepts connections from this computer.
- Settings are optional environment variables; see `.env.example`. Example:

```powershell
$env:INTERVIEW_AI_WHISPER_MODEL = "tiny"
```

- **Command-line version** (same backend, no browser):

```powershell
.\.venv\Scripts\python.exe -m backend.app --role "Machine Learning Engineer" --question 3 --seconds 20
```

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Camera "could not be opened" | Close other apps using the camera (Teams, Zoom, browser tabs). Check Settings > Privacy & security > Camera > "Let desktop apps access your camera". Try `INTERVIEW_AI_CAMERA_INDEX=1`. |
| Microphone "could not be opened" or "No sound reached the microphone" | Check Settings > Privacy & security > Microphone, and that the right input device is the Windows default (Settings > System > Sound > Input). |
| A model "could not be loaded" | Check your internet connection for the first download, then press *Load models* again. Details are in `logs/app.log`. |
| First page appears but loading models takes minutes | This is expected on the first run: TensorFlow, torch, Whisper and MiniLM take tens of seconds each to import. Later loads in the same app session are instant. |
| "Another recording is already running" | Only one recording can run at a time; finish or stop it in the other tab. |
| `face_landmarker.task` downloaded into another folder | That was cvzone's behaviour. The app now loads the file from the project root and never downloads it. |
