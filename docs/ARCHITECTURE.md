# Architecture

```
frontend/app.py            Streamlit pages (one per session Stage); polls LiveStatus
frontend/ui_components.py  Rendering helpers (native Streamlit components only)

backend/core/
  results.py   Metric (available / unavailable / failed) and result dataclasses
  states.py    RunState machine (one recording) and DeviceStatus
  session.py   InterviewSession: page flow, question order, outcomes
  models.py    ModelRegistry: lazy, cached, timed model loading
backend/config.py           Validated environment settings
backend/services/interview_service.py
               InterviewService.run_question(): devices -> countdown ->
               recording -> processing -> QuestionResult
backend/vision/
  capture.py        VisionSession: capture thread + analysis thread, cleanup
  face_analysis.py  One landmark pass -> gaze, head pose, expression
  landmarks.py      MediaPipe FaceLandmarker (absolute model path)
  eye_contact.py, head_pose.py, emotion_detector.py   pure rules / model call
  guidance.py       Smoothed live guidance
backend/audio/
  recorder.py        MicrophoneRecorder (sounddevice InputStream)
  speech_analyzer.py Whisper (in memory) + delivery metrics + YAMNet label
  speech_metrics.py  Pure rules: fillers, pace, levels
backend/ai/
  question_bank.py   data/questions.json loader + validation
  answer_evaluator.py  Topic relevance + concept coverage
backend/scoring/scoring_engine.py  Overall practice score (docs/SCORING.md)
```

## Interview workflow

```
WELCOME -> SETUP -> READINESS -> CONSENT -> QUESTION <-> REVIEW -> SUMMARY
```

1. **Setup:** choose role, questions (in order), answer time, and camera on or off.
2. **Readiness:**
   - "Load models" loads each model and shows its real load time.
   - "Check devices" opens the microphone and camera briefly; nothing is kept.
   - Continue is enabled only when everything required is ready.
   - If the camera fails, the user can continue in an explicitly labelled audio-only mode.
3. **Consent:** explains when recording happens and what is kept; requires a checkbox.
4. **Question:** only the question text is shown, never the concepts or reference answers. *Start recording* starts a worker thread.
5. **Review:** full score breakdown, or a clear failure message with Retry. Each question's outcome is recorded exactly once; Retry discards it.
6. **Summary:** per-question status and the average over scored questions.

## One recording (`RunState`)

```
IDLE -> INITIALIZING -> COUNTDOWN -> RECORDING -> PROCESSING -> COMPLETED
          |   \-> UNAVAILABLE (device could not be opened)
          \-> FAILED / CANCELLED (from INITIALIZING, COUNTDOWN, RECORDING)
PROCESSING -> FAILED
```

- **INITIALIZING:**
  - Required models are fetched from the registry.
  - The camera is opened and must deliver a first frame; the capture thread starts.
  - The microphone stream is opened.
  - If any device fails here, nothing has been recorded.
- **COUNTDOWN:** live video is shown, but frames are not analysed and audio samples are discarded.
- **RECORDING:**
  - The microphone callback starts keeping samples, and the analysis thread starts analysing the newest frame every `vision_interval_s` (≈3 per second).
  - Both start from the same moment. The measured gap between the first kept audio block and the first kept video frame is reported.
  - Exact synchronisation is not guaranteed: it depends on the audio block size (64 ms) and the camera frame rate.
- **PROCESSING:**
  - Devices are already released.
  - Steps: Whisper transcription (in memory) → delivery metrics → YAMNet label → answer evaluation → scoring.
  - Each step is timed.
- Cancellation is possible until processing starts. Devices are released in a `finally` block on every path.

## Threads

| Thread | Does | Touches |
|---|---|---|
| Streamlit script thread | renders pages; polls `LiveStatus` at ≤ `display_fps` | UI only |
| `interview-run` | `InterviewService.run_question` | devices, models, results |
| `camera-capture` | reads frames into a single "latest frame" slot | camera |
| `face-analysis` | analyses the latest frame at a fixed interval | landmark + expression models |
| PortAudio callback | appends audio blocks while capturing | microphone buffer |

- Worker threads never call Streamlit.
- Only one recording runs per process (`_RUN_LOCK`). A second browser tab gets a clear "already running" message.
- MediaPipe and Keras models are only used from the analysis thread.

## Models

| Name | Model | Loaded from | Used for | Scored |
|---|---|---|---|---|
| `face_mesh` | MediaPipe FaceLandmarker | `face_landmarker.task` (repo root) | gaze, head pose, face box | yes (gaze, head) |
| `emotion` | EfficientNetB0 (project-trained) | `backend/models/best_model.keras` | expression label | no |
| `whisper` | OpenAI Whisper `base` | `~/.cache/whisper` (downloaded) | transcription, speech timing | via content and delivery |
| `yamnet` | YAMNet (TF Hub) | TF Hub cache (downloaded) | background label | no |
| `sentence` | `all-MiniLM-L6-v2` | Hugging Face cache (downloaded) | relevance, concept coverage | yes |

## Legacy modules

These are kept for compatibility. The app doesn't use them.

- **`backend/vision/vision_service.py` and `vision_pipeline.py`:** they open the camera in the calling thread.
- **`backend/vision/interview_analyzer.py`:** now a wrapper over `VisionSession`.
- **`backend/audio/audio_pipeline.py`:** a small wrapper.
- **`backend/audio/record_audio.py`, `speech_to_text.py`, `sound_classifier.py`:** developer scripts. They now run only via `python -m`, and write to `recordings/` instead of overwriting `sample.wav`.
- **`backend/analytics/timeline.py`:** unused.
