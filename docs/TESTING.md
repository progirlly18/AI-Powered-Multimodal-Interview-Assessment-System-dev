# Testing

| Suite | Command | Needs |
|---|---|---|
| Fast unit tests (default) | `.\.venv\Scripts\python.exe -m pytest` | nothing (fake camera, microphone and models) |
| Integration (real models) | `.\.venv\Scripts\python.exe -m pytest -m integration` | model downloads on first run; Windows speech synthesiser for the speech test |
| Benchmarks | `.\.venv\Scripts\python.exe scripts\benchmark.py` (add `--hardware` to time camera and microphone) | models; results go to `benchmarks/results/` |
| Manual hardware checks | see below | webcam, microphone, a person |

## Unit tests (`tests/unit`)

| File | Covers |
|---|---|
| `test_scoring_engine.py` | every scoring rule and the worked examples in SCORING.md; range, NaN and missing-data handling |
| `test_states_and_session.py` | run-state and page transitions; consent; one result per question; retry; repeated sessions |
| `test_question_bank.py` | JSON validation; empty or malformed banks; first and last question; invalid indices; rubric hidden from the candidate view |
| `test_answer_evaluator.py` | phrase matching; empty transcripts; encoder or model failures; reference answers |
| `test_speech.py` | fillers vs ordinary words; pace from speaking time; silence; transcription failure; quiet room not penalised |
| `test_vision.py` | gaze and head rules; aggregation; no-face handling; guidance smoothing; capture vs analysis threads; disconnects; idempotent release |
| `test_models_registry.py` | load once; concurrency; failures reported and retryable; missing files |
| `test_interview_service.py` | full runs with fakes: success, camera or microphone unavailable or disconnected, cancellation, concurrency lock, no fabricated scores, recording retention |
| `test_config.py` | environment validation; Streamlit security settings; `.env.example`; `.gitignore` |
| `test_app_ui.py` | headless Streamlit (`AppTest`): full flow; Continue disabled until ready; consent; audio-only fallback; HTML escaping |

All synthetic data is generated in the tests. No personal recordings are used.

## Manual hardware verification

These steps need a person at the computer.

1. Run `.\.venv\Scripts\python.exe -m streamlit run frontend/app.py`.
2. **Readiness:**
   - Press *Load models*; each model shows ✅ and a load time.
   - Press *Check devices*; the camera and microphone show 🟢.
   - Cover or unplug the camera and check again; it should show 🔴, and the audio-only option should appear.
3. **Recording:**
   - The video appears during "Starting camera…" and the countdown.
   - The red "Recording" indicator and timer appear only after the countdown.
   - Live gaze, head and expression values update about every 0.3 s.
4. Look away for several seconds; the guidance message changes after about 2 s, and a single glance does not trigger it.
5. Press *Stop recording* during recording; the result is "Cancelled. Nothing was saved." and the camera light turns off.
6. **After a normal recording:**
   - "Processing — camera and microphone are off" is shown while transcribing.
   - The review page shows the score breakdown, the transcript, and an audio/video start offset under *Technical details*.
7. Mute the microphone and record; the review says no sound reached the microphone, and the score is **withheld**.
8. Open the app in a second tab and start recording in both; the second shows "Another recording is already running".
