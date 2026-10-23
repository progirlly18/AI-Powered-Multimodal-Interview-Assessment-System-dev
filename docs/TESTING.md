# Testing

| Suite | Command | Needs |
|---|---|---|
| Fast unit + headless UI tests (default) | `.\.venv\Scripts\python.exe -m pytest` | nothing (fake camera, microphone and models) |
| Integration (real models) | `.\.venv\Scripts\python.exe -m pytest -m integration` | model downloads on first run; Windows speech synthesiser for the speech test |
| Benchmarks | `.\.venv\Scripts\python.exe scripts\benchmark.py` (add `--hardware` to time camera and microphone) | models; results go to `benchmarks/results/` |
| Manual hardware checks | see below | webcam, microphone, a person |

## Unit tests (`tests/unit`)

| File | Covers |
|---|---|
| `test_scoring_engine.py` | every scoring rule and the worked examples in SCORING.md (including the rubric); ranges, NaN, missing data; de-duplicated suggestions |
| `test_rubric_and_text_features.py` | rubric validation (weights sum to 1, all criteria present); elaboration, structure and examples heuristics |
| `test_answer_evaluator.py` | phrase matching; empty transcripts; encoder, model or rubric failures; criteria and explanations; strengths and improvements; behavioural weighting |
| `test_emotion_pipeline.py` | probability validation; 3 s smoothing vs. raw; sampling once per interval; multiple faces; expression never affects scores |
| `test_storage.py` | session records; SQLite save/list/get/delete/clear; transcripts excluded by default; damaged rows; CSV and HTML export (escaping, partial, not-measured) |
| `test_states_and_session.py` | run-state and page transitions (incl. history); consent; one result per question; retry; repeated sessions |
| `test_question_bank.py` | JSON validation; empty or malformed banks; first and last question; invalid indices; rubric hidden from the candidate |
| `test_speech.py` | fillers vs. ordinary words; pace from speaking time; silence; transcription failure; quiet room not penalised |
| `test_vision.py` | gaze and head rules; aggregation; no-face; guidance smoothing; capture vs. analysis threads; disconnects; release |
| `test_models_registry.py` | load once; concurrency; failures reported and retryable; missing files and packages |
| `test_interview_service.py` | full runs with fakes: success, devices unavailable or disconnected, cancellation, concurrency lock, no fabricated scores, recording retention |
| `test_config.py` | environment validation; Streamlit security settings; `.env.example`; `.gitignore` |
| `test_app_ui.py` | headless Streamlit: landing; full flow with HTML-injection name and transcript; save to history and export; difficulty filter; device gating; consent; audio-only fallback; empty history |

All synthetic data is generated in the tests. No personal recordings are used.

## Manual hardware verification

These steps need a person at the computer.

1. Run:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run frontend/app.py
   ```

   Then press **Start Interview**.

2. **Configure:** choose *Machine Learning Engineer*, difficulty *Medium*, the question "What is overfitting and how can you prevent it?", and 30 s.
3. **System check:**
   - **Load models** shows ✅ with a time for each model.
   - **Check devices** shows both *Ready* and a camera preview frame.
   - Cover the camera and check again: *Unavailable*, plus the audio-only option.
4. **Consent:** tick the box, then press **Start interview**.
5. **Console:**
   - Press **Start recording**. The chips show *Camera live* and *Get ready · not recording* during the countdown, then *Recording* with a timer.
   - The behavioral-signal bars and the expression card start updating. Expression shows "Collecting predictions…" until at least 2 predictions exist.
6. Look away for 3–4 s. The coaching hint changes after about 2 s; a single glance does not trigger it.
7. **Second face:** have a second person lean into view. After about 2 s, "More than one face is in view…" appears.
8. Read this aloud:

   > "First, overfitting happens when a model performs very well on the training data but does not generalize to new data, because it memorises noise. For example, in my project accuracy fell from 98% to 70% on test data. However, regularization and cross validation help. Overall, simpler models generalize better."

   The review should show a transcript close to that, high *Structure* and *Examples* scores, and several concepts covered.

9. **Stop recording** during a retry: "Cancelled. Nothing was saved.", and the camera light turns off.
10. **Muted microphone:** "No sound reached the microphone…", and the score is shown as **Withheld**.
11. **Finish → dashboard:**
    - **Download report** opens as a styled HTML page; print it to PDF.
    - **Download data** opens as a CSV in Excel.
    - **Save to history**, then open **History**: the session appears. Delete it.
12. **Cleanup checks:**

    ```powershell
    git status --short
    ```

    This should show no `sample.wav` and no recordings.

    ```powershell
    Test-Path recordings
    ```

    This should print `False`. Also confirm that `logs\app.log` contains no transcript text.
