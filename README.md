# HUMAN NEXT — AI-Powered Multimodal Interview Assessment System

**Decode your interview potential.** Practise answering interview questions out loud and get
explainable, coaching-oriented feedback on:

- **What you said:** transcription with OpenAI Whisper, then a five-criterion answer rubric — relevance, concept coverage, elaboration, structure, examples.
- **How you said it:** speaking pace and hesitation fillers.
- **How you presented on camera:** face in view, looking at the camera, facing the camera (MediaPipe face landmarks), with live smoothed hints.
- **Facial expression:** your team's EfficientNet transfer-learning model, shown as an uncertain *model estimate* that is never scored.

> **Prototype practice aid, not a hiring decision system.** Scores are heuristic, have not been
> validated against human interviewers, and must not be used to make or justify hiring decisions.
> Gaze and expression estimates are imprecise and culturally variable; they say nothing about
> honesty, personality or ability.

**Team:** Yashica G, Balaji S, Nidhi Mishra, Gagan K. **Guide:** Mr. Saurav Kumar Topo.

## Install and run (Windows, Python 3.11)

```powershell
cd D:\DOWNLOADS\interview-ai-dev
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

- Open http://localhost:8501. The server only accepts connections from this computer.
- **Always start it with `.\.venv\Scripts\python.exe`**; plain `python` may pick up a different installation.
- The first *Load models* takes a few minutes and downloads Whisper, MiniLM and YAMNet once.
- FFmpeg is **not** needed, and **no API keys** are used.
- Details and troubleshooting: [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md). Linux/WSL (untested): [docs/SETUP_LINUX.md](docs/SETUP_LINUX.md).

**Optional settings** (environment variables; see [.env.example](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `INTERVIEW_AI_CAMERA_INDEX` | 0 | which webcam |
| `INTERVIEW_AI_WHISPER_MODEL` | base | `tiny`, `base`, `small` or `medium` |
| `INTERVIEW_AI_COUNTDOWN` | 3 | seconds before recording starts |
| `INTERVIEW_AI_VISION_INTERVAL` | 0.33 | seconds between gaze/head analyses |
| `INTERVIEW_AI_EMOTION_INTERVAL` | 1.0 | seconds between expression predictions (~0.3 s each) |
| `INTERVIEW_AI_DISPLAY_FPS` | 10 | live video refresh limit |
| `INTERVIEW_AI_HISTORY_PATH` | `instance/history.db` | history database location |
| `INTERVIEW_AI_KEEP_RECORDINGS` | false | keep answer audio as WAV (off by default) |
| `HF_HUB_OFFLINE` | — | set to `1` to stay offline after the first model download |
| `INTERVIEW_AI_REQUIRE_SIGN_IN` | true | email-code sign-in; `false` only on a private computer |
| `INTERVIEW_AI_SMTP_HOST` / `_USERNAME` / `_PASSWORD` | — | email account that sends sign-in codes (better: `.streamlit/secrets.toml`) |
| `INTERVIEW_AI_ALLOWED_EMAIL_DOMAINS` | — | optional allow-list, e.g. `cmrit.ac.in` |

**Sign-in needs an email account to send codes from.** Until it's configured the app shows setup instructions and lets nobody in. Gmail setup takes about 3 minutes — see [docs/AUTH.md](docs/AUTH.md).

## The experience

0. **Sign in:** enter your email; a 6-digit code is sent to that inbox. Malformed, reserved and disposable addresses are refused before anything is sent, and only the inbox owner can finish signing in. History is private to each account. See [docs/AUTH.md](docs/AUTH.md).
1. **Landing:** HUMAN NEXT hero, the three capabilities, the workflow, and a privacy notice.
2. **Configure:**
   - optional name or session label;
   - role (5) and difficulty (filters the questions);
   - question order, answer time, and camera on or off.
3. **System check:**
   - each model loads with its real load time;
   - the camera and microphone are tested, with a single-frame camera preview that isn't saved.
   - Continue unlocks only when everything required is ready.
   - If the camera fails, you can continue in audio-only mode.
4. **Consent:** exactly when recording happens and what is kept.
5. **Live interview console:**
   - **Left:** the question (without its rubric), controls, status chips (*Camera live*, *Recording* only while true), the countdown and timer, and live video with the face box.
   - **Right:** the run timeline; behavioral signals over the last ~10 s (measured); smoothed coaching hints; and facial expression, smoothed over 3 s with the latest raw prediction shown separately, labelled *model estimate · not scored*.
6. **Answer review:**
   - score ring and the weighted components with their contributions;
   - the rubric, each criterion with its reason;
   - strengths and recommendations;
   - tabs for transcript and concepts, speaking, camera, expression (model estimate) and technical timings.
7. **Results dashboard:**
   - average score, per-question scores, recurring strengths and recommendations;
   - question-by-question feedback and "how scores are calculated";
   - **download** as an HTML report (print to PDF) or CSV;
   - **save to history** (transcripts excluded unless you tick the box).
8. **History:** saved sessions open as the same dashboard. Delete one, or clear all.

## What works, and what doesn't

| Feature | Status |
|---|---|
| Landing, configuration, system check, consent, review, dashboard, history, export | ✅ Working (unit-, UI- and browser-tested) |
| Real model loading (MediaPipe, EfficientNet, Whisper, MiniLM, YAMNet) | ✅ Working in the real app |
| Transcription, rubric evaluation, scoring | ✅ Tested with real models on synthetic speech |
| Live video, live signals, expression smoothing | ✅ Implemented and tested with simulated devices; ⚠️ **needs a manual check with a real webcam** ([docs/TESTING.md](docs/TESTING.md)) |
| Audio/video synchronisation | ⚠️ Both start from one signal and the offset is measured; not frame-exact |
| Pause/resume | ❌ Not supported: each answer is one continuous recording (use Stop and Retry) |
| Typed (non-spoken) answers, LLM evaluation, reference answers | ❌ Not implemented; see the roadmap |
| Email-code sign-in, fake-email rejection, per-user history | ✅ Working (tested with a local SMTP server); ⚠️ delivery through a real provider needs your SMTP credentials |
| Deployment (Docker, cloud, public multi-user hosting) | ❌ Not supported: local only |

## Scoring in brief

- **Practice score per answer:** answer content 50%, delivery 20%, looking at camera 15%, facing camera 15%.
- **Missing measurements** are never treated as 0 or as perfect. They are left out, the weights are rescaled, and the score is labelled **partial**.
- **No transcript** means the score is **withheld**.
- **Answer content** is the rubric in `data/rubric.json`.
- **Not scored:** expression and background sound.

Full specification and worked examples: [docs/SCORING.md](docs/SCORING.md).

## Architecture

```
frontend/   app.py (pages + live console), theme.py (palette + CSS), components.py (escaped HTML pieces),
            ui_components.py (review + dashboard)
backend/    core/ (results, state machines, session, model registry)   services/interview_service.py
            vision/ (capture threads, landmarks, gaze/head, expression, smoothing, guidance)
            audio/ (recorder, Whisper analysis, delivery metrics)   ai/ (question bank, rubric, evaluator)
            scoring/ (practice score)   storage/ (records, SQLite history, CSV/HTML export)
data/       questions.json (26 questions, 5 roles), rubric.json
```

Design notes:
- **Capture:** the camera capture thread never runs a model. A separate analysis thread analyses only the latest frame: gaze and head about 3 times a second, expression about once a second.
- **UI updates:** the UI polls a thread-safe status object and sends the browser only what changed.
- **Models:** loaded once per process.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest                 # fast unit + headless UI tests (no hardware)
.\.venv\Scripts\python.exe -m pytest -m integration  # real models
.\.venv\Scripts\python.exe scripts\benchmark.py      # timings
```

## Privacy

- Audio and video stay in memory and are discarded after analysis.
- History stores scores and feedback, not recordings.
- Nothing is uploaded.

See [docs/PRIVACY.md](docs/PRIVACY.md), including a known issue with `sample.wav` in the public Git history.

## Presenting it

See [docs/DEMO.md](docs/DEMO.md) for a judging-panel walkthrough.

Engineering history: [docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md).
