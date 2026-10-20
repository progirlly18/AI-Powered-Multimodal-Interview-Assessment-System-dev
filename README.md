# AI Interview Practice (Multimodal Interview Assessment System)

Practise answering interview questions out loud and get transparent,
coaching-oriented feedback on:

- **Answer content:** topic relevance and coverage of the expected concepts (Sentence-Transformers `all-MiniLM-L6-v2`);
- **Speaking delivery:** pace and hesitation fillers (OpenAI Whisper);
- **On-camera presence:** face in view, looking at the camera, facing the camera (MediaPipe face landmarks).

> **This is a practice tool, not a hiring assessment.** Scores are heuristic,
> have not been validated against human interviewers, and must not be used to
> make or justify hiring decisions. Gaze and expression estimates are imprecise
> and say nothing about honesty or ability.

## Team

Yashica G, Balaji S, Nidhi Mishra, Gagan K. Guide: Mr. Saurav Kumar Topo.

## Quick start (Windows, Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Then open http://localhost:8501.
- Full instructions and troubleshooting: [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md).
- Linux/WSL (untested): [docs/SETUP_LINUX.md](docs/SETUP_LINUX.md).
- FFmpeg is **not** required. No API keys are needed.

## How an interview works

1. Choose a role, the questions (in order) and the answer time. The camera is optional.
2. **Readiness check:**
   - each AI model is loaded and its real load time is shown;
   - the camera and microphone are opened briefly to check they work.
   - Continue is only enabled when everything required is ready.
   - If the camera fails, you can retry or continue in a labelled audio-only mode.
3. **Consent:** an explanation of exactly when recording happens and what is kept.
4. **For each question:**
   - read the question (the grading rubric is not shown);
   - press *Start recording*: a countdown runs first, then a red **Recording** indicator with a timer;
   - live video with smoothed positioning hints;
   - processing (devices off) → results.
5. **Results:**
   - the practice score, or "Withheld" with the reason;
   - each component with its weight and contribution;
   - which measurements were unavailable and why;
   - the transcript and suggestions;
   - Retry or move on.
6. **Summary** across questions.

## Scoring in one paragraph

- **Weights:** answer content 50%, delivery 20%, looking at camera 15%, facing camera 15%.
- **Missing measurements** are never treated as 0 or as perfect. They are left out, the remaining weights are rescaled, and the score is labelled **partial**.
- **No answer content** (e.g. no words recognised) means the overall score is **withheld**.
- **Not scored:** facial-expression and background-sound labels.
- Full specification with worked examples: [docs/SCORING.md](docs/SCORING.md).

## Project layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module map, the state machines and the threading model.

| Path | Contents |
|---|---|
| `frontend/app.py` | Streamlit app |
| `backend/services/interview_service.py` | one recorded answer, start to finish |
| `backend/core/` | result types, state machines, session flow, model registry |
| `backend/vision/`, `backend/audio/`, `backend/ai/`, `backend/scoring/` | capture, analysis and scoring |
| `data/questions.json` | question bank (26 questions, 5 roles); validated on load |
| `tests/` | unit, UI and integration tests ([docs/TESTING.md](docs/TESTING.md)) |
| `scripts/benchmark.py` | timing measurements |
| `backend/training/` | emotion-model training code (unchanged) |

## Editing the question bank

Questions live in `data/questions.json`. Each entry needs:
- a unique `id`, `role`, `category` (`technical` or `behavioural`), `difficulty` (`Easy`, `Medium` or `Hard`), and `question`;
- `concepts`: each with a `name`, `aliases` (paraphrases used for semantic matching) and optional `evidence` phrases (exact whole-word matches);
- `reference_answer`: optional (`null` for every question at the moment).

The file is validated at start-up; a malformed entry produces a clear error naming the question.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest                 # fast unit + UI tests (no hardware)
.\.venv\Scripts\python.exe -m pytest -m integration  # real models
```

## Privacy

- Recordings stay in memory and are discarded after analysis.
- Nothing is uploaded, and the server only listens on localhost.
- Details, retention, and a known issue with `sample.wav` in the Git history: [docs/PRIVACY.md](docs/PRIVACY.md).

## Known limitations

- **Not validated.** The scores have never been compared with human ratings. Topic relevance is uncalibrated cosine similarity. There are no reference answers yet, so correctness and reasoning are not assessed.
- **Gaze and head pose** use fixed pixel and ratio thresholds and do not account for distance from the camera or mirrored images.
- **Expression model:** about 51% validation accuracy on FER2013; shown for information only.
- **Fillers:** Whisper often drops "um" and "uh", so filler counts are a lower bound.
- **Audio/video sync:** both start from the same signal and the measured offset is reported; frame-exact synchronisation is not guaranteed.
- **Cold start:** the first model load takes minutes. The page itself opens immediately.
- **One recording at a time** per running app.
- **Hardware testing:** this release was tested with fake devices and real models. Live camera and microphone behaviour must still be checked by hand ([docs/TESTING.md](docs/TESTING.md#manual-hardware-verification)).
- **Deployment:** only local use on Windows has been tested. Docker, cloud hosting and multi-user use are **not** supported.

Engineering history and decisions: [docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md).
