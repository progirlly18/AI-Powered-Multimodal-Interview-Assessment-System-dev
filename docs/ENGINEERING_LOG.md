# Engineering Log — `feature/reliable-interview-system`

Concise record of work, decisions, failures and open issues. No secrets,
transcripts or recordings are recorded here.

## 2026-10-19

### Safety baseline
- Dev clone `D:\DOWNLOADS\interview-ai-dev`, branched from `main` @ `18b8cb3`.
- Remotes: `origin` = private dev repo; `upstream` (original, public) and
  `teammate` (Balaji's fork) have push URL `DISABLED`.
- Original folder `D:\DOWNLOADS\interview-ai` untouched (4 uncommitted files
  still present there).
- WIP backup `D:\DOWNLOADS\interview-ai-wip-backup-2026-09-27` verified by
  SHA-256 (`25C29761…`, `8D70A762…`, `22D95D7B…`) and restored onto the
  feature branch as the starting point. `sample.wav` not copied, opened or read.
- Nothing committed or pushed (requires owner approval).

### Environment findings
- FFmpeg is not installed. `main` transcribes via `whisper.transcribe("sample.wav")`,
  which shells out to FFmpeg, so transcription on `main` cannot work on this machine.
- The working venv contains both `opencv-python 5.0.0.93` (needed by cvzone) and
  `opencv-contrib-python 4.10.0.84` (needed by mediapipe); both provide `cv2`.
- Cold import of the backend on `main`: 81–133 s (measured, two runs).
- Emotion model: project docs report ~51% validation accuracy on FER2013 (7 classes).

- The new dev venv (built from `pip freeze` of the working venv) loaded `cv2 4.10.0`;
  the original venv loads `cv2 5.0.0`. Which OpenCV wins depends on install order.
- **setuptools must be < 81.** `pip freeze` omits setuptools, so the new venv got
  84.0.0, which no longer ships `pkg_resources`; `tensorflow_hub` then fails to
  import and YAMNet cannot load. Pinned `setuptools==80.9.0` (the version in the
  working venv). This confirms the teammate's `setuptools<=80.10.2` note.

### Implementation (uncommitted, on the feature branch)
- `backend/core/`: `Metric`/result types with explicit availability, run-state and
  session state machines, lazy timed `ModelRegistry`.
- Scoring rewritten to a documented spec (`docs/SCORING.md`): answer content now
  counts, the speech cap at 20 is gone, missing data is excluded and labelled
  (partial / withheld), emotion and background are informational only.
- Speech: Whisper gets in-memory float32 audio (no FFmpeg, no `sample.wav`),
  pace from speech segments, only hesitation sounds are penalised, silence and
  transcription failures are explained, YAMNet labels parsed as CSV.
- Vision: one landmark pass per frame (was three), MediaPipe model loaded by
  absolute path (cvzone used the working directory and downloaded the file),
  capture thread + analysis thread with latest-frame strategy, release on every path.
- Service: devices verified before a countdown, samples/frames kept only after
  it, measured A/V start offset, one run per process, cancellation.
- Question bank moved verbatim to `data/questions.json` with IDs and validation;
  no reference answers invented; candidates only see the question text.
- UI rebuilt on native Streamlit components: welcome, setup, readiness (real
  model/device checks), consent, live recording view, review, summary.
- Streamlit config: localhost only, XSRF/CORS on, usage stats off.

### Test results so far
- Unit + UI: 204 + 5 passing (fakes only).
- Two of my own test expectations were wrong and were corrected (165 wpm is within
  the 90-170 range; a 1.3 s recording correctly has no pace). Two real defects were
  found by tests and fixed: `is_complete` ignored partial scores; A/V offsets read
  0.0 ms because `time.monotonic()` ticks every ~16 ms on Windows (now `perf_counter`).
- Integration run 1: YAMNet failed to load (setuptools issue above).

- Integration run 2 (after the setuptools pin): 5/5 passed with real models.

### Measurements (same machine, same dev venv; `scripts/benchmark.py`)
| What | `main` @ 18b8cb3 | This branch |
|---|---|---|
| Wait before the first page renders | 126-140 s (backend imported at page load) | ~1.6 s (lazy backend import); page shown in seconds in the real app |
| Model loading | hidden inside the first page load | explicit step: 114 s in the real UI (face 26.7, emotion 6.0, Whisper 18.8, MiniLM 35.8, YAMNet 26.5) |
| Face landmark work per analysed frame (blank frame; lower bound) | 24.6 ms (3 FaceMesh passes) | 7.9 ms (1 pass) |
| Expression model per analysed frame | ~310 ms, **inside the capture loop** (every 10th frame) | ~310 ms in a separate analysis thread |
| Transcription of 10.5 s synthetic speech | **fails**: FileNotFoundError (Whisper needs FFmpeg for files) | 7.0 s in memory, WER 0.00 |
| YAMNet on 10.5 s | n/a | 97 ms |
| Answer evaluation | n/a | 148 ms |
| Memory after all models | not measured | 1.47 GB RSS |

- Evidence for the lag on `main`: every 10th frame the capture loop blocked for ~310 ms
  of expression inference plus three landmark passes, so the live view froze about
  once per third of a second; transcription could not run at all here.
- Hypothesis rejected by measurement: calling the Keras model directly
  (`model(x)`) was ~1010 ms vs ~310 ms for `model.predict()`; the change was reverted.
- Not measured (needs a person at the webcam): live frame rate, end-to-end latency with a
  real face, real A/V offset, CPU load during recording.

### Real-app check (browser pane, localhost only)
- Server bound to 127.0.0.1. Welcome, setup and readiness pages render; the first question is
  selected by default; all five real models load through the UI with real timings; Continue
  stays disabled until devices are checked.
- Device checks and recording were **not** run: they would turn on the owner's camera and
  microphone while unattended. Manual steps are in docs/TESTING.md.

### Decisions
- Do not merge the teammate branch or copy it wholesale. Selectively reimplemented:
  lazy model loading, question selection, in-memory Whisper input.
  Rejected: fabricated fallback scores, hard-coded device badges, rubric shown
  before answering, disabled CORS/XSRF, removal of the countdown.
- The teammate's custom HTML/CSS theme is not reused: its `.gitignore` mentions
  "reverse-engineering artifacts", so its provenance and licence are unclear.
  Credit: the lazy model loading, question selection, in-memory Whisper input and
  the setuptools pin follow ideas from Balaji's branch.
- Emotion and background sound removed from the score (see docs/SCORING.md); scoring
  weights are provisional and documented, not fitted to data.
- Legacy helper scripts kept, but no longer run on import or overwrite `sample.wav`.

### Open issues / next steps
1. Manual hardware verification (docs/TESTING.md) by the owner.
2. Decide on `sample.wav` in the public repository history (docs/PRIVACY.md).
3. Collect reference answers and a small human-rated answer set to calibrate scoring.
4. Gaze/head thresholds are pixel-based and unvalidated.
5. Lint and type checking are not configured in this project.
6. `opencv-python` and `opencv-contrib-python` both installed (dependency conflict
   inherited from cvzone/mediapipe); cvzone is no longer used at runtime and could be
   removed after testing.
7. Nothing committed: awaiting owner approval.
