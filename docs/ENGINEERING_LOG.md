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

## 2026-10-23 — HUMAN NEXT redesign (`feature/human-next-ui`, from `1bdd3d2`)

### What changed
- **Visual identity:** `frontend/theme.py` (owner's palette, glass cards, gradient hero,
  status chips, reduced-motion support, no external fonts) and `frontend/components.py`
  (escaped HTML pieces). Streamlit `[theme]` set to match. The reference image mentioned in
  the brief was not received; the design follows the written palette and direction.
- **Pages:** landing; 3-step setup (configure with difficulty filter and optional name,
  system check with one-frame camera preview, consent); two-column live console; answer
  review; results dashboard; history.
- **Answer rubric** (`data/rubric.json`, `backend/ai/rubric.py`, `text_features.py`):
  relevance, coverage, elaboration, structure, examples, each with an explanation,
  plus strengths and improvements. Replaces `0.4·relevance + 0.6·coverage`
  (documented change; weights provisional, not fitted).
- **Expression pipeline:** probabilities validated; model sampled at most once per
  `INTERVIEW_AI_EMOTION_INTERVAL` (default 1 s; it costs ~310 ms); 3 s smoothing kept
  separate from raw output; never scored.
- **Multiple faces:** landmarker detects up to 2; frames are counted, with a live warning and a result warning.
- **Persistence/export:** `SessionRecord` → SQLite history (`instance/history.db`), CSV,
  and an HTML report (print to PDF). Transcripts are excluded unless the user opts in. No new dependencies.

### Verification
- Fast suite: 265 passed (incl. 7 headless UI tests).
- Real app: landing, setup and system check rendered; all 5 real models loaded (face
  landmarker 50.9 s, emotion 4.6 s, Whisper 18.4 s, MiniLM 42.7 s, YAMNet 13.5 s on this run).
- Console, review, dashboard and history were inspected visually through a throwaway
  harness (scratch folder, not in the repo) that fed the app the test-suite fakes. Its
  numbers are synthetic; it was used only to check the layout. Found and fixed: unequal card
  heights, wrapped run timeline, controls below the fold, heading anchor icons,
  duplicated "missing concepts" recommendation.
- Not verified: real webcam and microphone (would turn devices on unattended). The owner's
  own app session was running on port 8501 with the system Python and was left untouched.

## 2026-10-23 — Email-code sign-in (`feature/human-next-ui`)

- Owner chose passwordless email codes (over Google OIDC or local-only checks) so that
  only inbox owners can sign in. Standard library only (`smtplib`, `secrets`, `hashlib`,
  `sqlite3`); no new dependencies.
- `backend/auth/`: address rules (format, reserved, disposable list, optional allow-list),
  salted-hash codes (10 min, single use, 5 tries), rate limits (60 s, 5/hour/address,
  10/browser session), SMTP mailer, `AuthService`, settings from env or secrets.toml.
- Gate in `frontend/app.py` before every page. The app **fails closed** if SMTP is not
  configured; `INTERVIEW_AI_REQUIRE_SIGN_IN=false` must be set explicitly to disable it.
- History is now per owner (migration adds an `owner` column; legacy rows are hidden).
- Tests: auth unit tests, a real SMTP conversation against a local test server, and
  headless UI tests of the gate.
- Not verified: delivery through a real provider (requires the owner's SMTP credentials).
