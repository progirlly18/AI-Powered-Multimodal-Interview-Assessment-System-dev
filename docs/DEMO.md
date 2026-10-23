# Demonstrating HUMAN NEXT to judges

A 7–8 minute walkthrough that shows both the product and the engineering behind it.

## Before the session (10 minutes)

1. Start the app with the project's own Python, so you don't pick up a different install:

   ```powershell
   cd D:\DOWNLOADS\interview-ai-dev
   .\.venv\Scripts\python.exe -m streamlit run frontend/app.py
   ```

2. **Sign in first.** Sending the code needs internet and the SMTP account from docs/AUTH.md. A sign-in lasts 12 hours in that browser tab, so do it before the judges arrive. To show the feature live, sign out and sign in again in front of them. Also show that `someone@mailinator.com` and `test@example.com` are refused instantly.
3. Go through **Configure → System check** and press **Load models** *before* the judges arrive. The first load takes about 2 minutes; afterwards it is instant for that app session.
4. Press **Check devices** and confirm both are 🟢. Close Teams, Zoom and browser tabs that use the camera.
5. Pick a quiet spot with your face well lit. Use 30 s answers.
6. Optional: set `HF_HUB_OFFLINE=1` if the venue Wi-Fi is unreliable. The models are already cached.

## Walkthrough

| Time | Show | Say |
|---|---|---|
| 0:00 | **Landing page** | "HUMAN NEXT gives explainable feedback on spoken interview answers. Everything runs locally." |
| 0:45 | **Configure:** role, difficulty filter, 2 questions | "The question bank is data, not code: 26 questions in validated JSON, by role and difficulty." |
| 1:15 | **System check:** model load times, device status, camera preview | "Nothing claims to be ready until it has actually been checked. If the camera fails, it offers an honest audio-only mode." |
| 1:45 | **Consent** | "Recording happens only while the magenta indicator is on. Audio and video stay in memory." |
| 2:00 | **Live console:** Start recording, answer a question | Point at the countdown ("not recording yet"), then the **Recording** chip, the live signal bars and the *model estimate · not scored* expression card. Look away for 3 seconds to show the smoothed coaching hint. |
| 3:30 | **Answer review:** score ring, components, rubric | "Every number shows its weight and its reason: concepts found, words, connectives, examples. Measured, model-estimated and unavailable values are labelled differently." |
| 5:00 | **Second answer, deliberately short and vague** | Show the lower elaboration, structure and examples scores, and the concrete improvement tips. |
| 6:00 | **Results dashboard → Download report → Save to history → History** | "Exportable report; local SQLite history that stores scores, not recordings." |
| 7:00 | **Engineering slide** (optional) | Points below. |

## Engineering points worth making

- **Measured performance, not guesses** (scripts/benchmark.py):
  - The original first page blocked for **126–140 s** importing models; now it renders in about **1.6 s**.
  - Face landmark work dropped from 3 passes to 1 per frame (**24.6 → 7.9 ms**).
  - The ~310 ms expression model runs in a background thread about once a second, never in the capture loop.
- **Honest data:**
  - A missing camera or microphone never becomes a score.
  - The previous version gave **100/100 with no hardware**; a regression test now prevents that.
- **Explainable scoring:** a documented rubric (`data/rubric.json`, `docs/SCORING.md`) with hand-calculated examples checked by unit tests.
- **Responsible AI:** expression recognition is shown as an uncertain model estimate (~51% test accuracy on FER2013) and is never used to judge the candidate.
- **Quality:** 265 automated unit and UI tests, plus real-model integration tests.

## If something goes wrong

| Problem | Recovery |
|---|---|
| Camera unavailable | Use **Continue with audio only**; the score is labelled *partial*. |
| "No sound reached the microphone" | Check the Windows input device and retry the question. |
| A model fails to load | Press *Load models* again; details are in `logs/app.log`. |
| Want a clean slate | **New interview**, or **History → Clear all history**. |

## Questions judges may ask

- **"Is the score accurate?"** It is a transparent heuristic, not validated against human raters. The next step is to collect reference answers and a small human-rated set to calibrate it (roadmap in the README).
- **"Do you judge emotions?"** No. Expression estimates are shown for exploration only, with their limitations, and have zero weight.
- **"Is it real time?"** The video and live signals update about 3 times a second during recording. Transcription and content evaluation run right after each answer, which takes a few seconds on a laptop CPU.
