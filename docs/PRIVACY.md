# Recording privacy and retention

## What is recorded, and when

- The camera and microphone are opened:
  - on the readiness page, for a few seconds to check they work (nothing is kept);
  - for each answer, from *Start recording* until processing begins.
- Audio and analysed video are kept **only while the red "Recording" indicator is shown**. Nothing is kept during the countdown.
- Everything runs locally. Nothing is uploaded.
  - Network access is limited to model hubs. On every start the sentence-model library (Hugging Face) checks huggingface.co for model updates; no audio, video, transcripts or results are sent. To stay fully offline after the first download, set `HF_HUB_OFFLINE=1`.
  - The first run downloads model files (Whisper, YAMNet, MiniLM) from their public sources.
  - Streamlit usage statistics are disabled in `.streamlit/config.toml`.
- The Streamlit server listens on `localhost` only.

## What is stored

| Data | Where | How long |
|---|---|---|
| Audio of an answer | memory | until analysis finishes, then discarded |
| Video frames | memory (latest frame only) | discarded continuously; never written to disk |
| Per-frame face labels (gaze, head, expression) | memory | until the result is built |
| Transcript and scores | browser session state | until you start a new interview or close the app |
| Diagnostic log `logs/app.log` | disk (git-ignored) | rotated at 1 MB, 3 files kept; contains timings, states and errors, **never** transcripts, audio or video |
| Audio copies (only if `INTERVIEW_AI_KEEP_RECORDINGS=true`) | `recordings/` (git-ignored) | until you delete them |

The original code wrote every answer to `sample.wav` in the project folder, which is tracked by Git. The new code writes no recording unless you turn on `INTERVIEW_AI_KEEP_RECORDINGS`.

## Known privacy issue in the repository history (not changed)

`sample.wav` is tracked in Git, appears in 3 commits, and the original repository is **public**. Its contents were not opened during this work.

Remediation is a separate decision for the repository owner. Options, each needing explicit approval:
1. Make the original repository private.
2. Remove the file from future commits (`git rm --cached sample.wav`) and add it to `.gitignore`. This does **not** remove it from history.
3. Rewrite history with `git filter-repo` to remove it everywhere. This changes every commit ID, so every collaborator has to re-clone.

## Use of results

Gaze, head-pose and expression estimates are imprecise coaching signals. They must not be interpreted as evidence of dishonesty, nervousness, competence or suitability for a job.
