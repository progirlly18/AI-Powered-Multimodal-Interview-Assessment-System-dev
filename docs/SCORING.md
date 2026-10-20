# Scoring specification

The practice score is a **coaching aid**. It is heuristic, has **not been
validated** against human interviewers, and must not be used for hiring
decisions or to compare candidates.

Code: `backend/scoring/scoring_engine.py`. Tests:
`tests/unit/test_scoring_engine.py` (every rule and example below).

## 1. Principles

1. Every measurement is a `Metric` that is **available** (has a value),
   **unavailable** (its input was missing, e.g. no face in view) or
   **failed** (something broke, e.g. a model error). Unavailable and failed
   metrics never carry a number.
2. A missing measurement is **never** treated as 0, as a perfect result, or
   as a "normal" default.
3. All component scores use the same 0–100 scale. Values outside 0–100,
   NaN or infinity are rejected as programming errors.
4. Content, delivery and on-camera presence are kept separate. Semantic
   similarity is **not** evidence that an answer is correct.

## 2. Components

| Key | Component | Inputs (valid range) | Rule | Weight |
|---|---|---|---|---|
| `answer` | Answer content | topic relevance 0–100, concept coverage 0–100 | `0.4 × relevance + 0.6 × coverage` | **50** (required) |
| `delivery` | Speaking delivery | speaking pace (wpm), hesitation fillers (≥ 0) | `100 − 3 × fillers − (10 if pace < 90 or > 170 wpm)`, clamped to 0–100 | 20 |
| `eye_contact` | Looking at the camera | % of face frames with gaze "Looking Center" | percentage as is | 15 |
| `head_facing` | Facing the camera | % of face frames with head "Center" | percentage as is | 15 |

**Not scored (informational only):**
- **Facial-expression label.** The EfficientNet model reports about 51% validation accuracy on FER2013 (posed lab images), and "smiling" is not a validated sign of a good answer.
- **Background-sound label (YAMNet).** The original rule rewarded rain or birdsong and penalised a quiet room.

### 2.1 Answer content inputs

- **Topic relevance** = `clamp(cosine(question, answer), 0, 1) × 100`, using `all-MiniLM-L6-v2` embeddings.
  - It measures how on-topic the answer is. It is **uncalibrated**: no human-rated data maps similarity to answer quality.
  - Typical relevant answers land around 40–70, not 100.
- **Concept coverage** = mean over the question's expected concepts of:
  - **100** if any evidence phrase occurs in the answer as whole words (case-insensitive; hyphens count as spaces, so "cross-validation" matches "cross validation", and "L1" does not match "L10");
  - otherwise **min(similarity × 100, 70)** if the best alias-to-sentence cosine similarity is ≥ 0.50;
  - otherwise **0**.
- **Similarity to a reference answer** is reported separately and is not part of the score.
  - It is only available for questions that have one; none do yet.
  - No reference answers were invented when the question bank was moved to `data/questions.json`.

### 2.2 Delivery inputs

- **Speaking pace** = words ÷ speaking time.
  - Speaking time is the sum of Whisper's speech segments, not the whole recording window.
  - If Whisper returns no segments, the recording length is used and the result is labelled an estimate.
  - Pace is unavailable with no words or < 3 s of speech.
- **Hesitation fillers** counted (whole words): `um, umm, uh, uhm, erm, er, ah, hmm, mm`.
  - Whisper often omits these, so counts are a lower bound.
- **Possible fillers** — `like, so, actually, basically, you know, kind of, sort of, i mean` — are shown as coaching notes but **not penalised**, because they are also ordinary words.

### 2.3 Camera inputs

- Faces are analysed about 3 times per second during recording.
- If fewer than 3 frames were analysed, or the face was visible in < 50% of analysed frames, both camera components are **unavailable** (not 0).
- Gaze and head thresholds are fixed heuristics (iris ratio 0.35–0.65; nose offset ±15 px horizontally, −10/+15 px vertically) and have **not been validated**. Treat them as coaching hints. They say nothing about honesty or ability.

## 3. Weights and their rationale

- **Answer content 50%:** the substance of the answer is the purpose of an interview question.
- **Delivery 20%:** pace and hesitation affect how an answer is received.
- **Camera presence 30%:** split evenly between gaze (15%) and head direction (15%).

These weights are **provisional design choices, not fitted to data**. They replace the original weights:
- **emotion 25** — removed (low-accuracy model, unvalidated proxy);
- **eye 25 / head 20**;
- **speech 20** — but capped so it was effectively constant;
- **background 10** — removed (inverted rule).

The original weights also left answer content out of the overall score entirely.

## 4. Unavailable components

```
if answer content is unavailable or failed:
    overall = withheld ("No overall score: …")
else:
    used    = components that are available
    overall = Σ(weight × value) / Σ(weight) over used components
    partial = any component is missing
```

- A partial score is always labelled **"(partial)"**. The explanation names each missing component and its reason, and states that the score is not directly comparable with a complete score.
- The session average is the mean over questions that have an overall score. The count is always shown (e.g. "2 of 3 questions scored").

## 5. Rounding

Values are kept as floats internally. They are displayed as whole numbers, rounding halves **up** (83.5 → 84), via `round_half_up`. Python's built-in `round` (83.5 → 84 but 82.5 → 82) is not used for display.

## 6. Worked examples

These are tested in `tests/unit/test_scoring_engine.py`.

**Example 1: complete.** Answer 80, delivery 90, gaze 70, head 100:
```
(50×80 + 20×90 + 15×70 + 15×100) / 100 = (4000 + 1800 + 1050 + 1500) / 100 = 83.5 → 84
```

**Example 2: audio only (camera off).** Answer 80, delivery 90, gaze and head unavailable:
```
(50×80 + 20×90) / (50 + 20) = 5800 / 70 = 82.86 → 83 (partial)
```

**Example 3: no transcript.** Answer unavailable ("No words were recognised"), delivery 90, gaze 100, head 100:
```
overall withheld
```

**Example 4: delivery.** 165 wpm with one "um":
```
100 − 3 = 97
```
At 80 wpm with no fillers:
```
100 − 10 = 90
```

**Example 5: answer content.** Relevance 50, coverage 75:
```
0.4×50 + 0.6×75 = 65
```

**Regression.** No camera and no microphone used to score **100/100** on the teammate branch, because its fallback values were the top value of every rule. Now the overall score is withheld.

## 7. What this score can and cannot tell you

- **Can:**
  - whether the answer mentioned the expected concepts;
  - whether it stayed on topic;
  - roughly how fast you spoke;
  - roughly how often you faced the camera.
- **Cannot:**
  - whether the answer is technically correct;
  - whether your reasoning is sound;
  - whether you would succeed in a job.

Better evaluation needs reference answers and a set of human-rated answers to calibrate against (see the roadmap in the README).
