# 0004: Metacognition module and confidence-based easiness adjustment

Date: 2026-10-28
Status: Accepted

## Context

Section 3.3 of the proposal describes a Metacognition Module. Before each question the student
rates confidence as "sure", "unsure" or "guessing". The module compares the rating with the
result and keeps a calibration score over a rolling window. Persistent overconfidence triggers a
delayed, notes-free recall task. Persistent under-confidence raises difficulty and reduces
support. Confidently wrong answers lower the SM-2 easiness factor more sharply than simple
mistakes. Section 3.8 adds a fallback to correctness-only scheduling until enough ratings exist.
The proposal gives no formulas, so the form below is a design decision.

## Decision

- Each rating is stored with whether the answer was correct.
- Each confidence level has an expected accuracy: sure 1.0, unsure 0.5, guessing 0.0.
- The gap for one rating is the expected accuracy minus the result (1 for right, 0 for wrong).
- Bias is the mean gap over the last 15 ratings. A positive bias means overconfident.
- The calibration score is 1 minus the mean absolute gap. A score of 1 is perfect calibration.
- No verdict is given until 10 ratings exist. Until then scheduling uses correctness only.
- Bias above 0.30 is overconfident. Bias below -0.30 is under-confident. Anything between,
  including the edges, is calibrated.
- Overconfident leads to a delayed recall task. Under-confident leads to reduced support. The
  module only recommends. The orchestrator decides.
- The ease adjuster lowers the easiness factor by 0.25 when the student was sure and wrong, and
  by a further 0.10 while overconfidence persists. The floor is 1.3.
- Simple mistakes keep the SM-2 treatment. The count restarts and the topic returns in
  10 minutes with the easiness unchanged.
- The adjuster is separate from sm2.py, so plain SM-2 stays available as the baseline for
  research question 2.

## Why

- Bias shows the direction of the error, and the score shows its size. Both are needed because
  opposite errors can cancel in the bias.
- Three confidence levels are quick for students to give and easy to explain.
- Keeping the adjuster apart lets the evaluation compare plain SM-2 with the adjusted version.

## Alternatives considered

- Counting only "sure and wrong" answers misses under-confidence.
- A reliability diagram per confidence level needs far more ratings than a short prototype will
  collect.

## To check

- The expected accuracies, the 0.30 thresholds, the penalties and the window sizes are
  assumptions. The persona simulations should vary them one at a time.
- A student who says "sure" is not always right. A level of 0.9 for sure may fit better.
- The proposal says confidently wrong answers reduce the easiness factor more sharply than
  simple mistakes. Here simple mistakes leave it unchanged, so the difference holds, but this
  reading should be confirmed with the supervisor.