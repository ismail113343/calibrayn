# 0009: Evaluation design

Date: 2026-11-06
Status: Accepted

## Context

Section 3.6 of the proposal describes three checks: a 100 question dataset to measure retrieval
accuracy, simulated learners such as a crammer and an overconfident student, and an informal test
with one real student. The proposal also asks whether using confidence in the schedule helps.

## Decision

Search check
- Questions are written from the student's own notes by the language model, spread over the
  difficulty levels 2 to 5. Each question keeps the topic it came from.
- Search is scored by hit rate in the top 1, 3 and 5 results and by mean reciprocal rank.
  The depth searched covers the whole library, so the rank of the source topic is exact.
- Duplicate questions are dropped, so fewer than the requested number can be made from small
  notes. The tool says so.

Learner simulation
- Four learners: a crammer who studies once a week, a steady learner who studies daily, an
  overconfident learner who studies every other day and feels surer than they should, and an
  under-confident learner who doubts good memory.
- Memory is a half-life model. Recall halves every half-life. A correct review multiplies the
  half-life by one plus the gain times the chance of forgetting. A failed review cuts it.
  All learners start with a half-life of 4 days and a gain of 6.
- Confidence is the true recall chance plus the learner's bias plus noise. Above 0.75 is sure,
  above 0.4 is unsure, and below that is guessing.
- Two schedules are compared on the same learners: plain SM-2, and SM-2 with the confidence
  adjustment and recall checks. Both use the real scheduler and metacognition code.
- Each comparison repeats with different random seeds. Differences are reported with a 95 percent
  bootstrap interval.
- Measures: retention (mean recall chance over all topics at the end), share of topics known at
  80 percent or more, number of reviews, topics known per 100 reviews, confident errors, recall
  checks and mean easiness.
- A sensitivity run varies one setting at a time: the confident error penalty, the extra penalty,
  the overconfidence threshold and the recall delay.

## Why

- The simulation uses the real scheduling code, so it tests the code that ships.
- Confidence intervals and several repeats show whether a difference is more than noise.

## To check

- The memory model and its numbers are assumptions. They are not fitted to real learners. The
  report must say so, and the sensitivity runs should vary the half-life and gain too.
- With the default numbers the adjusted schedule lowers easiness clearly but changes retention
  only a little. That is a result to report honestly, not to hide.
- Questions written by a model may use words from the notes, which makes search look easier than
  it is. Read a sample of 20 questions by hand.
- The offline model writes template questions. A dataset for the report needs a real model.
