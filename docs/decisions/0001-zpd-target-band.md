# 0001: ZPD target band and tracker design

Date: 2026-10-18
Status: Accepted

## Context

Sections 2.2 and 3.2 of the proposal ask for a ZPD Tracker that keeps "a rolling average of
recent answer accuracy" and uses it to hold question difficulty inside a productive range. The
proposal gives no numbers, so the values below are design decisions and have to be justified in
the report.

## Decision

- Accuracy is the mean of the last 10 answer scores earned at the current difficulty level.
  Scores run from 0 to 1, so partial credit from the grader counts.
- The productive band is 0.60 to 0.85 accuracy. Below it the material is too hard, above it
  too easy, and both edges count as inside the zone.
- No verdict is given until 5 scores exist at the current level.
- Difficulty moves one level at a time on a scale of 1 to 5, starting at level 3.
- Changing level clears the history, so the next verdict only reflects work done at the new
  level. This also stops the tracker from bouncing between levels.
- The tracker is per learner. Whether one tracker is kept per topic or one across all topics is
  left to the orchestrator.

## Why

- A rolling mean is what the proposal describes and is easy to explain to a student or an
  examiner.
- A window of 10 reacts within a single study session without swinging on one answer.
- Resetting on a level change keeps the evidence honest. Scores from an easier level would
  otherwise hide whether the student can cope with the harder one.

## Alternatives considered

- Knowledge tracing (proposal 2.4) is more precise but needs far more data per student than a
  short prototype will collect.
- An exponential moving average would weight recent answers more heavily. It is harder to
  explain, and the window already limits how far back the tracker looks.

## To check

- The band is an assumption. The evaluation should test its sensitivity, for example 0.55 to
  0.80 against 0.65 to 0.90, using the persona simulations from section 3.6.
- A possible source for the upper end is Wilson et al. (2019), "The Eighty Five Percent Rule
  for optimal learning". Confirm the citation before using it in the report.