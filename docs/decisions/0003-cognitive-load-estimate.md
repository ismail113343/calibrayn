# 0003: Cognitive load estimate

Date: 2026-11-03
Status: Accepted

## Context

Section 3.2 of the proposal describes a Cognitive Load Estimator that produces a simple numerical
estimate of how demanding the material is, based on recent accuracy, response time and time in
the session. The proposal gives no formula, so the form below is a design decision.

## Decision

- The estimate runs from 0 (no sign of load) to 1 (heavy load), with a level: low, moderate, high
  or unknown.
- Accuracy signal: one minus the mean score of the last 10 answers.
- Response time signal: how much slower the median of the last 10 answers is than the student's
  own earlier median. The same pace gives 0 and twice as slow gives 1. Each time is capped at 300
  seconds. It needs 5 earlier answers. Until then the signal is left out and the weights are
  rescaled.
- Session signal: 0 up to 20 minutes, rising in a straight line to 1 at 60 minutes.
- Weights are 0.5, 0.3 and 0.2. Low is below 0.35 and high is above 0.65.
- No estimate is given before 5 answers.
- The result carries its three signals so a decision can be explained.

## Why

- These are the three signals the proposal names, and the app already sees all of them.
- A personal baseline stops a slow reader from looking overloaded.
- Medians and a cap stop one interruption from looking like load.

## Alternatives considered

- Asking for a mental effort rating after every answer is more direct but interrupts the student.
- A trained model would need labelled load data that this project does not have.

## To check

- This is a behavioural proxy, not a validated measure of cognitive load.
- The weights, thresholds and the 20 to 60 minute range are assumptions to test with the persona
  simulations.
- A single mental effort rating (Paas, 1992) in the student test can be compared with the
  estimate. Confirm the citation before using it.
