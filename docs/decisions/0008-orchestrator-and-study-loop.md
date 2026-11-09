# 0008: Orchestrator and study loop

Date: 2026-11-03
Status: Accepted

## Context

Section 3.5 of the proposal describes a loop in which the monitors update after each answer and
the orchestrator decides the next topic, difficulty and kind of help.

## Decision

- Before each question the plan is made from the ZPD zone, the load level, the metacognition
  recommendation and any due recall. The plan keeps a list of plain reasons, shown to the student
  under "Why this question?".
- High load or a too-hard zone brings a worked example first.
- Under-confidence raises the level and removes help, unless the load is high. Heavy load wins.
- A due recall is asked with no help.
- A break is suggested when the load is high and the session signal is at least 0.5.
- Topic order: due recalls, topics being relearned, the most overdue topics, new topics in
  reading order, then the soonest upcoming topic.
- A score of 0.6 or more counts as correct. Scores map to SM-2 grades: 0.9 gives 5, 0.75 gives 4,
  0.6 gives 3, 0.4 gives 2, 0.2 gives 1. A guessed answer is capped at grade 3.
- When the student is overconfident and answers a sure question wrongly, that topic is queued for
  a recall after 30 minutes.
- After each answer the ZPD tracker may move the level. A move clears its history.
- A topic that is not yet due can still be asked when nothing else is due. This is extra practice.
  A correct answer is recorded but leaves the schedule unchanged, because counting it would
  stretch the intervals after a single sitting. A wrong answer counts as a failure as usual. The
  plan says when a question is extra practice. This rule was added after a demonstration session
  produced an interval of more than three years.

## Why

- A visible list of reasons makes the agent explainable, as the proposal asks.

## To check

- The score thresholds, the 30 minute delay and the 0.5 break signal are assumptions.
- The confidence-adjusted schedule must be compared with plain SM-2 in the persona simulations.
