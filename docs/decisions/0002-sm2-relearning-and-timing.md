# 0002: SM-2 relearning in minutes and day-based intervals

Date: 2026-10-26
Status: Accepted

## Context

Section 2.4 of the proposal says that items answered incorrectly are scheduled to reappear
within minutes. The first SM-2 implementation followed the original algorithm and sent a failed
topic back after one day, which does not meet that requirement. The proposal also calls the
quantity the "Easiness Factor", while the code and the config file used different names for it.

## Decision

- A failing grade (0 to 2) restarts the repetition count and makes the topic due again after
  10 minutes. The easiness factor is left unchanged, as in the original algorithm.
- The first successful review after a failure schedules the topic one day ahead, then six days,
  then interval times easiness, exactly as for a new topic.
- Intervals after a pass are counted in calendar days. A topic falls due at the start of its
  due day and stays due for the rest of that day.
- Schedules store timestamps (`due_at`, `last_reviewed_at`) instead of dates, and functions take
  a `now` argument so tests do not depend on the clock.
- The easiness factor is called `easiness` throughout the code, and the yaml keys match the
  field names of the config classes.
- A test builds the config objects from `configs/settings.yaml` and compares them with the
  defaults, so the two cannot drift apart.

## Why

- It meets the proposal wording and keeps a failed topic inside the same study session.
- Counting passes in calendar days stops a topic from being "not due" because the student
  studied earlier in the day than last time.
- One vocabulary across the proposal, the code and the config makes the report easier to follow.

## Alternatives considered

- Keep dates only and handle short repeats in the session queue. A schedule that cannot express
  "in ten minutes" would lose that information as soon as the session ended.
- A list of learning steps, such as 1 minute then 10 minutes then 1 day. More realistic, but more
  to explain and to test than the proposal asks for.

## To check

- Ten minutes is an assumption, because the proposal only says "within minutes". The evaluation
  can compare other values using the persona simulations from section 3.6.
- Interval timing uses the timezone of the `now` value passed in. The storage layer must save
  and restore timestamps consistently, all naive or all timezone-aware.