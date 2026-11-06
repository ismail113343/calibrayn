# 0005: SQLite storage and the workspace

Date: 2026-11-03
Status: Accepted

## Context

The app has to remember notes, schedules and every answer between sessions. It serves one student
at a time and runs on that student's computer.

## Decision

- One SQLite file holds documents, chunks with their vectors, schedules, attempts, queued recalls
  and a small state table.
- Each chunk is one topic. Its schedule is stored separately and only exists after the first
  review, so a missing schedule means a new topic.
- Every attempt is stored with its level, confidence, score and response time. The ZPD tracker,
  the load estimator and the metacognition tracker are rebuilt from these attempts when the app
  starts, so nothing has to be saved twice.
- Deleting a document removes its chunks, schedules, attempts and recalls in one step.
- Timestamps are stored as local time text.
- A Workspace object builds and holds all the parts. The interface only talks to it.

## Why

- A single file needs no server and is easy to back up or delete.
- Rebuilding trackers from attempts keeps one source of truth.

## To check

- A student who studies from two computers would need a different design.
- If the embedding model changes, vectors are rebuilt from the stored text on start.
