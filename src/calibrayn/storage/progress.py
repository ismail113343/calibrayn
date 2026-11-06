from datetime import datetime

from calibrayn.models.calibration import Confidence
from calibrayn.models.learning import Attempt
from calibrayn.scheduler.sm2 import Schedule, new_schedule


def _when(text):
    return datetime.fromisoformat(text) if text else None


def _stamp(moment):
    return moment.isoformat() if moment else None


def _schedule(row):
    return Schedule(
        easiness=row["easiness"],
        interval_days=row["interval_days"],
        repetitions=row["repetitions"],
        due_at=_when(row["due_at"]),
        last_reviewed_at=_when(row["last_reviewed_at"]),
    )


def _attempt(row):
    return Attempt(
        id=row["id"],
        chunk_id=row["chunk_id"],
        asked_at=datetime.fromisoformat(row["asked_at"]),
        level=row["level"],
        confidence=Confidence(row["confidence"]),
        score=row["score"],
        correct=bool(row["correct"]),
        response_seconds=row["response_seconds"],
        delayed_recall=bool(row["delayed_recall"]),
    )


class Progress:
    def __init__(self, database):
        self.database = database

    def schedule_for(self, chunk_id):
        rows = self.database.read("SELECT * FROM schedules WHERE chunk_id = ?", (chunk_id,))
        return _schedule(rows[0]) if rows else new_schedule()

    def has_schedule(self, chunk_id):
        rows = self.database.read("SELECT 1 FROM schedules WHERE chunk_id = ?", (chunk_id,))
        return bool(rows)

    def schedules(self):
        rows = self.database.read("SELECT * FROM schedules")
        return {row["chunk_id"]: _schedule(row) for row in rows}

    def save_schedule(self, chunk_id, schedule):
        self.database.write(
            "INSERT INTO schedules"
            " (chunk_id, easiness, interval_days, repetitions, due_at, last_reviewed_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(chunk_id) DO UPDATE SET easiness = excluded.easiness,"
            " interval_days = excluded.interval_days, repetitions = excluded.repetitions,"
            " due_at = excluded.due_at, last_reviewed_at = excluded.last_reviewed_at",
            (
                chunk_id,
                schedule.easiness,
                schedule.interval_days,
                schedule.repetitions,
                _stamp(schedule.due_at),
                _stamp(schedule.last_reviewed_at),
            ),
        )

    def record_attempt(self, attempt):
        cursor = self.database.write(
            "INSERT INTO attempts (chunk_id, asked_at, level, confidence, score, correct,"
            " response_seconds, delayed_recall) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                attempt.chunk_id,
                attempt.asked_at.isoformat(),
                attempt.level,
                attempt.confidence.value,
                attempt.score,
                int(attempt.correct),
                attempt.response_seconds,
                int(attempt.delayed_recall),
            ),
        )
        return Attempt(**{**attempt.__dict__, "id": cursor.lastrowid})

    def attempts(self, limit=None):
        if limit is None:
            rows = self.database.read("SELECT * FROM attempts ORDER BY id")
            return [_attempt(row) for row in rows]
        rows = self.database.read("SELECT * FROM attempts ORDER BY id DESC LIMIT ?", (limit,))
        return [_attempt(row) for row in reversed(rows)]

    def attempt_count(self):
        return self.database.read("SELECT COUNT(*) AS total FROM attempts")[0]["total"]

    def get_state(self, key, default=None):
        rows = self.database.read("SELECT value FROM state WHERE key = ?", (key,))
        return rows[0]["value"] if rows else default

    def set_state(self, key, value):
        self.database.write(
            "INSERT INTO state (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    def queue_recall(self, chunk_id, due_at):
        self.database.write(
            "INSERT INTO recalls (chunk_id, due_at) VALUES (?, ?)"
            " ON CONFLICT(chunk_id) DO UPDATE SET due_at = excluded.due_at",
            (chunk_id, due_at.isoformat()),
        )

    def due_recalls(self, now):
        rows = self.database.read(
            "SELECT chunk_id FROM recalls WHERE due_at <= ? ORDER BY due_at", (now.isoformat(),)
        )
        return [row["chunk_id"] for row in rows]

    def pending_recalls(self):
        rows = self.database.read("SELECT chunk_id FROM recalls ORDER BY due_at")
        return [row["chunk_id"] for row in rows]

    def clear_recall(self, chunk_id):
        self.database.write("DELETE FROM recalls WHERE chunk_id = ?", (chunk_id,))
