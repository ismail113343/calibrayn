from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from calibrayn.orchestrator.metacognition import Rating, assess_calibration
from calibrayn.scheduler.sm2 import is_due


@dataclass(frozen=True)
class Summary:
    documents: int
    topics: int
    started: int
    due_now: int
    attempts: int
    accuracy: float | None


@dataclass(frozen=True)
class DayPoint:
    day: date
    attempts: int
    accuracy: float


@dataclass(frozen=True)
class TrendPoint:
    number: int
    score: float
    bias: float


@dataclass(frozen=True)
class WeakTopic:
    chunk_id: int
    title: str
    document: str
    attempts: int
    accuracy: float


class Analytics:
    def __init__(self, library, progress, tuning):
        self.library = library
        self.progress = progress
        self.tuning = tuning

    def summary(self, now):
        chunks = self.library.chunks()
        schedules = self.progress.schedules()
        attempts = self.progress.attempts()
        due = sum(
            1 for chunk in chunks if chunk.id not in schedules or is_due(schedules[chunk.id], now)
        )
        accuracy = None
        if attempts:
            accuracy = sum(attempt.score for attempt in attempts) / len(attempts)
        return Summary(
            documents=len(self.library.documents()),
            topics=len(chunks),
            started=len(schedules),
            due_now=due,
            attempts=len(attempts),
            accuracy=accuracy,
        )

    def daily(self):
        grouped = defaultdict(list)
        for attempt in self.progress.attempts():
            grouped[attempt.asked_at.date()].append(attempt.score)
        return [
            DayPoint(day, len(scores), sum(scores) / len(scores))
            for day, scores in sorted(grouped.items())
        ]

    def calibration_trend(self):
        ratings = [Rating(a.confidence, a.correct) for a in self.progress.attempts()]
        points = []
        for number in range(1, len(ratings) + 1):
            reading = assess_calibration(ratings[:number], self.tuning.meta)
            if reading.score is not None:
                points.append(TrendPoint(number, reading.score, reading.bias))
        return points

    def level_history(self):
        return [
            (number, attempt.level) for number, attempt in enumerate(self.progress.attempts(), 1)
        ]

    def weakest(self, limit=5):
        chunks = {chunk.id: chunk for chunk in self.library.chunks()}
        names = {document.id: document.name for document in self.library.documents()}
        grouped = defaultdict(list)
        for attempt in self.progress.attempts():
            grouped[attempt.chunk_id].append(attempt.score)
        topics = [
            WeakTopic(
                chunk_id=chunk_id,
                title=chunks[chunk_id].title,
                document=names.get(chunks[chunk_id].document_id, ""),
                attempts=len(scores),
                accuracy=sum(scores) / len(scores),
            )
            for chunk_id, scores in grouped.items()
            if chunk_id in chunks
        ]
        topics.sort(key=lambda topic: (topic.accuracy, -topic.attempts))
        return topics[:limit]
