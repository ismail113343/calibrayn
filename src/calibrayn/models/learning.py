from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from calibrayn.models.calibration import Confidence


class Support(Enum):
    WORKED_EXAMPLE = "worked_example"
    DIRECT_RECALL = "direct_recall"


@dataclass(frozen=True)
class Question:
    chunk_id: int
    text: str
    answer_key: str
    level: int
    support: Support
    worked_example: str
    delayed_recall: bool


@dataclass(frozen=True)
class Attempt:
    chunk_id: int
    asked_at: datetime
    level: int
    confidence: Confidence
    score: float
    correct: bool
    response_seconds: float
    delayed_recall: bool
    id: int | None = None
