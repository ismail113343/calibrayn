"""Zone of Proximal Development tracking.

Implements the ZPD Tracker from sections 2.2 and 3.2 of the proposal: a rolling average of
recent answer accuracy that is used to keep question difficulty inside a productive band,
"stretch but don't snap". Too many right answers means the material is too easy. Too few means
it is too hard.

The core of it is two pure functions, assess() and suggest_level(). ZPDTracker wraps them with
the small amount of state the orchestrator needs to carry between questions.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class Zone(Enum):
    INSUFFICIENT_DATA = "insufficient_data"
    TOO_HARD = "too_hard"
    IN_ZONE = "in_zone"
    TOO_EASY = "too_easy"


@dataclass(frozen=True)
class ZPDConfig:
    """Settings for the tracker. See docs/decisions/0001-zpd-target-band.md for the reasoning."""

    window: int = 10
    target_low: float = 0.60
    target_high: float = 0.85
    min_attempts: int = 5
    min_level: int = 1
    max_level: int = 5
    start_level: int = 3

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if not 1 <= self.min_attempts <= self.window:
            raise ValueError("min_attempts must be between 1 and window")
        if not 0.0 <= self.target_low < self.target_high <= 1.0:
            raise ValueError("targets must satisfy 0 <= target_low < target_high <= 1")
        if self.min_level > self.max_level:
            raise ValueError("min_level cannot be above max_level")
        if not self.min_level <= self.start_level <= self.max_level:
            raise ValueError("start_level must sit between min_level and max_level")


DEFAULT_CONFIG = ZPDConfig()


@dataclass(frozen=True)
class ZPDStatus:
    zone: Zone
    accuracy: float | None
    attempts_used: int


def _check_score(score: float) -> None:
    if not isinstance(score, int | float):
        raise TypeError(f"score must be a number, got {type(score).__name__}")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be between 0 and 1, got {score}")


def _check_level(level: int, config: ZPDConfig) -> None:
    if not config.min_level <= level <= config.max_level:
        raise ValueError(
            f"level must be between {config.min_level} and {config.max_level}, got {level}"
        )


def assess(recent_scores: Sequence[float], config: ZPDConfig = DEFAULT_CONFIG) -> ZPDStatus:
    """Place the student relative to the zone.

    Scores run from 0.0 (wrong) to 1.0 (fully correct), oldest first, so partial credit from
    the grader is allowed. Only the latest `window` scores are used.
    """
    recent = list(recent_scores)[-config.window :]
    for score in recent:
        _check_score(score)

    if len(recent) < config.min_attempts:
        return ZPDStatus(Zone.INSUFFICIENT_DATA, None, len(recent))

    accuracy = sum(recent) / len(recent)
    if accuracy < config.target_low:
        zone = Zone.TOO_HARD
    elif accuracy > config.target_high:
        zone = Zone.TOO_EASY
    else:
        zone = Zone.IN_ZONE
    return ZPDStatus(zone, accuracy, len(recent))


def suggest_level(status: ZPDStatus, current_level: int, config: ZPDConfig = DEFAULT_CONFIG) -> int:
    """Difficulty to use next: one step up when too easy, one step down when too hard."""
    _check_level(current_level, config)
    if status.zone is Zone.TOO_EASY:
        return min(current_level + 1, config.max_level)
    if status.zone is Zone.TOO_HARD:
        return max(current_level - 1, config.min_level)
    return current_level


class ZPDTracker:
    """Holds the recent scores and the current difficulty level for one learner.

    Scores only mean something relative to the level they were earned at, so changing level
    clears the history. After a change the tracker needs `min_attempts` fresh scores before it
    will suggest another move, which also stops it from bouncing between levels.
    """

    def __init__(
        self,
        config: ZPDConfig = DEFAULT_CONFIG,
        level: int | None = None,
        scores: Sequence[float] = (),
    ) -> None:
        self._config = config
        self._level = config.start_level if level is None else level
        _check_level(self._level, config)
        self._scores: deque[float] = deque(maxlen=config.window)
        for score in scores:
            self.record(score)

    @property
    def level(self) -> int:
        return self._level

    @property
    def scores(self) -> tuple[float, ...]:
        """Current history, oldest first. Handy for saving the tracker and restoring it later."""
        return tuple(self._scores)

    def record(self, score: float) -> None:
        _check_score(score)
        self._scores.append(float(score))

    def status(self) -> ZPDStatus:
        return assess(tuple(self._scores), self._config)

    def suggested_level(self) -> int:
        return suggest_level(self.status(), self._level, self._config)

    def set_level(self, level: int) -> None:
        _check_level(level, self._config)
        if level != self._level:
            self._level = level
            self._scores.clear()

    def follow_suggestion(self) -> int:
        """Move to the suggested level, if it differs, and return the level now in force."""
        self.set_level(self.suggested_level())
        return self._level

    def __repr__(self) -> str:
        return f"ZPDTracker(level={self._level}, attempts={len(self._scores)})"
