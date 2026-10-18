"""SM-2 spaced repetition scheduling.

Follows the algorithm described by P. A. Wozniak (1990). The module is pure: it takes a
schedule and a grade and returns a new schedule, with no reads or writes to disk. That keeps
it easy to test and easy to call from the orchestrator.

Grades:
    5  perfect response
    4  correct after some hesitation
    3  correct, but recalled with serious difficulty
    2  incorrect, but the answer felt familiar once seen
    1  incorrect, and the answer was remembered once seen
    0  complete blackout
"""

import math
from dataclasses import dataclass, replace
from datetime import date, timedelta

MIN_GRADE = 0
MAX_GRADE = 5


@dataclass(frozen=True)
class SM2Config:
    """Tunable constants. The defaults are the values from the original algorithm."""

    initial_ease: float = 2.5
    min_ease: float = 1.3
    first_interval_days: int = 1
    second_interval_days: int = 6
    pass_grade: int = 3
    repeat_below_grade: int = 4
    max_interval_days: int | None = None

    def __post_init__(self) -> None:
        if self.min_ease <= 0 or self.min_ease > self.initial_ease:
            raise ValueError("min_ease must be positive and no higher than initial_ease")
        if self.first_interval_days < 1:
            raise ValueError("first_interval_days must be at least 1")
        if self.second_interval_days < self.first_interval_days:
            raise ValueError("second_interval_days cannot be shorter than the first interval")
        if not MIN_GRADE < self.pass_grade <= MAX_GRADE:
            raise ValueError("pass_grade must be between 1 and 5")
        if self.repeat_below_grade < self.pass_grade:
            raise ValueError("repeat_below_grade cannot be lower than pass_grade")
        cap = self.max_interval_days
        if cap is not None and cap < self.second_interval_days:
            raise ValueError("max_interval_days cannot be shorter than the second interval")


DEFAULT_CONFIG = SM2Config()


@dataclass(frozen=True)
class Schedule:
    """Where a single topic stands in its review cycle."""

    ease: float = DEFAULT_CONFIG.initial_ease
    interval_days: int = 0
    repetitions: int = 0
    due_on: date | None = None
    last_reviewed_on: date | None = None


def new_schedule(config: SM2Config = DEFAULT_CONFIG) -> Schedule:
    """Schedule for a topic that has never been reviewed. It counts as due straight away."""
    return Schedule(ease=config.initial_ease)


def _check_grade(grade: int) -> None:
    if isinstance(grade, bool) or not isinstance(grade, int):
        raise TypeError(f"grade must be an int, got {type(grade).__name__}")
    if not MIN_GRADE <= grade <= MAX_GRADE:
        raise ValueError(f"grade must be between {MIN_GRADE} and {MAX_GRADE}, got {grade}")


def _round_half_up(number: float) -> int:
    # Python's round() sends 2.5 to 2. Day counts should round the way people expect.
    return int(math.floor(number + 0.5))


def ease_adjustment(grade: int) -> float:
    """How much a grade moves the ease factor: 0.1 - s * (0.08 + s * 0.02), s = 5 - grade."""
    _check_grade(grade)
    shortfall = MAX_GRADE - grade
    return 0.1 - shortfall * (0.08 + shortfall * 0.02)


def next_schedule(
    schedule: Schedule,
    grade: int,
    today: date | None = None,
    config: SM2Config = DEFAULT_CONFIG,
) -> Schedule:
    """Return the schedule that follows a review.

    A failing grade restarts the repetition count and brings the topic back after the first
    interval, leaving the ease factor alone, as the original algorithm specifies. A passing
    grade moves the topic along the 1 day, 6 days, then interval * ease sequence. The new
    interval is worked out from the ease factor the topic had before this review.
    """
    _check_grade(grade)
    today = today or date.today()

    if grade < config.pass_grade:
        return replace(
            schedule,
            interval_days=config.first_interval_days,
            repetitions=0,
            due_on=today + timedelta(days=config.first_interval_days),
            last_reviewed_on=today,
        )

    if schedule.repetitions == 0:
        interval = config.first_interval_days
    elif schedule.repetitions == 1:
        interval = config.second_interval_days
    else:
        interval = _round_half_up(schedule.interval_days * schedule.ease)

    if config.max_interval_days is not None:
        interval = min(interval, config.max_interval_days)

    ease = max(config.min_ease, schedule.ease + ease_adjustment(grade))

    return Schedule(
        ease=ease,
        interval_days=interval,
        repetitions=schedule.repetitions + 1,
        due_on=today + timedelta(days=interval),
        last_reviewed_on=today,
    )


def is_due(schedule: Schedule, today: date | None = None) -> bool:
    today = today or date.today()
    return schedule.due_on is None or schedule.due_on <= today


def days_overdue(schedule: Schedule, today: date | None = None) -> int:
    """Whole days past the due date. Zero for topics that are not late or are brand new."""
    if schedule.due_on is None:
        return 0
    today = today or date.today()
    return max(0, (today - schedule.due_on).days)


def needs_repeat_in_session(grade: int, config: SM2Config = DEFAULT_CONFIG) -> bool:
    """SM-2 asks for anything graded below 4 to be shown again before the session ends."""
    _check_grade(grade)
    return grade < config.repeat_below_grade
