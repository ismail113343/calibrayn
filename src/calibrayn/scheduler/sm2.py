"""SM-2 spaced repetition scheduling.

Follows the algorithm described by P. A. Wozniak (1990), with one change taken from section 2.4
of the proposal: a topic answered wrongly comes back within minutes instead of the next day.
The module is pure. It takes a schedule and a grade and returns a new schedule, with no reads or
writes to disk, so it is easy to test and easy to call from the orchestrator.

Grades:
    5  perfect response
    4  correct after some hesitation
    3  correct, but recalled with serious difficulty
    2  incorrect, but the answer felt familiar once seen
    1  incorrect, and the answer was remembered once seen
    0  complete blackout

Timing:
    Intervals after a pass are counted in calendar days. A topic reviewed on Monday with a one
    day interval falls due at the start of Tuesday and stays due for the rest of that day.
    Relearning after a failure is counted in minutes from the moment of the failed review.
    Pass the same kind of datetime throughout, either all timezone-aware or all naive.
"""

import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

MIN_GRADE = 0
MAX_GRADE = 5


@dataclass(frozen=True)
class SM2Config:
    """Tunable constants. The names match the sm2 block in configs/settings.yaml."""

    initial_easiness: float = 2.5
    min_easiness: float = 1.3
    first_interval_days: int = 1
    second_interval_days: int = 6
    pass_grade: int = 3
    repeat_below_grade: int = 4
    relearn_minutes: int = 10
    max_interval_days: int | None = None

    def __post_init__(self) -> None:
        if self.min_easiness <= 0 or self.min_easiness > self.initial_easiness:
            raise ValueError("min_easiness must be positive and no higher than initial_easiness")
        if self.first_interval_days < 1:
            raise ValueError("first_interval_days must be at least 1")
        if self.second_interval_days < self.first_interval_days:
            raise ValueError("second_interval_days cannot be shorter than the first interval")
        if not MIN_GRADE < self.pass_grade <= MAX_GRADE:
            raise ValueError("pass_grade must be between 1 and 5")
        if self.repeat_below_grade < self.pass_grade:
            raise ValueError("repeat_below_grade cannot be lower than pass_grade")
        if self.relearn_minutes < 1:
            raise ValueError("relearn_minutes must be at least 1")
        cap = self.max_interval_days
        if cap is not None and cap < self.second_interval_days:
            raise ValueError("max_interval_days cannot be shorter than the second interval")


DEFAULT_CONFIG = SM2Config()


@dataclass(frozen=True)
class Schedule:
    """Where a single topic stands in its review cycle."""

    easiness: float = DEFAULT_CONFIG.initial_easiness
    interval_days: int = 0
    repetitions: int = 0
    due_at: datetime | None = None
    last_reviewed_at: datetime | None = None

    @property
    def is_relearning(self) -> bool:
        """True for a topic that has been reviewed but has no successful run behind it."""
        return self.repetitions == 0 and self.last_reviewed_at is not None


def new_schedule(config: SM2Config = DEFAULT_CONFIG) -> Schedule:
    """Schedule for a topic that has never been reviewed. It counts as due straight away."""
    return Schedule(easiness=config.initial_easiness)


def _check_grade(grade: int) -> None:
    if isinstance(grade, bool) or not isinstance(grade, int):
        raise TypeError(f"grade must be an int, got {type(grade).__name__}")
    if not MIN_GRADE <= grade <= MAX_GRADE:
        raise ValueError(f"grade must be between {MIN_GRADE} and {MAX_GRADE}, got {grade}")


def _round_half_up(number: float) -> int:
    # Python's round() sends 2.5 to 2. Day counts should round the way people expect.
    return int(math.floor(number + 0.5))


def _start_of_day(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def easiness_adjustment(grade: int) -> float:
    """How much a grade moves the easiness factor: 0.1 - s * (0.08 + s * 0.02), s = 5 - grade."""
    _check_grade(grade)
    shortfall = MAX_GRADE - grade
    return 0.1 - shortfall * (0.08 + shortfall * 0.02)


def next_schedule(
    schedule: Schedule,
    grade: int,
    now: datetime | None = None,
    config: SM2Config = DEFAULT_CONFIG,
) -> Schedule:
    """Return the schedule that follows a review.

    A failing grade restarts the repetition count and brings the topic back after
    `relearn_minutes`, leaving the easiness factor alone, as the original algorithm specifies.
    The first pass after that moves it onto the 1 day, 6 days, then interval * easiness sequence.
    The new interval is worked out from the easiness the topic had before this review.
    """
    _check_grade(grade)
    now = now or datetime.now()

    if grade < config.pass_grade:
        return replace(
            schedule,
            interval_days=0,
            repetitions=0,
            due_at=now + timedelta(minutes=config.relearn_minutes),
            last_reviewed_at=now,
        )

    if schedule.repetitions == 0:
        interval = config.first_interval_days
    elif schedule.repetitions == 1:
        interval = config.second_interval_days
    else:
        interval = _round_half_up(schedule.interval_days * schedule.easiness)

    if config.max_interval_days is not None:
        interval = min(interval, config.max_interval_days)

    easiness = max(config.min_easiness, schedule.easiness + easiness_adjustment(grade))

    return Schedule(
        easiness=easiness,
        interval_days=interval,
        repetitions=schedule.repetitions + 1,
        due_at=_start_of_day(now) + timedelta(days=interval),
        last_reviewed_at=now,
    )


def is_due(schedule: Schedule, now: datetime | None = None) -> bool:
    now = now or datetime.now()
    return schedule.due_at is None or schedule.due_at <= now


def days_overdue(schedule: Schedule, now: datetime | None = None) -> int:
    """Whole days past the due time. Zero for topics that are not late or are brand new."""
    if schedule.due_at is None:
        return 0
    now = now or datetime.now()
    return max(0, (now - schedule.due_at).days)


def needs_repeat_in_session(grade: int, config: SM2Config = DEFAULT_CONFIG) -> bool:
    """SM-2 asks for anything graded below 4 to be shown again before the session ends."""
    _check_grade(grade)
    return grade < config.repeat_below_grade
