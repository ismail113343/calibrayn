from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from statistics import fmean

from calibrayn.models.calibration import CalibrationState, Confidence


class Intervention(Enum):
    NONE = "none"
    DELAYED_RECALL = "delayed_recall"
    REDUCE_SUPPORT = "reduce_support"


@dataclass(frozen=True)
class MetaConfig:
    window: int = 15
    min_ratings_before_use: int = 10
    overconfidence_threshold: float = 0.30
    underconfidence_threshold: float = 0.30
    sure_level: float = 1.0
    unsure_level: float = 0.5
    guessing_level: float = 0.0

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if not 1 <= self.min_ratings_before_use <= self.window:
            raise ValueError("min_ratings_before_use must be between 1 and window")
        for name in ("overconfidence_threshold", "underconfidence_threshold"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be above 0 and at most 1")
        levels = [self.guessing_level, self.unsure_level, self.sure_level]
        if min(levels) < 0.0 or max(levels) > 1.0 or levels != sorted(levels):
            raise ValueError("levels must run from guessing up to sure, inside 0 and 1")

    def level_for(self, confidence: Confidence) -> float:
        if confidence is Confidence.SURE:
            return self.sure_level
        if confidence is Confidence.UNSURE:
            return self.unsure_level
        return self.guessing_level


DEFAULT_CONFIG = MetaConfig()


@dataclass(frozen=True)
class Rating:
    confidence: Confidence
    correct: bool

    def __post_init__(self) -> None:
        if not isinstance(self.confidence, Confidence):
            raise TypeError(
                f"confidence must be a Confidence, got {type(self.confidence).__name__}"
            )
        if not isinstance(self.correct, bool):
            raise TypeError(f"correct must be a bool, got {type(self.correct).__name__}")


@dataclass(frozen=True)
class CalibrationReading:
    state: CalibrationState
    score: float | None
    bias: float | None
    confident_errors: int
    ratings_used: int


def assess_calibration(
    ratings: Sequence[Rating], config: MetaConfig = DEFAULT_CONFIG
) -> CalibrationReading:
    recent = list(ratings)[-config.window :]
    for rating in recent:
        if not isinstance(rating, Rating):
            raise TypeError(f"ratings must be Rating objects, got {type(rating).__name__}")

    confident_errors = sum(
        1 for rating in recent if rating.confidence is Confidence.SURE and not rating.correct
    )
    if len(recent) < config.min_ratings_before_use:
        return CalibrationReading(
            CalibrationState.INSUFFICIENT_DATA, None, None, confident_errors, len(recent)
        )

    gaps = [config.level_for(r.confidence) - (1.0 if r.correct else 0.0) for r in recent]
    bias = fmean(gaps)
    score = 1.0 - fmean(abs(gap) for gap in gaps)

    if bias > config.overconfidence_threshold:
        state = CalibrationState.OVERCONFIDENT
    elif bias < -config.underconfidence_threshold:
        state = CalibrationState.UNDERCONFIDENT
    else:
        state = CalibrationState.CALIBRATED
    return CalibrationReading(state, score, bias, confident_errors, len(recent))


def recommend(reading: CalibrationReading) -> Intervention:
    if reading.state is CalibrationState.OVERCONFIDENT:
        return Intervention.DELAYED_RECALL
    if reading.state is CalibrationState.UNDERCONFIDENT:
        return Intervention.REDUCE_SUPPORT
    return Intervention.NONE


class MetacognitionTracker:
    def __init__(self, config: MetaConfig = DEFAULT_CONFIG, ratings: Sequence[Rating] = ()) -> None:
        self._config = config
        self._ratings: deque[Rating] = deque(maxlen=config.window)
        for rating in ratings:
            if not isinstance(rating, Rating):
                raise TypeError(f"ratings must be Rating objects, got {type(rating).__name__}")
            self._ratings.append(rating)

    @property
    def ratings(self) -> tuple[Rating, ...]:
        return tuple(self._ratings)

    def record(self, confidence: Confidence, correct: bool) -> None:
        self._ratings.append(Rating(confidence, correct))

    def reading(self) -> CalibrationReading:
        return assess_calibration(tuple(self._ratings), self._config)

    def intervention(self) -> Intervention:
        return recommend(self.reading())

    def clear(self) -> None:
        self._ratings.clear()

    def __repr__(self) -> str:
        return f"MetacognitionTracker(ratings={len(self._ratings)})"
