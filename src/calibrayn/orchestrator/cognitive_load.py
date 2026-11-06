import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from statistics import fmean, median


class LoadLevel(Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass(frozen=True)
class CLConfig:
    window: int = 10
    baseline_window: int = 30
    min_attempts: int = 5
    min_baseline_attempts: int = 5
    max_response_seconds: float = 300.0
    slow_ratio: float = 2.0
    fatigue_start_minutes: float = 20.0
    fatigue_full_minutes: float = 60.0
    accuracy_weight: float = 0.5
    response_time_weight: float = 0.3
    session_weight: float = 0.2
    low_below: float = 0.35
    high_above: float = 0.65

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if not 1 <= self.min_attempts <= self.window:
            raise ValueError("min_attempts must be between 1 and window")
        if self.min_baseline_attempts < 1:
            raise ValueError("min_baseline_attempts must be at least 1")
        if self.baseline_window < self.window + self.min_baseline_attempts:
            raise ValueError("baseline_window must hold the window plus the baseline attempts")
        if self.max_response_seconds <= 0:
            raise ValueError("max_response_seconds must be positive")
        if self.slow_ratio <= 1.0:
            raise ValueError("slow_ratio must be above 1")
        if not 0 <= self.fatigue_start_minutes < self.fatigue_full_minutes:
            raise ValueError("fatigue minutes must satisfy 0 <= start < full")
        weights = (self.accuracy_weight, self.response_time_weight, self.session_weight)
        if min(weights) < 0 or not math.isclose(sum(weights), 1.0):
            raise ValueError("weights must be non-negative and add up to 1")
        if self.accuracy_weight + self.session_weight <= 0:
            raise ValueError("accuracy_weight and session_weight cannot both be zero")
        if not 0.0 < self.low_below < self.high_above < 1.0:
            raise ValueError("thresholds must satisfy 0 < low_below < high_above < 1")


DEFAULT_CONFIG = CLConfig()


def _check_score(score: float) -> None:
    if not isinstance(score, int | float):
        raise TypeError(f"score must be a number, got {type(score).__name__}")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be between 0 and 1, got {score}")


def _check_positive_number(value: float, name: str, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'zero or more' if allow_zero else 'above zero'}")


@dataclass(frozen=True)
class Answer:
    score: float
    response_seconds: float

    def __post_init__(self) -> None:
        _check_score(self.score)
        _check_positive_number(self.response_seconds, "response_seconds")


@dataclass(frozen=True)
class LoadEstimate:
    level: LoadLevel
    load: float | None
    accuracy_signal: float | None
    response_time_signal: float | None
    session_signal: float | None
    attempts_used: int


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _response_time_signal(
    recent: Sequence[Answer], earlier: Sequence[Answer], config: CLConfig
) -> float | None:
    if len(earlier) < config.min_baseline_attempts:
        return None
    cap = config.max_response_seconds
    baseline = median(min(a.response_seconds, cap) for a in earlier)
    current = median(min(a.response_seconds, cap) for a in recent)
    return _clamp((current / baseline - 1.0) / (config.slow_ratio - 1.0))


def _level_for(load: float, config: CLConfig) -> LoadLevel:
    if load < config.low_below:
        return LoadLevel.LOW
    if load > config.high_above:
        return LoadLevel.HIGH
    return LoadLevel.MODERATE


def estimate_load(
    attempts: Sequence[Answer], session_minutes: float, config: CLConfig = DEFAULT_CONFIG
) -> LoadEstimate:
    _check_positive_number(session_minutes, "session_minutes", allow_zero=True)

    history = list(attempts)[-config.baseline_window :]
    recent = history[-config.window :]
    earlier = history[: -config.window]

    if len(recent) < config.min_attempts:
        return LoadEstimate(LoadLevel.UNKNOWN, None, None, None, None, len(recent))

    accuracy_signal = _clamp(1.0 - fmean(a.score for a in recent))
    fatigue_span = config.fatigue_full_minutes - config.fatigue_start_minutes
    session_signal = _clamp((session_minutes - config.fatigue_start_minutes) / fatigue_span)
    response_signal = _response_time_signal(recent, earlier, config)

    weighted = [
        (config.accuracy_weight, accuracy_signal),
        (config.session_weight, session_signal),
    ]
    if response_signal is not None:
        weighted.append((config.response_time_weight, response_signal))
    total_weight = sum(weight for weight, _ in weighted)
    load = _clamp(sum(weight * signal for weight, signal in weighted) / total_weight)

    return LoadEstimate(
        level=_level_for(load, config),
        load=load,
        accuracy_signal=accuracy_signal,
        response_time_signal=response_signal,
        session_signal=session_signal,
        attempts_used=len(recent),
    )


class CognitiveLoadEstimator:
    def __init__(self, config: CLConfig = DEFAULT_CONFIG, attempts: Sequence[Answer] = ()) -> None:
        self._config = config
        self._attempts: deque[Answer] = deque(maxlen=config.baseline_window)
        for attempt in attempts:
            if not isinstance(attempt, Answer):
                raise TypeError(f"attempts must be Answer objects, got {type(attempt).__name__}")
            self._attempts.append(attempt)

    @property
    def attempts(self) -> tuple[Answer, ...]:
        return tuple(self._attempts)

    def record(self, score: float, response_seconds: float) -> None:
        self._attempts.append(Answer(score, response_seconds))

    def estimate(self, session_minutes: float) -> LoadEstimate:
        return estimate_load(tuple(self._attempts), session_minutes, self._config)

    def clear(self) -> None:
        self._attempts.clear()

    def __repr__(self) -> str:
        return f"CognitiveLoadEstimator(attempts={len(self._attempts)})"
