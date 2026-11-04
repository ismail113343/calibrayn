from dataclasses import dataclass, replace

from calibrayn.models.calibration import CalibrationState, Confidence
from calibrayn.scheduler.sm2 import Schedule


@dataclass(frozen=True)
class EaseAdjustConfig:
    confident_error_penalty: float = 0.25
    persistent_overconfidence_extra: float = 0.10
    min_easiness: float = 1.3

    def __post_init__(self) -> None:
        if self.confident_error_penalty < 0:
            raise ValueError("confident_error_penalty cannot be negative")
        if self.persistent_overconfidence_extra < 0:
            raise ValueError("persistent_overconfidence_extra cannot be negative")
        if self.min_easiness <= 0:
            raise ValueError("min_easiness must be positive")


DEFAULT_CONFIG = EaseAdjustConfig()


def penalty_for(
    confidence: Confidence,
    correct: bool,
    state: CalibrationState,
    config: EaseAdjustConfig = DEFAULT_CONFIG,
) -> float:
    if not isinstance(confidence, Confidence):
        raise TypeError(f"confidence must be a Confidence, got {type(confidence).__name__}")
    if not isinstance(correct, bool):
        raise TypeError(f"correct must be a bool, got {type(correct).__name__}")
    if not isinstance(state, CalibrationState):
        raise TypeError(f"state must be a CalibrationState, got {type(state).__name__}")

    if state is CalibrationState.INSUFFICIENT_DATA or correct:
        return 0.0
    if confidence is not Confidence.SURE:
        return 0.0
    if state is CalibrationState.OVERCONFIDENT:
        return config.confident_error_penalty + config.persistent_overconfidence_extra
    return config.confident_error_penalty


def adjust_for_confidence(
    schedule: Schedule,
    confidence: Confidence,
    correct: bool,
    state: CalibrationState,
    config: EaseAdjustConfig = DEFAULT_CONFIG,
) -> Schedule:
    penalty = penalty_for(confidence, correct, state, config)
    if penalty == 0.0:
        return schedule
    return replace(schedule, easiness=max(config.min_easiness, schedule.easiness - penalty))
