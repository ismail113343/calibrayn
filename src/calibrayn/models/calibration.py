from enum import Enum


class Confidence(Enum):
    SURE = "sure"
    UNSURE = "unsure"
    GUESSING = "guessing"


class CalibrationState(Enum):
    INSUFFICIENT_DATA = "insufficient_data"
    OVERCONFIDENT = "overconfident"
    UNDERCONFIDENT = "underconfident"
    CALIBRATED = "calibrated"
