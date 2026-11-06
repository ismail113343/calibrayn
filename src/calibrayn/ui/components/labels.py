from calibrayn.models.calibration import CalibrationState, Confidence
from calibrayn.orchestrator.cognitive_load import LoadLevel
from calibrayn.orchestrator.zpd import Zone

CONFIDENCE_CHOICES = {
    "Sure": Confidence.SURE,
    "Unsure": Confidence.UNSURE,
    "Guessing": Confidence.GUESSING,
}

ZONE_TEXT = {
    Zone.INSUFFICIENT_DATA: "Not enough answers yet",
    Zone.TOO_HARD: "A bit too hard",
    Zone.IN_ZONE: "Just right",
    Zone.TOO_EASY: "A bit too easy",
}

LOAD_TEXT = {
    LoadLevel.UNKNOWN: "Not enough answers yet",
    LoadLevel.LOW: "Light",
    LoadLevel.MODERATE: "Moderate",
    LoadLevel.HIGH: "Heavy",
}


def describe_next_review(schedule, now):
    if schedule.is_relearning:
        minutes = max(1, int((schedule.due_at - schedule.last_reviewed_at).total_seconds() // 60))
        return f"in about {minutes} minutes"
    days = (schedule.due_at.date() - now.date()).days
    if days <= 0:
        return "later today"
    if days == 1:
        return "tomorrow"
    return f"in {days} days, on {schedule.due_at:%d %b}"


def describe_calibration(reading, needed):
    if reading.state is CalibrationState.INSUFFICIENT_DATA:
        return f"Confidence check starts after {needed} answers ({reading.ratings_used} so far)."
    if reading.state is CalibrationState.OVERCONFIDENT:
        return "You often feel sure when you are wrong. Expect a few recall checks."
    if reading.state is CalibrationState.UNDERCONFIDENT:
        return "You know more than you think. Expect harder questions with less help."
    return "Your confidence matches your results."
