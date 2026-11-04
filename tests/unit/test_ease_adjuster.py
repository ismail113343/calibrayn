from datetime import datetime

import pytest

from calibrayn.models.calibration import CalibrationState, Confidence
from calibrayn.scheduler.ease_adjuster import (
    EaseAdjustConfig,
    adjust_for_confidence,
    penalty_for,
)
from calibrayn.scheduler.sm2 import Schedule, next_schedule

NOW = datetime(2026, 10, 19, 9, 30)
LEARNED = Schedule(
    easiness=2.5,
    interval_days=15,
    repetitions=3,
    due_at=datetime(2026, 10, 19),
    last_reviewed_at=datetime(2026, 10, 4),
)

SURE = Confidence.SURE
UNSURE = Confidence.UNSURE
GUESSING = Confidence.GUESSING
CALIBRATED = CalibrationState.CALIBRATED
OVERCONFIDENT = CalibrationState.OVERCONFIDENT
UNDERCONFIDENT = CalibrationState.UNDERCONFIDENT
NO_DATA = CalibrationState.INSUFFICIENT_DATA


def climb(schedule, passes):
    moment = schedule.due_at
    for _ in range(passes):
        schedule = next_schedule(schedule, 4, now=moment)
        moment = schedule.due_at
    return schedule


class TestPenalty:
    @pytest.mark.parametrize(
        ("confidence", "correct", "state", "expected"),
        [
            (SURE, False, CALIBRATED, 0.25),
            (SURE, False, UNDERCONFIDENT, 0.25),
            (SURE, False, OVERCONFIDENT, 0.35),
            (SURE, False, NO_DATA, 0.0),
            (SURE, True, OVERCONFIDENT, 0.0),
            (UNSURE, False, OVERCONFIDENT, 0.0),
            (GUESSING, False, OVERCONFIDENT, 0.0),
            (UNSURE, True, UNDERCONFIDENT, 0.0),
            (GUESSING, True, UNDERCONFIDENT, 0.0),
        ],
    )
    def test_only_a_confident_error_is_penalised(self, confidence, correct, state, expected):
        assert penalty_for(confidence, correct, state) == pytest.approx(expected)

    def test_amounts_follow_the_config(self):
        config = EaseAdjustConfig(confident_error_penalty=0.4, persistent_overconfidence_extra=0.2)
        assert penalty_for(SURE, False, CALIBRATED, config) == pytest.approx(0.4)
        assert penalty_for(SURE, False, OVERCONFIDENT, config) == pytest.approx(0.6)


class TestAdjust:
    def test_confident_error_lowers_easiness(self):
        result = adjust_for_confidence(LEARNED, SURE, False, CALIBRATED)
        assert result.easiness == pytest.approx(2.25)

    def test_persistent_overconfidence_lowers_it_further(self):
        result = adjust_for_confidence(LEARNED, SURE, False, OVERCONFIDENT)
        assert result.easiness == pytest.approx(2.15)

    def test_other_fields_are_untouched(self):
        result = adjust_for_confidence(LEARNED, SURE, False, OVERCONFIDENT)
        assert result.interval_days == LEARNED.interval_days
        assert result.repetitions == LEARNED.repetitions
        assert result.due_at == LEARNED.due_at
        assert result.last_reviewed_at == LEARNED.last_reviewed_at

    def test_no_penalty_returns_the_same_schedule(self):
        assert adjust_for_confidence(LEARNED, UNSURE, False, OVERCONFIDENT) == LEARNED
        assert adjust_for_confidence(LEARNED, SURE, True, OVERCONFIDENT) == LEARNED

    def test_too_few_ratings_means_correctness_only_scheduling(self):
        assert adjust_for_confidence(LEARNED, SURE, False, NO_DATA) == LEARNED

    def test_easiness_never_falls_below_the_floor(self):
        low = Schedule(easiness=1.4, interval_days=6, repetitions=2)
        assert adjust_for_confidence(low, SURE, False, OVERCONFIDENT).easiness == 1.3

    def test_floor_follows_the_config(self):
        config = EaseAdjustConfig(min_easiness=2.4)
        assert adjust_for_confidence(LEARNED, SURE, False, CALIBRATED, config).easiness == 2.4

    def test_original_schedule_is_not_modified(self):
        adjust_for_confidence(LEARNED, SURE, False, OVERCONFIDENT)
        assert LEARNED.easiness == 2.5


class TestWithSM2:
    def test_confident_error_is_hit_harder_than_a_simple_mistake(self):
        failed = next_schedule(LEARNED, 1, now=NOW)
        confident = adjust_for_confidence(failed, SURE, False, CALIBRATED)
        simple = adjust_for_confidence(failed, UNSURE, False, CALIBRATED)
        assert confident.easiness < simple.easiness
        assert simple.easiness == failed.easiness

    def test_the_topic_still_returns_within_minutes(self):
        failed = next_schedule(LEARNED, 1, now=NOW)
        confident = adjust_for_confidence(failed, SURE, False, OVERCONFIDENT)
        assert confident.due_at == failed.due_at

    def test_later_gaps_are_shorter_after_a_confident_error(self):
        failed = next_schedule(LEARNED, 1, now=NOW)
        confident = adjust_for_confidence(failed, SURE, False, CALIBRATED)
        simple = adjust_for_confidence(failed, UNSURE, False, CALIBRATED)
        assert climb(confident, 3).interval_days < climb(simple, 3).interval_days


class TestValidation:
    def test_confidence_must_be_a_confidence(self):
        with pytest.raises(TypeError):
            penalty_for("sure", False, CALIBRATED)

    @pytest.mark.parametrize("correct", [0, 1, None])
    def test_correct_must_be_a_bool(self, correct):
        with pytest.raises(TypeError):
            penalty_for(SURE, correct, CALIBRATED)

    def test_state_must_be_a_calibration_state(self):
        with pytest.raises(TypeError):
            penalty_for(SURE, False, "calibrated")

    @pytest.mark.parametrize(
        "bad_settings",
        [
            {"confident_error_penalty": -0.1},
            {"persistent_overconfidence_extra": -0.1},
            {"min_easiness": 0},
        ],
    )
    def test_bad_config_is_rejected(self, bad_settings):
        with pytest.raises(ValueError):
            EaseAdjustConfig(**bad_settings)
