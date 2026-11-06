from datetime import datetime, timedelta

import pytest

from calibrayn.models.calibration import CalibrationState, Confidence
from calibrayn.orchestrator.cognitive_load import LoadLevel
from calibrayn.orchestrator.metacognition import CalibrationReading
from calibrayn.orchestrator.zpd import Zone
from calibrayn.scheduler.sm2 import Schedule
from calibrayn.ui.components.labels import (
    CONFIDENCE_CHOICES,
    LOAD_TEXT,
    ZONE_TEXT,
    describe_calibration,
    describe_next_review,
)

NOW = datetime(2026, 11, 3, 15, 30)


def reading(state, used=0):
    return CalibrationReading(state, None, None, 0, used)


def test_every_zone_and_load_level_has_text():
    assert set(ZONE_TEXT) == set(Zone)
    assert set(LOAD_TEXT) == set(LoadLevel)


def test_the_three_confidence_choices_map_to_the_enum():
    assert list(CONFIDENCE_CHOICES.values()) == [
        Confidence.SURE,
        Confidence.UNSURE,
        Confidence.GUESSING,
    ]


class TestNextReview:
    def test_a_failed_topic_says_how_many_minutes(self):
        schedule = Schedule(repetitions=0, due_at=NOW + timedelta(minutes=10), last_reviewed_at=NOW)
        assert describe_next_review(schedule, NOW) == "in about 10 minutes"

    def test_tomorrow(self):
        schedule = Schedule(repetitions=1, due_at=datetime(2026, 11, 4), last_reviewed_at=NOW)
        assert describe_next_review(schedule, NOW) == "tomorrow"

    def test_later_days_show_the_date(self):
        schedule = Schedule(repetitions=2, due_at=datetime(2026, 11, 9), last_reviewed_at=NOW)
        assert describe_next_review(schedule, NOW) == "in 6 days, on 09 Nov"

    def test_the_same_day(self):
        schedule = Schedule(repetitions=2, due_at=datetime(2026, 11, 3), last_reviewed_at=NOW)
        assert describe_next_review(schedule, NOW) == "later today"


class TestCalibrationText:
    def test_before_enough_answers_it_shows_progress(self):
        text = describe_calibration(reading(CalibrationState.INSUFFICIENT_DATA, used=4), 10)
        assert text == "Confidence check starts after 10 answers (4 so far)."

    @pytest.mark.parametrize(
        ("state", "phrase"),
        [
            (CalibrationState.OVERCONFIDENT, "feel sure when you are wrong"),
            (CalibrationState.UNDERCONFIDENT, "know more than you think"),
            (CalibrationState.CALIBRATED, "matches your results"),
        ],
    )
    def test_each_state_has_its_own_message(self, state, phrase):
        assert phrase in describe_calibration(reading(state), 10)
