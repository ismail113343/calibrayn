import pytest

from calibrayn.models.calibration import CalibrationState, Confidence
from calibrayn.orchestrator.metacognition import (
    DEFAULT_CONFIG,
    CalibrationReading,
    Intervention,
    MetacognitionTracker,
    MetaConfig,
    Rating,
    assess_calibration,
    recommend,
)

SURE = Confidence.SURE
UNSURE = Confidence.UNSURE
GUESSING = Confidence.GUESSING


def ratings(confidence, correct, count):
    return [Rating(confidence, correct) for _ in range(count)]


class TestNotEnoughData:
    def test_too_few_ratings_gives_no_verdict(self):
        reading = assess_calibration(ratings(SURE, True, 9))
        assert reading.state is CalibrationState.INSUFFICIENT_DATA
        assert reading.score is None
        assert reading.bias is None
        assert reading.ratings_used == 9

    def test_empty_history_gives_no_verdict(self):
        assert assess_calibration([]).state is CalibrationState.INSUFFICIENT_DATA

    def test_confident_errors_are_counted_without_a_verdict(self):
        reading = assess_calibration(ratings(SURE, False, 4))
        assert reading.state is CalibrationState.INSUFFICIENT_DATA
        assert reading.confident_errors == 4

    def test_verdict_appears_at_the_minimum_ratings(self):
        reading = assess_calibration(ratings(SURE, True, 10))
        assert reading.state is CalibrationState.CALIBRATED


class TestCalibration:
    def test_sure_and_right_is_perfectly_calibrated(self):
        reading = assess_calibration(ratings(SURE, True, 10))
        assert reading.state is CalibrationState.CALIBRATED
        assert reading.score == 1.0
        assert reading.bias == 0.0

    def test_sure_and_wrong_is_overconfident(self):
        reading = assess_calibration(ratings(SURE, False, 10))
        assert reading.state is CalibrationState.OVERCONFIDENT
        assert reading.bias == 1.0
        assert reading.score == 0.0
        assert reading.confident_errors == 10

    def test_unsure_and_right_is_underconfident(self):
        reading = assess_calibration(ratings(UNSURE, True, 10))
        assert reading.state is CalibrationState.UNDERCONFIDENT
        assert reading.bias == pytest.approx(-0.5)
        assert reading.score == pytest.approx(0.5)

    def test_guessing_and_right_is_underconfident(self):
        reading = assess_calibration(ratings(GUESSING, True, 10))
        assert reading.state is CalibrationState.UNDERCONFIDENT
        assert reading.bias == pytest.approx(-1.0)
        assert reading.score == pytest.approx(0.0)

    def test_guessing_and_wrong_is_calibrated(self):
        reading = assess_calibration(ratings(GUESSING, False, 10))
        assert reading.state is CalibrationState.CALIBRATED
        assert reading.score == 1.0

    def test_unsure_and_half_right_is_calibrated(self):
        history = ratings(UNSURE, True, 5) + ratings(UNSURE, False, 5)
        reading = assess_calibration(history)
        assert reading.state is CalibrationState.CALIBRATED
        assert reading.bias == pytest.approx(0.0)
        assert reading.score == pytest.approx(0.5)

    def test_opposite_errors_cancel_in_bias_but_not_in_score(self):
        history = ratings(SURE, False, 5) + ratings(GUESSING, True, 5)
        reading = assess_calibration(history)
        assert reading.state is CalibrationState.CALIBRATED
        assert reading.bias == pytest.approx(0.0)
        assert reading.score == pytest.approx(0.0)

    def test_upper_threshold_counts_as_calibrated(self):
        history = ratings(SURE, False, 3) + ratings(SURE, True, 7)
        assert assess_calibration(history).state is CalibrationState.CALIBRATED

    def test_just_above_the_upper_threshold_is_overconfident(self):
        history = ratings(SURE, False, 4) + ratings(SURE, True, 6)
        assert assess_calibration(history).state is CalibrationState.OVERCONFIDENT

    def test_lower_threshold_counts_as_calibrated(self):
        history = ratings(UNSURE, True, 6) + ratings(SURE, True, 4)
        assert assess_calibration(history).state is CalibrationState.CALIBRATED

    def test_just_below_the_lower_threshold_is_underconfident(self):
        history = ratings(UNSURE, True, 7) + ratings(SURE, True, 3)
        assert assess_calibration(history).state is CalibrationState.UNDERCONFIDENT

    def test_only_the_latest_window_is_used(self):
        history = ratings(SURE, False, 20) + ratings(SURE, True, 15)
        reading = assess_calibration(history)
        assert reading.state is CalibrationState.CALIBRATED
        assert reading.ratings_used == 15

    def test_only_sure_and_wrong_counts_as_a_confident_error(self):
        history = ratings(SURE, False, 2) + ratings(UNSURE, False, 3) + ratings(SURE, True, 5)
        assert assess_calibration(history).confident_errors == 2

    def test_custom_levels_change_the_bias(self):
        config = MetaConfig(sure_level=0.8)
        reading = assess_calibration(ratings(SURE, True, 10), config)
        assert reading.bias == pytest.approx(-0.2)
        assert reading.score == pytest.approx(0.8)

    def test_custom_threshold_changes_the_verdict(self):
        config = MetaConfig(overconfidence_threshold=0.1)
        history = ratings(SURE, False, 2) + ratings(SURE, True, 8)
        assert assess_calibration(history, config).state is CalibrationState.OVERCONFIDENT

    def test_accepts_a_tuple(self):
        history = tuple(ratings(SURE, True, 10))
        assert assess_calibration(history).state is CalibrationState.CALIBRATED

    def test_does_not_change_the_input_list(self):
        history = ratings(SURE, True, 20)
        assess_calibration(history)
        assert len(history) == 20


class TestRecommend:
    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            (CalibrationState.OVERCONFIDENT, Intervention.DELAYED_RECALL),
            (CalibrationState.UNDERCONFIDENT, Intervention.REDUCE_SUPPORT),
            (CalibrationState.CALIBRATED, Intervention.NONE),
            (CalibrationState.INSUFFICIENT_DATA, Intervention.NONE),
        ],
    )
    def test_each_state_maps_to_one_intervention(self, state, expected):
        reading = CalibrationReading(state, None, None, 0, 0)
        assert recommend(reading) is expected


class TestTracker:
    def test_starts_empty(self):
        tracker = MetacognitionTracker()
        assert tracker.ratings == ()
        assert tracker.reading().state is CalibrationState.INSUFFICIENT_DATA
        assert tracker.intervention() is Intervention.NONE

    def test_recorded_ratings_feed_the_reading(self):
        tracker = MetacognitionTracker()
        for _ in range(10):
            tracker.record(SURE, False)
        assert tracker.reading().state is CalibrationState.OVERCONFIDENT
        assert tracker.intervention() is Intervention.DELAYED_RECALL

    def test_history_is_trimmed_to_the_window(self):
        tracker = MetacognitionTracker()
        for _ in range(40):
            tracker.record(SURE, True)
        assert len(tracker.ratings) == DEFAULT_CONFIG.window

    def test_can_be_rebuilt_from_saved_ratings(self):
        original = MetacognitionTracker()
        for index in range(12):
            original.record(UNSURE if index % 2 else SURE, index % 3 != 0)
        copy = MetacognitionTracker(ratings=original.ratings)
        assert copy.ratings == original.ratings
        assert copy.reading() == original.reading()

    def test_clear_forgets_the_history(self):
        tracker = MetacognitionTracker()
        tracker.record(SURE, True)
        tracker.clear()
        assert tracker.ratings == ()

    def test_bad_rating_is_rejected_and_not_stored(self):
        tracker = MetacognitionTracker()
        with pytest.raises(TypeError):
            tracker.record("sure", True)
        with pytest.raises(TypeError):
            tracker.record(SURE, 1)
        assert tracker.ratings == ()

    def test_restoring_requires_rating_objects(self):
        with pytest.raises(TypeError):
            MetacognitionTracker(ratings=[(SURE, True)])

    def test_repr_shows_how_many_ratings_are_held(self):
        tracker = MetacognitionTracker()
        tracker.record(SURE, True)
        assert repr(tracker) == "MetacognitionTracker(ratings=1)"


class TestValidation:
    def test_confidence_must_be_a_confidence(self):
        with pytest.raises(TypeError):
            Rating("sure", True)

    @pytest.mark.parametrize("correct", [1, 0, "yes", None])
    def test_correct_must_be_a_bool(self, correct):
        with pytest.raises(TypeError):
            Rating(SURE, correct)

    def test_assessing_requires_rating_objects(self):
        with pytest.raises(TypeError):
            assess_calibration([(SURE, True)] * 10)

    @pytest.mark.parametrize(
        "bad_settings",
        [
            {"window": 0},
            {"min_ratings_before_use": 0},
            {"min_ratings_before_use": 16},
            {"overconfidence_threshold": 0.0},
            {"underconfidence_threshold": 1.5},
            {"sure_level": 1.2},
            {"guessing_level": -0.1},
            {"guessing_level": 0.8},
            {"unsure_level": 1.0, "sure_level": 0.9},
        ],
    )
    def test_bad_config_is_rejected(self, bad_settings):
        with pytest.raises(ValueError):
            MetaConfig(**bad_settings)
