import pytest

from calibrayn.orchestrator.cognitive_load import (
    DEFAULT_CONFIG,
    Answer,
    CLConfig,
    CognitiveLoadEstimator,
    LoadLevel,
    estimate_load,
)


def answers(count, score=1.0, seconds=10.0):
    return [Answer(score, seconds) for _ in range(count)]


def with_baseline(recent_score, recent_seconds, baseline_seconds=10.0):
    return answers(5, 1.0, baseline_seconds) + answers(10, recent_score, recent_seconds)


class TestNotEnoughData:
    def test_too_few_answers_gives_no_estimate(self):
        estimate = estimate_load(answers(4), session_minutes=10)
        assert estimate.level is LoadLevel.UNKNOWN
        assert estimate.load is None
        assert estimate.attempts_used == 4

    def test_empty_history_gives_no_estimate(self):
        assert estimate_load([], 0).level is LoadLevel.UNKNOWN

    def test_estimate_appears_at_the_minimum(self):
        assert estimate_load(answers(5), 0).level is LoadLevel.LOW


class TestAccuracySignal:
    def test_all_correct_means_no_accuracy_load(self):
        estimate = estimate_load(answers(10), 0)
        assert estimate.accuracy_signal == 0.0
        assert estimate.load == 0.0
        assert estimate.level is LoadLevel.LOW

    def test_all_wrong_means_full_accuracy_load(self):
        estimate = estimate_load(answers(10, score=0.0), 0)
        assert estimate.accuracy_signal == 1.0
        assert estimate.level is LoadLevel.HIGH

    def test_partial_credit_is_averaged(self):
        estimate = estimate_load(answers(10, score=0.25), 0)
        assert estimate.accuracy_signal == pytest.approx(0.75)

    def test_only_the_latest_window_counts(self):
        history = answers(20, score=0.0) + answers(10, score=1.0)
        estimate = estimate_load(history, 0)
        assert estimate.accuracy_signal == 0.0
        assert estimate.attempts_used == 10


class TestResponseTimeSignal:
    def test_signal_is_missing_without_a_baseline(self):
        assert estimate_load(answers(10), 0).response_time_signal is None

    def test_signal_is_missing_with_too_few_baseline_answers(self):
        history = answers(4, 1.0, 10.0) + answers(10, 1.0, 30.0)
        assert estimate_load(history, 0).response_time_signal is None

    def test_normal_pace_gives_no_signal(self):
        assert estimate_load(with_baseline(1.0, 10.0), 0).response_time_signal == 0.0

    def test_faster_than_usual_gives_no_signal(self):
        assert estimate_load(with_baseline(1.0, 5.0), 0).response_time_signal == 0.0

    def test_halfway_to_the_slow_ratio_gives_half_signal(self):
        estimate = estimate_load(with_baseline(1.0, 15.0), 0)
        assert estimate.response_time_signal == pytest.approx(0.5)

    def test_twice_as_slow_gives_full_signal(self):
        estimate = estimate_load(with_baseline(1.0, 20.0), 0)
        assert estimate.response_time_signal == pytest.approx(1.0)

    def test_much_slower_is_capped_at_one(self):
        assert estimate_load(with_baseline(1.0, 100.0), 0).response_time_signal == 1.0

    def test_one_very_long_answer_does_not_distort_the_signal(self):
        recent = answers(9, 1.0, 10.0) + [Answer(1.0, 5000.0)]
        estimate = estimate_load(answers(5, 1.0, 10.0) + recent, 0)
        assert estimate.response_time_signal == 0.0

    def test_missing_signal_rescales_the_other_weights(self):
        estimate = estimate_load(answers(10, score=0.0), 0)
        assert estimate.load == pytest.approx(0.5 / 0.7)


class TestSessionSignal:
    @pytest.mark.parametrize(
        ("minutes", "expected"),
        [(0, 0.0), (20, 0.0), (30, 0.25), (40, 0.5), (60, 1.0), (120, 1.0)],
    )
    def test_fatigue_rises_between_the_start_and_full_marks(self, minutes, expected):
        estimate = estimate_load(answers(10), minutes)
        assert estimate.session_signal == pytest.approx(expected)

    def test_a_long_session_alone_raises_the_estimate(self):
        fresh = estimate_load(answers(10), 5)
        tired = estimate_load(answers(10), 90)
        assert tired.load > fresh.load


class TestCombination:
    def test_all_three_signals_are_weighted(self):
        history = answers(5, 1.0, 10.0) + answers(10, 0.0, 10.0)
        estimate = estimate_load(history, 40)
        assert estimate.load == pytest.approx(0.5 * 1.0 + 0.3 * 0.0 + 0.2 * 0.5)
        assert estimate.level is LoadLevel.MODERATE

    def test_equal_signals_give_that_value(self):
        history = answers(5, 1.0, 10.0) + answers(10, 0.5, 15.0)
        assert estimate_load(history, 40).load == pytest.approx(0.5)

    def test_load_stays_between_zero_and_one(self):
        for score in (0.0, 0.3, 1.0):
            for seconds in (1.0, 10.0, 500.0):
                for minutes in (0, 30, 600):
                    history = answers(5, 1.0, 10.0) + answers(10, score, seconds)
                    load = estimate_load(history, minutes).load
                    assert 0.0 <= load <= 1.0


class TestLevels:
    def test_middling_results_are_moderate(self):
        assert estimate_load(answers(10, score=0.5), 0).level is LoadLevel.MODERATE

    def test_boundaries_count_as_moderate(self):
        config = CLConfig(
            accuracy_weight=1.0,
            response_time_weight=0.0,
            session_weight=0.0,
            low_below=0.25,
            high_above=0.5,
        )
        assert estimate_load(answers(10, score=0.75), 0, config).level is LoadLevel.MODERATE
        assert estimate_load(answers(10, score=0.5), 0, config).level is LoadLevel.MODERATE
        assert estimate_load(answers(10, score=0.9), 0, config).level is LoadLevel.LOW
        assert estimate_load(answers(10, score=0.4), 0, config).level is LoadLevel.HIGH


class TestEstimator:
    def test_starts_empty(self):
        estimator = CognitiveLoadEstimator()
        assert estimator.attempts == ()
        assert estimator.estimate(0).level is LoadLevel.UNKNOWN

    def test_recorded_answers_feed_the_estimate(self):
        estimator = CognitiveLoadEstimator()
        for _ in range(10):
            estimator.record(0.0, 12.0)
        assert estimator.estimate(0).level is LoadLevel.HIGH

    def test_history_is_trimmed_to_the_baseline_window(self):
        estimator = CognitiveLoadEstimator()
        for _ in range(50):
            estimator.record(1.0, 10.0)
        assert len(estimator.attempts) == DEFAULT_CONFIG.baseline_window

    def test_can_be_rebuilt_from_saved_answers(self):
        original = CognitiveLoadEstimator()
        for index in range(20):
            original.record(1.0 if index % 3 else 0.0, 8.0 + index)
        copy = CognitiveLoadEstimator(attempts=original.attempts)
        assert copy.attempts == original.attempts
        assert copy.estimate(25) == original.estimate(25)

    def test_clear_forgets_the_history(self):
        estimator = CognitiveLoadEstimator()
        estimator.record(1.0, 10.0)
        estimator.clear()
        assert estimator.attempts == ()

    def test_bad_answer_is_rejected_and_not_stored(self):
        estimator = CognitiveLoadEstimator()
        with pytest.raises(ValueError):
            estimator.record(1.5, 10.0)
        assert estimator.attempts == ()

    def test_restoring_requires_answer_objects(self):
        with pytest.raises(TypeError):
            CognitiveLoadEstimator(attempts=[(1.0, 10.0)])

    def test_repr_shows_how_many_answers_are_held(self):
        estimator = CognitiveLoadEstimator()
        estimator.record(1.0, 10.0)
        assert repr(estimator) == "CognitiveLoadEstimator(attempts=1)"


class TestValidation:
    @pytest.mark.parametrize("score", [-0.1, 1.1, float("nan")])
    def test_score_outside_zero_to_one_is_rejected(self, score):
        with pytest.raises(ValueError):
            Answer(score, 10.0)

    @pytest.mark.parametrize("seconds", [0, -3.0, float("nan"), float("inf")])
    def test_unusable_response_time_is_rejected(self, seconds):
        with pytest.raises(ValueError):
            Answer(1.0, seconds)

    @pytest.mark.parametrize("seconds", ["10", None, True])
    def test_non_numeric_response_time_is_rejected(self, seconds):
        with pytest.raises(TypeError):
            Answer(1.0, seconds)

    def test_non_numeric_score_is_rejected(self):
        with pytest.raises(TypeError):
            Answer("1", 10.0)

    @pytest.mark.parametrize("minutes", [-1.0, float("nan"), float("inf")])
    def test_unusable_session_length_is_rejected(self, minutes):
        with pytest.raises(ValueError):
            estimate_load(answers(10), minutes)

    def test_non_numeric_session_length_is_rejected(self):
        with pytest.raises(TypeError):
            estimate_load(answers(10), "30")

    @pytest.mark.parametrize(
        "bad_settings",
        [
            {"window": 0},
            {"min_attempts": 0},
            {"min_attempts": 11},
            {"min_baseline_attempts": 0},
            {"baseline_window": 12},
            {"max_response_seconds": 0},
            {"slow_ratio": 1.0},
            {"fatigue_start_minutes": -1.0},
            {"fatigue_start_minutes": 60.0},
            {"accuracy_weight": 0.6},
            {"accuracy_weight": -0.1, "response_time_weight": 0.9},
            {"accuracy_weight": 0.0, "response_time_weight": 1.0, "session_weight": 0.0},
            {"low_below": 0.0},
            {"low_below": 0.7},
            {"high_above": 1.0},
        ],
    )
    def test_bad_config_is_rejected(self, bad_settings):
        with pytest.raises(ValueError):
            CLConfig(**bad_settings)
