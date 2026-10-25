import pytest

from calibrayn.orchestrator.zpd import (
    DEFAULT_CONFIG,
    Zone,
    ZPDConfig,
    ZPDStatus,
    ZPDTracker,
    assess,
    suggest_level,
)


def scores(correct, wrong):
    return [1.0] * correct + [0.0] * wrong


class TestAssess:
    def test_too_few_attempts_gives_no_verdict(self):
        status = assess(scores(3, 1))
        assert status.zone is Zone.INSUFFICIENT_DATA
        assert status.accuracy is None
        assert status.attempts_used == 4

    def test_empty_history_gives_no_verdict(self):
        assert assess([]).zone is Zone.INSUFFICIENT_DATA

    def test_verdict_appears_at_the_minimum_attempts(self):
        assert assess(scores(5, 0)).zone is Zone.TOO_EASY

    def test_low_accuracy_is_too_hard(self):
        status = assess(scores(3, 7))
        assert status.zone is Zone.TOO_HARD
        assert status.accuracy == pytest.approx(0.3)

    def test_high_accuracy_is_too_easy(self):
        status = assess(scores(9, 1))
        assert status.zone is Zone.TOO_EASY
        assert status.accuracy == pytest.approx(0.9)

    def test_middle_accuracy_is_in_zone(self):
        assert assess(scores(7, 3)).zone is Zone.IN_ZONE

    def test_lower_boundary_counts_as_in_zone(self):
        assert assess(scores(6, 4)).zone is Zone.IN_ZONE

    def test_upper_boundary_counts_as_in_zone(self):
        config = ZPDConfig(window=20)
        assert assess(scores(17, 3), config).zone is Zone.IN_ZONE

    def test_only_the_latest_window_is_used(self):
        status = assess(scores(0, 10) + scores(10, 0))
        assert status.attempts_used == 10
        assert status.zone is Zone.TOO_EASY

    def test_partial_credit_is_averaged(self):
        status = assess([0.5] * 10)
        assert status.accuracy == pytest.approx(0.5)
        assert status.zone is Zone.TOO_HARD

    def test_accepts_a_tuple(self):
        assert assess(tuple(scores(7, 3))).zone is Zone.IN_ZONE

    def test_does_not_change_the_input_list(self):
        history = scores(12, 3)
        assess(history)
        assert len(history) == 15


class TestSuggestLevel:
    def test_too_easy_moves_up_one_level(self):
        assert suggest_level(ZPDStatus(Zone.TOO_EASY, 0.95, 10), 2) == 3

    def test_too_hard_moves_down_one_level(self):
        assert suggest_level(ZPDStatus(Zone.TOO_HARD, 0.3, 10), 3) == 2

    def test_in_zone_stays_put(self):
        assert suggest_level(ZPDStatus(Zone.IN_ZONE, 0.7, 10), 3) == 3

    def test_no_data_stays_put(self):
        assert suggest_level(ZPDStatus(Zone.INSUFFICIENT_DATA, None, 2), 4) == 4

    def test_cannot_go_above_the_top_level(self):
        status = ZPDStatus(Zone.TOO_EASY, 0.95, 10)
        assert suggest_level(status, DEFAULT_CONFIG.max_level) == DEFAULT_CONFIG.max_level

    def test_cannot_go_below_the_bottom_level(self):
        status = ZPDStatus(Zone.TOO_HARD, 0.2, 10)
        assert suggest_level(status, DEFAULT_CONFIG.min_level) == DEFAULT_CONFIG.min_level

    @pytest.mark.parametrize("level", [0, 6, -1])
    def test_level_outside_the_range_is_rejected(self, level):
        with pytest.raises(ValueError):
            suggest_level(ZPDStatus(Zone.IN_ZONE, 0.7, 10), level)


class TestTrackerBasics:
    def test_starts_at_the_configured_level_with_no_history(self):
        tracker = ZPDTracker()
        assert tracker.level == DEFAULT_CONFIG.start_level
        assert tracker.scores == ()
        assert tracker.status().zone is Zone.INSUFFICIENT_DATA

    def test_can_start_at_a_chosen_level(self):
        assert ZPDTracker(level=5).level == 5

    def test_invalid_starting_level_is_rejected(self):
        with pytest.raises(ValueError):
            ZPDTracker(level=9)

    def test_recorded_scores_feed_the_status(self):
        tracker = ZPDTracker()
        for score in scores(7, 3):
            tracker.record(score)
        assert tracker.status().zone is Zone.IN_ZONE

    def test_history_is_trimmed_to_the_window(self):
        tracker = ZPDTracker()
        for _ in range(15):
            tracker.record(1.0)
        assert len(tracker.scores) == DEFAULT_CONFIG.window

    def test_can_be_rebuilt_from_saved_scores(self):
        original = ZPDTracker(level=4)
        for score in scores(8, 2):
            original.record(score)
        copy = ZPDTracker(level=original.level, scores=original.scores)
        assert copy.scores == original.scores
        assert copy.status() == original.status()

    def test_integer_and_boolean_scores_are_stored_as_floats(self):
        tracker = ZPDTracker()
        tracker.record(1)
        tracker.record(True)
        assert tracker.scores == (1.0, 1.0)

    def test_bad_score_is_rejected_and_not_stored(self):
        tracker = ZPDTracker()
        with pytest.raises(ValueError):
            tracker.record(1.5)
        assert tracker.scores == ()

    def test_repr_shows_level_and_attempts(self):
        tracker = ZPDTracker()
        tracker.record(1.0)
        assert repr(tracker) == "ZPDTracker(level=3, attempts=1)"


class TestTrackerLevelChanges:
    def test_changing_level_clears_history(self):
        tracker = ZPDTracker()
        tracker.record(1.0)
        tracker.set_level(4)
        assert tracker.level == 4
        assert tracker.scores == ()

    def test_setting_the_same_level_keeps_history(self):
        tracker = ZPDTracker()
        tracker.record(1.0)
        tracker.set_level(tracker.level)
        assert tracker.scores == (1.0,)

    def test_set_level_rejects_values_out_of_range(self):
        with pytest.raises(ValueError):
            ZPDTracker().set_level(0)

    def test_a_winning_streak_raises_the_level(self):
        tracker = ZPDTracker(level=2)
        for score in scores(10, 0):
            tracker.record(score)
        assert tracker.follow_suggestion() == 3
        assert tracker.level == 3

    def test_a_losing_streak_lowers_the_level(self):
        tracker = ZPDTracker(level=3)
        for score in scores(1, 9):
            tracker.record(score)
        assert tracker.follow_suggestion() == 2

    def test_history_restarts_after_a_move(self):
        tracker = ZPDTracker(level=2)
        for score in scores(10, 0):
            tracker.record(score)
        tracker.follow_suggestion()
        assert tracker.status().zone is Zone.INSUFFICIENT_DATA

    def test_no_second_move_until_enough_new_scores_exist(self):
        tracker = ZPDTracker(level=2)
        for score in scores(10, 0):
            tracker.record(score)
        tracker.follow_suggestion()
        for score in scores(4, 0):
            tracker.record(score)
        assert tracker.follow_suggestion() == 3
        tracker.record(1.0)
        assert tracker.follow_suggestion() == 4

    def test_staying_in_zone_keeps_level_and_history(self):
        tracker = ZPDTracker()
        for score in scores(7, 3):
            tracker.record(score)
        assert tracker.follow_suggestion() == DEFAULT_CONFIG.start_level
        assert len(tracker.scores) == 10

    def test_top_level_stays_put_and_keeps_history(self):
        tracker = ZPDTracker(level=DEFAULT_CONFIG.max_level)
        for score in scores(10, 0):
            tracker.record(score)
        assert tracker.follow_suggestion() == DEFAULT_CONFIG.max_level
        assert len(tracker.scores) == 10

    def test_scores_are_exposed_as_an_immutable_tuple(self):
        tracker = ZPDTracker()
        tracker.record(1.0)
        assert isinstance(tracker.scores, tuple)


class TestValidation:
    @pytest.mark.parametrize("bad_score", [-0.1, 1.1, float("nan")])
    def test_score_outside_zero_to_one_is_rejected(self, bad_score):
        with pytest.raises(ValueError):
            assess([1.0, 1.0, 1.0, 1.0, bad_score])

    @pytest.mark.parametrize("bad_score", ["1", None])
    def test_non_numeric_score_is_rejected(self, bad_score):
        with pytest.raises(TypeError):
            assess([1.0, 1.0, 1.0, 1.0, bad_score])

    @pytest.mark.parametrize(
        "bad_settings",
        [
            {"window": 0},
            {"min_attempts": 0},
            {"min_attempts": 11},
            {"target_low": 0.9, "target_high": 0.8},
            {"target_low": 0.7, "target_high": 0.7},
            {"target_low": -0.1},
            {"target_high": 1.2},
            {"min_level": 4, "max_level": 2},
            {"start_level": 0},
            {"start_level": 6},
        ],
    )
    def test_bad_config_is_rejected(self, bad_settings):
        with pytest.raises(ValueError):
            ZPDConfig(**bad_settings)
