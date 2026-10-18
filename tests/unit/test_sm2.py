from dataclasses import FrozenInstanceError
from datetime import date, timedelta

import pytest

from calibrayn.scheduler.sm2 import (
    DEFAULT_CONFIG,
    Schedule,
    SM2Config,
    days_overdue,
    ease_adjustment,
    is_due,
    needs_repeat_in_session,
    new_schedule,
    next_schedule,
)

TODAY = date(2026, 10, 19)


def run_reviews(grades, config=DEFAULT_CONFIG):
    schedule = new_schedule(config)
    day = TODAY
    for grade in grades:
        schedule = next_schedule(schedule, grade, today=day, config=config)
        day = schedule.due_on
    return schedule


class TestNewSchedule:
    def test_starts_with_default_ease_and_no_history(self):
        schedule = new_schedule()
        assert schedule.ease == 2.5
        assert schedule.interval_days == 0
        assert schedule.repetitions == 0
        assert schedule.due_on is None
        assert schedule.last_reviewed_on is None

    def test_uses_ease_from_custom_config(self):
        assert new_schedule(SM2Config(initial_ease=2.8)).ease == 2.8


class TestPassingReviews:
    def test_first_pass_comes_back_in_one_day(self):
        schedule = next_schedule(new_schedule(), 4, today=TODAY)
        assert schedule.interval_days == 1
        assert schedule.repetitions == 1
        assert schedule.due_on == TODAY + timedelta(days=1)
        assert schedule.last_reviewed_on == TODAY

    def test_second_pass_comes_back_in_six_days(self):
        schedule = run_reviews([4, 4])
        assert schedule.interval_days == 6
        assert schedule.repetitions == 2

    def test_third_pass_multiplies_by_ease(self):
        assert run_reviews([4, 4, 4]).interval_days == 15

    def test_interval_uses_ease_from_before_the_review(self):
        # two grade-5 reviews lift ease to 2.7, and the third review should use that 2.7
        assert run_reviews([5, 5, 5]).interval_days == 16

    def test_intervals_keep_growing_for_a_strong_learner(self):
        day = TODAY
        schedule = new_schedule()
        intervals = []
        for _ in range(10):
            schedule = next_schedule(schedule, 5, today=day)
            intervals.append(schedule.interval_days)
            day = schedule.due_on
        assert intervals[2:] == sorted(set(intervals[2:]))
        assert schedule.repetitions == 10

    def test_half_days_round_up(self):
        schedule = Schedule(ease=2.5, interval_days=5, repetitions=2)
        assert next_schedule(schedule, 4, today=TODAY).interval_days == 13


class TestFailingReviews:
    @pytest.mark.parametrize("grade", [0, 1, 2])
    def test_failure_restarts_the_cycle(self, grade):
        learned = Schedule(ease=2.2, interval_days=15, repetitions=3)
        schedule = next_schedule(learned, grade, today=TODAY)
        assert schedule.repetitions == 0
        assert schedule.interval_days == 1
        assert schedule.due_on == TODAY + timedelta(days=1)

    @pytest.mark.parametrize("grade", [0, 1, 2])
    def test_failure_leaves_ease_untouched(self, grade):
        learned = Schedule(ease=2.2, interval_days=15, repetitions=3)
        assert next_schedule(learned, grade, today=TODAY).ease == 2.2

    def test_a_topic_can_recover_after_failing(self):
        schedule = run_reviews([4, 4, 4, 1, 4, 4])
        assert schedule.repetitions == 2
        assert schedule.interval_days == 6


class TestEase:
    @pytest.mark.parametrize(
        ("grade", "expected"),
        [(5, 0.10), (4, 0.0), (3, -0.14), (2, -0.32), (1, -0.54), (0, -0.80)],
    )
    def test_adjustment_matches_the_sm2_formula(self, grade, expected):
        assert ease_adjustment(grade) == pytest.approx(expected)

    def test_hard_passes_lower_ease(self):
        assert next_schedule(new_schedule(), 3, today=TODAY).ease == pytest.approx(2.36)

    def test_ease_never_falls_below_the_floor(self):
        schedule = run_reviews([3] * 20)
        assert schedule.ease == DEFAULT_CONFIG.min_ease

    def test_floor_follows_a_custom_config(self):
        config = SM2Config(min_ease=1.8)
        assert run_reviews([3] * 8, config).ease == 1.8


class TestIntervalCap:
    def test_cap_limits_long_intervals(self):
        config = SM2Config(max_interval_days=20)
        schedule = run_reviews([5] * 8, config)
        assert schedule.interval_days == 20


class TestDueDates:
    def test_new_topic_is_due(self):
        assert is_due(new_schedule(), today=TODAY)

    def test_due_today_counts_as_due(self):
        assert is_due(Schedule(due_on=TODAY), today=TODAY)

    def test_future_topic_is_not_due(self):
        assert not is_due(Schedule(due_on=TODAY + timedelta(days=1)), today=TODAY)

    def test_overdue_topic_is_due(self):
        assert is_due(Schedule(due_on=TODAY - timedelta(days=3)), today=TODAY)

    def test_days_overdue_counts_late_days(self):
        assert days_overdue(Schedule(due_on=TODAY - timedelta(days=3)), today=TODAY) == 3

    def test_days_overdue_is_zero_when_not_late(self):
        assert days_overdue(Schedule(due_on=TODAY + timedelta(days=2)), today=TODAY) == 0
        assert days_overdue(new_schedule(), today=TODAY) == 0


class TestSessionRepeat:
    @pytest.mark.parametrize(("grade", "expected"), [(0, True), (3, True), (4, False), (5, False)])
    def test_repeat_rule(self, grade, expected):
        assert needs_repeat_in_session(grade) is expected


class TestValidation:
    @pytest.mark.parametrize("grade", [-1, 6, 100])
    def test_out_of_range_grade_is_rejected(self, grade):
        with pytest.raises(ValueError):
            next_schedule(new_schedule(), grade, today=TODAY)

    @pytest.mark.parametrize("grade", [True, 4.0, "4", None])
    def test_non_integer_grade_is_rejected(self, grade):
        with pytest.raises(TypeError):
            next_schedule(new_schedule(), grade, today=TODAY)

    def test_schedule_is_not_modified_in_place(self):
        original = new_schedule()
        next_schedule(original, 5, today=TODAY)
        assert original == new_schedule()
        with pytest.raises(FrozenInstanceError):
            original.ease = 3.0

    @pytest.mark.parametrize(
        "bad_settings",
        [
            {"min_ease": 0},
            {"min_ease": 3.0},
            {"first_interval_days": 0},
            {"second_interval_days": 0},
            {"pass_grade": 0},
            {"pass_grade": 6},
            {"pass_grade": 4, "repeat_below_grade": 3},
            {"max_interval_days": 3},
        ],
    )
    def test_bad_config_is_rejected(self, bad_settings):
        with pytest.raises(ValueError):
            SM2Config(**bad_settings)
