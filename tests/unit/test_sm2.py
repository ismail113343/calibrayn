from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from calibrayn.scheduler.sm2 import (
    DEFAULT_CONFIG,
    Schedule,
    SM2Config,
    days_overdue,
    easiness_adjustment,
    is_due,
    needs_repeat_in_session,
    new_schedule,
    next_schedule,
)

NOW = datetime(2026, 10, 19, 9, 30)


def run_reviews(grades, config=DEFAULT_CONFIG):
    """Review a topic each time it falls due, and return the schedule at the end."""
    schedule = new_schedule(config)
    moment = NOW
    for grade in grades:
        schedule = next_schedule(schedule, grade, now=moment, config=config)
        moment = schedule.due_at
    return schedule


class TestNewSchedule:
    def test_starts_with_default_easiness_and_no_history(self):
        schedule = new_schedule()
        assert schedule.easiness == 2.5
        assert schedule.interval_days == 0
        assert schedule.repetitions == 0
        assert schedule.due_at is None
        assert schedule.last_reviewed_at is None
        assert not schedule.is_relearning

    def test_uses_easiness_from_custom_config(self):
        assert new_schedule(SM2Config(initial_easiness=2.8)).easiness == 2.8


class TestPassingReviews:
    def test_first_pass_comes_back_the_next_day(self):
        schedule = next_schedule(new_schedule(), 4, now=NOW)
        assert schedule.interval_days == 1
        assert schedule.repetitions == 1
        assert schedule.due_at == datetime(2026, 10, 20)
        assert schedule.last_reviewed_at == NOW

    def test_second_pass_comes_back_six_days_later(self):
        schedule = run_reviews([4, 4])
        assert schedule.interval_days == 6
        assert schedule.repetitions == 2
        assert schedule.due_at == datetime(2026, 10, 26)

    def test_third_pass_multiplies_by_easiness(self):
        assert run_reviews([4, 4, 4]).interval_days == 15

    def test_interval_uses_easiness_from_before_the_review(self):
        # two grade-5 reviews lift easiness to 2.7, and the third review should use that 2.7
        assert run_reviews([5, 5, 5]).interval_days == 16

    def test_intervals_keep_growing_for_a_strong_learner(self):
        schedule = new_schedule()
        moment = NOW
        intervals = []
        for _ in range(10):
            schedule = next_schedule(schedule, 5, now=moment)
            intervals.append(schedule.interval_days)
            moment = schedule.due_at
        assert intervals[2:] == sorted(set(intervals[2:]))
        assert schedule.repetitions == 10

    def test_half_days_round_up(self):
        schedule = Schedule(easiness=2.5, interval_days=5, repetitions=2)
        assert next_schedule(schedule, 4, now=NOW).interval_days == 13

    def test_intervals_count_calendar_days(self):
        late = datetime(2026, 10, 19, 23, 59)
        assert next_schedule(new_schedule(), 4, now=late).due_at == datetime(2026, 10, 20)


class TestFailingReviews:
    def test_failed_topic_comes_back_in_minutes(self):
        schedule = next_schedule(new_schedule(), 1, now=NOW)
        assert schedule.due_at == NOW + timedelta(minutes=10)
        assert schedule.last_reviewed_at == NOW

    def test_relearn_delay_follows_the_config(self):
        schedule = next_schedule(new_schedule(), 0, now=NOW, config=SM2Config(relearn_minutes=5))
        assert schedule.due_at == NOW + timedelta(minutes=5)

    @pytest.mark.parametrize("grade", [0, 1, 2])
    def test_failure_restarts_the_cycle(self, grade):
        learned = Schedule(easiness=2.2, interval_days=15, repetitions=3)
        schedule = next_schedule(learned, grade, now=NOW)
        assert schedule.repetitions == 0
        assert schedule.interval_days == 0
        assert schedule.is_relearning

    @pytest.mark.parametrize("grade", [0, 1, 2])
    def test_failure_leaves_easiness_untouched(self, grade):
        learned = Schedule(easiness=2.2, interval_days=15, repetitions=3)
        assert next_schedule(learned, grade, now=NOW).easiness == 2.2

    def test_topic_is_not_due_until_the_delay_has_passed(self):
        schedule = next_schedule(new_schedule(), 1, now=NOW)
        assert not is_due(schedule, now=NOW + timedelta(minutes=9))
        assert is_due(schedule, now=NOW + timedelta(minutes=10))

    def test_passing_after_relearning_starts_the_sequence_again(self):
        learned = Schedule(easiness=2.2, interval_days=15, repetitions=3)
        failed = next_schedule(learned, 1, now=NOW)
        passed = next_schedule(failed, 4, now=failed.due_at)
        assert passed.repetitions == 1
        assert passed.interval_days == 1
        assert not passed.is_relearning

    def test_a_topic_can_recover_after_failing(self):
        schedule = run_reviews([4, 4, 4, 1, 4, 4])
        assert schedule.repetitions == 2
        assert schedule.interval_days == 6


class TestEasiness:
    @pytest.mark.parametrize(
        ("grade", "expected"),
        [(5, 0.10), (4, 0.0), (3, -0.14), (2, -0.32), (1, -0.54), (0, -0.80)],
    )
    def test_adjustment_matches_the_sm2_formula(self, grade, expected):
        assert easiness_adjustment(grade) == pytest.approx(expected)

    def test_hard_passes_lower_easiness(self):
        assert next_schedule(new_schedule(), 3, now=NOW).easiness == pytest.approx(2.36)

    def test_easiness_never_falls_below_the_floor(self):
        schedule = run_reviews([3] * 20)
        assert schedule.easiness == DEFAULT_CONFIG.min_easiness

    def test_floor_follows_a_custom_config(self):
        config = SM2Config(min_easiness=1.8)
        assert run_reviews([3] * 8, config).easiness == 1.8


class TestIntervalCap:
    def test_cap_limits_long_intervals(self):
        config = SM2Config(max_interval_days=20)
        assert run_reviews([5] * 8, config).interval_days == 20


class TestDueTimes:
    def test_new_topic_is_due(self):
        assert is_due(new_schedule(), now=NOW)

    def test_due_exactly_now_counts_as_due(self):
        assert is_due(Schedule(due_at=NOW), now=NOW)

    def test_future_topic_is_not_due(self):
        assert not is_due(Schedule(due_at=NOW + timedelta(days=1)), now=NOW)

    def test_overdue_topic_is_due(self):
        assert is_due(Schedule(due_at=NOW - timedelta(days=3)), now=NOW)

    def test_days_overdue_counts_whole_late_days(self):
        assert days_overdue(Schedule(due_at=NOW - timedelta(days=3)), now=NOW) == 3
        assert days_overdue(Schedule(due_at=NOW - timedelta(days=2, hours=5)), now=NOW) == 2

    def test_days_overdue_is_zero_when_not_a_full_day_late(self):
        assert days_overdue(Schedule(due_at=NOW - timedelta(hours=5)), now=NOW) == 0
        assert days_overdue(Schedule(due_at=NOW + timedelta(days=2)), now=NOW) == 0
        assert days_overdue(new_schedule(), now=NOW) == 0


class TestTimezones:
    def test_aware_datetimes_keep_their_timezone(self):
        moment = datetime(2026, 10, 19, 9, 30, tzinfo=timezone.utc)
        passed = next_schedule(new_schedule(), 4, now=moment)
        assert passed.due_at == datetime(2026, 10, 20, tzinfo=timezone.utc)
        failed = next_schedule(new_schedule(), 1, now=moment)
        assert failed.due_at == moment + timedelta(minutes=10)
        assert failed.due_at.tzinfo is timezone.utc


class TestSessionRepeat:
    @pytest.mark.parametrize(("grade", "expected"), [(0, True), (3, True), (4, False), (5, False)])
    def test_repeat_rule(self, grade, expected):
        assert needs_repeat_in_session(grade) is expected


class TestValidation:
    @pytest.mark.parametrize("grade", [-1, 6, 100])
    def test_out_of_range_grade_is_rejected(self, grade):
        with pytest.raises(ValueError):
            next_schedule(new_schedule(), grade, now=NOW)

    @pytest.mark.parametrize("grade", [True, 4.0, "4", None])
    def test_non_integer_grade_is_rejected(self, grade):
        with pytest.raises(TypeError):
            next_schedule(new_schedule(), grade, now=NOW)

    def test_schedule_is_not_modified_in_place(self):
        original = new_schedule()
        next_schedule(original, 5, now=NOW)
        assert original == new_schedule()
        with pytest.raises(FrozenInstanceError):
            original.easiness = 3.0

    @pytest.mark.parametrize(
        "bad_settings",
        [
            {"min_easiness": 0},
            {"min_easiness": 3.0},
            {"first_interval_days": 0},
            {"second_interval_days": 0},
            {"pass_grade": 0},
            {"pass_grade": 6},
            {"pass_grade": 4, "repeat_below_grade": 3},
            {"relearn_minutes": 0},
            {"max_interval_days": 3},
        ],
    )
    def test_bad_config_is_rejected(self, bad_settings):
        with pytest.raises(ValueError):
            SM2Config(**bad_settings)
