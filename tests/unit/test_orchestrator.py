from datetime import datetime, timedelta

import numpy as np
import pytest

from calibrayn.config import ROOT, Tuning
from calibrayn.llm.mock import MockClient
from calibrayn.models.calibration import CalibrationState, Confidence
from calibrayn.models.documents import ChunkDraft
from calibrayn.models.learning import Support
from calibrayn.orchestrator.agent import NothingToStudy, StaleQuestion, StudyAgent, grade_for
from calibrayn.orchestrator.cognitive_load import LoadEstimate, LoadLevel
from calibrayn.orchestrator.metacognition import Intervention
from calibrayn.orchestrator.policy import AgentConfig, make_plan
from calibrayn.orchestrator.selector import choose_chunk
from calibrayn.orchestrator.zpd import Zone
from calibrayn.rag.prompts import PromptBook
from calibrayn.rag.writer import Writer
from calibrayn.scheduler.sm2 import Schedule
from calibrayn.services.grading import Grader
from calibrayn.storage.database import Database
from calibrayn.storage.library import Library
from calibrayn.storage.progress import Progress

NOW = datetime(2026, 11, 3, 9, 0)
SURE = Confidence.SURE
UNSURE = Confidence.UNSURE
GUESSING = Confidence.GUESSING


def load(level, signal=0.1):
    return LoadEstimate(level, 0.5, 0.5, None, signal, 10)


def plan_for(**changes):
    values = {
        "level": 3,
        "zone": Zone.IN_ZONE,
        "load": load(LoadLevel.MODERATE),
        "intervention": Intervention.NONE,
        "recall_due": False,
        "max_level": 5,
    }
    values.update(changes)
    return make_plan(**values)


class TestPolicy:
    def test_steady_results_keep_the_plan(self):
        plan = plan_for()
        assert plan.level == 3
        assert plan.support is Support.DIRECT_RECALL
        assert not plan.delayed_recall
        assert not plan.suggest_break
        assert "steady" in plan.reasons[0]

    def test_high_load_brings_a_worked_example(self):
        plan = plan_for(load=load(LoadLevel.HIGH))
        assert plan.support is Support.WORKED_EXAMPLE
        assert "demanding" in plan.reasons[0]

    def test_a_too_hard_zone_brings_a_worked_example(self):
        plan = plan_for(zone=Zone.TOO_HARD)
        assert plan.support is Support.WORKED_EXAMPLE

    def test_under_confidence_raises_the_level_and_removes_help(self):
        plan = plan_for(intervention=Intervention.REDUCE_SUPPORT, zone=Zone.TOO_HARD)
        assert plan.level == 4
        assert plan.support is Support.DIRECT_RECALL

    def test_the_level_never_goes_above_the_top(self):
        plan = plan_for(level=5, intervention=Intervention.REDUCE_SUPPORT)
        assert plan.level == 5

    def test_a_student_under_heavy_load_keeps_their_help(self):
        plan = plan_for(intervention=Intervention.REDUCE_SUPPORT, load=load(LoadLevel.HIGH))
        assert plan.level == 3
        assert plan.support is Support.WORKED_EXAMPLE

    def test_a_due_recall_is_asked_with_no_help_even_under_heavy_load(self):
        plan = plan_for(recall_due=True, load=load(LoadLevel.HIGH))
        assert plan.delayed_recall
        assert plan.support is Support.DIRECT_RECALL
        assert any("no help" in reason for reason in plan.reasons)

    def test_a_break_is_suggested_when_load_is_high_late_in_a_session(self):
        assert plan_for(load=load(LoadLevel.HIGH, signal=0.6)).suggest_break

    def test_no_break_early_in_a_session_or_under_light_load(self):
        assert not plan_for(load=load(LoadLevel.HIGH, signal=0.2)).suggest_break
        assert not plan_for(load=load(LoadLevel.MODERATE, signal=0.9)).suggest_break

    def test_a_missing_session_signal_counts_as_zero(self):
        unknown = LoadEstimate(LoadLevel.UNKNOWN, None, None, None, None, 0)
        assert not plan_for(load=unknown).suggest_break

    @pytest.mark.parametrize(
        "bad_settings",
        [
            {"correct_from": 0.4},
            {"correct_from": 0.8},
            {"delayed_recall_minutes": 0},
            {"break_session_signal": 1.5},
            {"guessing_grade_cap": 6},
            {"guessing_grade_cap": -1},
        ],
    )
    def test_bad_config_is_rejected(self, bad_settings):
        with pytest.raises(ValueError):
            AgentConfig(**bad_settings)


def due(minutes_ago, **changes):
    return Schedule(due_at=NOW - timedelta(minutes=minutes_ago), repetitions=2, **changes)


class TestSelector:
    def test_nothing_to_choose_from(self):
        assert choose_chunk([], {}, NOW) is None

    def test_new_topics_come_in_reading_order(self):
        assert choose_chunk([7, 8, 9], {}, NOW) == 7

    def test_a_recall_comes_first(self):
        assert choose_chunk([1, 2, 3], {1: due(500)}, NOW, recalls=[3]) == 3

    def test_a_recall_for_a_removed_topic_is_ignored(self):
        assert choose_chunk([1, 2], {}, NOW, recalls=[99]) == 1

    def test_the_most_overdue_topic_comes_first(self):
        schedules = {1: due(10), 2: due(300), 3: due(60)}
        assert choose_chunk([1, 2, 3], schedules, NOW) == 2

    def test_a_topic_being_relearned_beats_older_due_topics(self):
        relearning = Schedule(
            repetitions=0, due_at=NOW - timedelta(minutes=1), last_reviewed_at=NOW
        )
        assert choose_chunk([1, 2], {1: due(5000), 2: relearning}, NOW) == 2

    def test_due_topics_come_before_new_ones(self):
        assert choose_chunk([1, 2], {2: due(5)}, NOW) == 2

    def test_a_topic_not_yet_due_waits_for_new_ones(self):
        later = Schedule(repetitions=1, due_at=NOW + timedelta(days=1))
        assert choose_chunk([1, 2], {1: later}, NOW) == 2

    def test_when_nothing_is_due_the_soonest_topic_is_studied_ahead(self):
        soon = Schedule(repetitions=1, due_at=NOW + timedelta(hours=1))
        later = Schedule(repetitions=1, due_at=NOW + timedelta(days=3))
        assert choose_chunk([1, 2], {1: later, 2: soon}, NOW) == 2

    def test_ties_follow_reading_order(self):
        schedules = {1: due(30), 2: due(30)}
        assert choose_chunk([2, 1], schedules, NOW) == 2


class TestGradeFor:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (1.0, 5),
            (0.9, 5),
            (0.8, 4),
            (0.75, 4),
            (0.6, 3),
            (0.59, 2),
            (0.4, 2),
            (0.2, 1),
            (0.1, 0),
        ],
    )
    def test_scores_map_to_sm2_grades(self, score, expected):
        assert grade_for(score, SURE, AgentConfig()) == expected

    def test_guessing_caps_the_grade(self):
        assert grade_for(1.0, GUESSING, AgentConfig()) == 3
        assert grade_for(0.5, GUESSING, AgentConfig()) == 2

    def test_unsure_does_not_cap_the_grade(self):
        assert grade_for(1.0, UNSURE, AgentConfig()) == 5

    def test_the_cap_follows_the_config(self):
        assert grade_for(1.0, GUESSING, AgentConfig(guessing_grade_cap=4)) == 4


TEXTS = [
    "Photosynthesis turns light into chemical energy. Chlorophyll absorbs the light. "
    "The process makes glucose and oxygen.",
    "Mitochondria release energy from glucose. This process is called respiration. "
    "It happens in every living cell.",
    "The Roman empire built long roads. Roads helped armies move quickly. "
    "Trade also grew along them.",
]


class Clock:
    def __init__(self, start=NOW):
        self.moment = start

    def __call__(self):
        return self.moment

    def advance(self, **delta):
        self.moment += timedelta(**delta)
        return self.moment


@pytest.fixture
def parts():
    database = Database(":memory:")
    library = Library(database)
    drafts = [ChunkDraft(i, i + 1, f"Topic {i}", text) for i, text in enumerate(TEXTS)]
    library.add_document("notes.txt", "section", drafts, np.zeros((3, 4)))
    prompts = PromptBook(ROOT / "configs" / "prompts")
    client = MockClient()
    yield {
        "database": database,
        "library": library,
        "progress": Progress(database),
        "writer": Writer(client, prompts),
        "grader": Grader(client, prompts),
    }
    database.close()


def make_agent(parts, clock):
    return StudyAgent(
        parts["library"], parts["progress"], parts["writer"], parts["grader"], Tuning(), clock=clock
    )


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def agent(parts, clock):
    return make_agent(parts, clock)


def study(agent, clock, answer="key", confidence=SURE, seconds=12.0, step=None):
    asked = agent.ask()
    text = asked.question.answer_key if answer == "key" else answer
    outcome = agent.answer(asked, text, confidence, seconds)
    clock.advance(**(step or {"minutes": 15}))
    return asked, outcome


class TestAgent:
    def test_an_empty_library_has_nothing_to_study(self, parts, clock):
        empty = Database(":memory:")
        agent = StudyAgent(
            Library(empty), Progress(empty), parts["writer"], parts["grader"], Tuning(), clock=clock
        )
        with pytest.raises(NothingToStudy):
            agent.ask()

    def test_the_first_question_is_a_new_topic_at_the_starting_level(self, agent, parts):
        asked = agent.ask()
        first = parts["library"].chunks()[0]
        assert asked.chunk.id == first.id
        assert asked.question.level == 3
        assert asked.question.support is Support.DIRECT_RECALL
        assert asked.question.worked_example == ""
        assert asked.question.text
        assert "steady" in asked.plan.reasons[0]

    def test_a_perfect_answer_schedules_the_topic_for_tomorrow(self, agent, parts, clock):
        asked, outcome = study(agent, clock)
        assert outcome.score == 1.0
        assert outcome.correct
        assert outcome.grade == 5
        assert outcome.marked_by == "model"
        assert outcome.schedule.repetitions == 1
        assert outcome.schedule.due_at == datetime(2026, 11, 4)
        stored = parts["progress"].attempts()
        assert len(stored) == 1
        assert stored[0].level == 3
        assert stored[0].confidence is SURE
        assert outcome.calibration.state is CalibrationState.INSUFFICIENT_DATA

    def test_a_wrong_answer_comes_back_in_minutes(self, agent, clock):
        _, outcome = study(agent, clock, answer="bananas are yellow")
        assert not outcome.correct
        assert outcome.grade == 0
        assert outcome.schedule.is_relearning
        assert outcome.schedule.due_at == NOW + timedelta(minutes=10)
        assert outcome.schedule.easiness == 2.5

    def test_a_blank_answer_is_marked_by_rule(self, agent, clock):
        _, outcome = study(agent, clock, answer="   ")
        assert outcome.score == 0.0
        assert outcome.marked_by == "rule"

    def test_a_guessed_right_answer_counts_as_hard_recall(self, agent, clock):
        _, outcome = study(agent, clock, confidence=GUESSING)
        assert outcome.grade == 3
        assert outcome.schedule.easiness == pytest.approx(2.36)

    def test_a_topic_being_relearned_returns_before_new_ones(self, agent, clock):
        first, _ = study(agent, clock, answer="bananas", step={"minutes": 11})
        again = agent.ask()
        assert again.chunk.id == first.chunk.id

    def test_before_the_delay_the_next_new_topic_is_used(self, agent, clock):
        first, _ = study(agent, clock, answer="bananas", step={"minutes": 2})
        other = agent.ask()
        assert other.chunk.id != first.chunk.id

    def test_a_streak_of_wins_raises_the_level_and_remembers_it(self, agent, parts, clock):
        for _ in range(10):
            study(agent, clock, step={"days": 1})
        assert agent.zpd.level == 5
        assert parts["progress"].get_state("level") == "5"

    def test_a_restart_restores_level_and_histories(self, agent, parts, clock):
        for _ in range(10):
            study(agent, clock, step={"days": 1})
        reopened = make_agent(parts, clock)
        assert reopened.zpd.level == agent.zpd.level
        assert reopened.meta.reading() == agent.meta.reading()
        assert reopened.load.attempts == agent.load.attempts
        assert reopened.zpd.scores == agent.zpd.scores

    def test_repeated_confident_errors_trigger_a_delayed_recall(self, agent, parts, clock):
        outcomes = []
        for _ in range(10):
            asked, outcome = study(agent, clock, answer="bananas", confidence=SURE)
            outcomes.append(outcome)
        last = outcomes[-1]
        assert last.calibration.state is CalibrationState.OVERCONFIDENT
        assert last.queued_recall
        assert last.schedule.easiness == pytest.approx(2.5 - 0.35)
        assert len(parts["progress"].pending_recalls()) == 1

        clock.advance(minutes=31)
        recall = agent.ask()
        assert recall.question.delayed_recall
        assert recall.question.support is Support.DIRECT_RECALL
        assert any("no help" in reason for reason in recall.plan.reasons)

        agent.answer(recall, recall.question.answer_key, SURE, 10.0)
        assert parts["progress"].pending_recalls() == []

    def test_a_wrong_recall_is_queued_again(self, agent, parts, clock):
        for _ in range(10):
            study(agent, clock, answer="bananas")
        clock.advance(minutes=31)
        recall = agent.ask()
        outcome = agent.answer(recall, "bananas", SURE, 10.0)
        assert outcome.queued_recall
        assert len(parts["progress"].pending_recalls()) == 1

    def test_heavy_load_brings_a_worked_example(self, agent, clock):
        for _ in range(6):
            study(agent, clock, answer="bananas")
        asked = agent.ask()
        assert asked.question.support is Support.WORKED_EXAMPLE
        assert asked.question.worked_example.startswith("Worked example")

    def test_a_long_hard_session_suggests_a_break(self, agent, clock):
        for _ in range(6):
            study(agent, clock, answer="bananas")
        clock.advance(minutes=45)
        assert agent.ask().plan.suggest_break

    def test_starting_a_new_session_resets_the_clock(self, agent, clock):
        clock.advance(minutes=50)
        assert agent.session_minutes() == pytest.approx(50)
        agent.start_session()
        assert agent.session_minutes() == 0

    def test_the_snapshot_shows_where_things_stand(self, agent, clock):
        before = agent.snapshot()
        assert before.due_now == 3
        assert before.level == 3
        assert before.calibration.state is CalibrationState.INSUFFICIENT_DATA
        study(agent, clock, answer="bananas", step={"minutes": 1})
        after = agent.snapshot()
        assert after.due_now == 2
        assert after.pending_recalls == 0

    def test_the_stored_level_is_kept_inside_the_allowed_range(self, agent, parts, clock):
        parts["progress"].set_state("level", 99)
        assert make_agent(parts, clock).zpd.level == 5


class TestStaleQuestions:
    def test_a_question_whose_notes_were_removed_is_refused(self, agent, parts):
        asked = agent.ask()
        parts["library"].delete_document(asked.chunk.document_id)
        with pytest.raises(StaleQuestion, match="were removed"):
            agent.answer(asked, "anything", SURE, 5.0)
        assert parts["progress"].attempt_count() == 0
        assert parts["progress"].schedules() == {}

    def test_after_a_refusal_a_new_question_can_be_asked(self, agent, parts):
        asked = agent.ask()
        document_id = asked.chunk.document_id
        parts["library"].delete_document(document_id)
        drafts = [ChunkDraft(0, 1, "Fresh topic", TEXTS[0])]
        parts["library"].add_document("again.txt", "section", drafts, np.zeros((1, 4)))
        fresh = agent.ask()
        outcome = agent.answer(fresh, fresh.question.answer_key, SURE, 5.0)
        assert outcome.correct
