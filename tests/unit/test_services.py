from datetime import datetime, timedelta

import numpy as np
import pytest

from calibrayn.config import ROOT, Settings, Tuning
from calibrayn.llm.base import LLMError, PrivacyError
from calibrayn.llm.mock import MockClient
from calibrayn.models.calibration import Confidence
from calibrayn.models.documents import Chunk, ChunkDraft
from calibrayn.models.learning import Attempt
from calibrayn.rag.prompts import PromptBook
from calibrayn.scheduler.sm2 import Schedule
from calibrayn.services.analytics import Analytics
from calibrayn.services.grading import Grader, keyword_verdict, read_score
from calibrayn.services.workspace import Workspace
from calibrayn.storage.database import Database
from calibrayn.storage.library import Library
from calibrayn.storage.progress import Progress

KEY = "Photosynthesis turns light into chemical energy. Chlorophyll absorbs the light."
CHUNK = Chunk(1, 1, 0, 1, "Photosynthesis", KEY)
PROMPTS = PromptBook(ROOT / "configs" / "prompts")


class FixedReply:
    name = "fixed"
    hosted = False

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error

    def complete(self, system, prompt, max_tokens=400, temperature=0.2):
        if self.error:
            raise self.error
        return self.reply


def grader_with(reply=None, error=None):
    return Grader(FixedReply(reply, error), PROMPTS)


class TestReadScore:
    @pytest.mark.parametrize(
        ("parsed", "expected"),
        [
            ({"score": 0.5}, 0.5),
            ({"score": 1}, 1.0),
            ({"score": "0.25"}, 0.25),
            ({"score": 7}, 1.0),
            ({"score": -2}, 0.0),
        ],
    )
    def test_numbers_are_read_and_clamped(self, parsed, expected):
        assert read_score(parsed) == expected

    @pytest.mark.parametrize(
        "parsed",
        [None, {}, {"score": None}, {"score": True}, {"score": "high"}, {"score": float("nan")}],
    )
    def test_anything_else_is_not_a_score(self, parsed):
        assert read_score(parsed) is None


class TestGrader:
    def test_the_model_marks_the_answer(self):
        verdict = Grader(MockClient(), PROMPTS).grade("Q?", KEY, KEY, CHUNK)
        assert verdict.score == 1.0
        assert verdict.marked_by == "model"
        assert verdict.feedback

    def test_a_blank_answer_scores_zero_without_asking_the_model(self):
        verdict = grader_with(error=LLMError("should not be called")).grade("Q?", KEY, "  ", CHUNK)
        assert (verdict.score, verdict.marked_by) == (0.0, "rule")

    def test_a_model_failure_falls_back_to_key_words(self):
        verdict = grader_with(error=LLMError("down")).grade("Q?", KEY, KEY, CHUNK)
        assert verdict.marked_by == "keywords"
        assert verdict.score == 1.0
        assert "model was not available" in verdict.feedback

    def test_a_reply_that_is_not_json_falls_back(self):
        assert grader_with("great answer!").grade("Q?", KEY, KEY, CHUNK).marked_by == "keywords"

    def test_a_reply_without_a_usable_score_falls_back(self):
        verdict = grader_with('{"score": "high"}').grade("Q?", KEY, KEY, CHUNK)
        assert verdict.marked_by == "keywords"

    def test_missing_feedback_gets_a_default(self):
        verdict = grader_with('{"score": 0.7}').grade("Q?", KEY, KEY, CHUNK)
        assert verdict.score == 0.7
        assert verdict.feedback == "Marked against your notes."

    def test_the_prompt_carries_the_question_key_and_answer(self):
        seen = {}

        class Spy(FixedReply):
            def complete(self, system, prompt, max_tokens=400, temperature=0.2):
                seen["prompt"] = prompt
                return '{"score": 0.5, "feedback": "ok"}'

        Grader(Spy(), PROMPTS).grade("What is it?", "the key", "my answer", CHUNK)
        assert "What is it?" in seen["prompt"]
        assert "the key" in seen["prompt"]
        assert "my answer" in seen["prompt"]

    def test_keyword_verdict_without_a_key_scores_zero(self):
        assert keyword_verdict("", "anything").score == 0.0

    def test_keyword_verdict_lists_missing_words(self):
        verdict = keyword_verdict(KEY, "light")
        assert verdict.score < 0.9
        assert "Try to include" in verdict.feedback


NOW = datetime(2026, 11, 3, 12, 0)


def attempt(chunk_id, score, day=0, level=3, confidence=Confidence.SURE):
    return Attempt(
        chunk_id=chunk_id,
        asked_at=NOW + timedelta(days=day),
        level=level,
        confidence=confidence,
        score=score,
        correct=score >= 0.6,
        response_seconds=10.0,
        delayed_recall=False,
    )


@pytest.fixture
def stores():
    database = Database(":memory:")
    library = Library(database)
    drafts = [ChunkDraft(i, 1, f"Topic {i}", f"Text {i}.") for i in range(3)]
    _, chunks = library.add_document("notes.txt", "section", drafts, np.zeros((3, 4)))
    yield library, Progress(database), chunks, Analytics(library, Progress(database), Tuning())
    database.close()


class TestAnalytics:
    def test_an_empty_summary(self, stores):
        _, _, _, analytics = stores
        summary = analytics.summary(NOW)
        assert (summary.documents, summary.topics, summary.started) == (1, 3, 0)
        assert summary.due_now == 3
        assert summary.attempts == 0
        assert summary.accuracy is None

    def test_the_summary_counts_attempts_and_due_topics(self, stores):
        library, progress, chunks, analytics = stores
        progress.record_attempt(attempt(chunks[0].id, 1.0))
        progress.record_attempt(attempt(chunks[0].id, 0.5))
        progress.save_schedule(
            chunks[0].id, Schedule(due_at=NOW + timedelta(days=2), repetitions=1)
        )
        summary = analytics.summary(NOW)
        assert summary.attempts == 2
        assert summary.accuracy == pytest.approx(0.75)
        assert summary.started == 1
        assert summary.due_now == 2

    def test_daily_points_group_attempts_by_day(self, stores):
        _, progress, chunks, analytics = stores
        for score, day in ((1.0, 0), (0.5, 0), (0.0, 2)):
            progress.record_attempt(attempt(chunks[0].id, score, day=day))
        days = analytics.daily()
        assert [(point.attempts, point.accuracy) for point in days] == [(2, 0.75), (1, 0.0)]
        assert days[0].day == NOW.date()

    def test_the_calibration_trend_starts_once_there_are_enough_ratings(self, stores):
        _, progress, chunks, analytics = stores
        for _ in range(12):
            progress.record_attempt(attempt(chunks[0].id, 1.0))
        trend = analytics.calibration_trend()
        assert [point.number for point in trend] == [10, 11, 12]
        assert all(point.score == 1.0 and point.bias == 0.0 for point in trend)

    def test_the_trend_follows_overconfidence(self, stores):
        _, progress, chunks, analytics = stores
        for _ in range(10):
            progress.record_attempt(attempt(chunks[0].id, 0.0))
        assert analytics.calibration_trend()[-1].bias == 1.0

    def test_the_level_history_numbers_each_attempt(self, stores):
        _, progress, chunks, analytics = stores
        progress.record_attempt(attempt(chunks[0].id, 1.0, level=3))
        progress.record_attempt(attempt(chunks[0].id, 1.0, level=4))
        assert analytics.level_history() == [(1, 3), (2, 4)]

    def test_the_weakest_topics_come_first(self, stores):
        _, progress, chunks, analytics = stores
        for score in (1.0, 1.0):
            progress.record_attempt(attempt(chunks[0].id, score))
        for score in (0.0, 0.5):
            progress.record_attempt(attempt(chunks[1].id, score))
        weakest = analytics.weakest(limit=2)
        assert [topic.chunk_id for topic in weakest] == [chunks[1].id, chunks[0].id]
        assert weakest[0].accuracy == pytest.approx(0.25)
        assert weakest[0].document == "notes.txt"
        assert weakest[0].attempts == 2

    def test_the_weakest_list_respects_the_limit(self, stores):
        _, progress, chunks, analytics = stores
        for chunk in chunks:
            progress.record_attempt(attempt(chunk.id, 0.5))
        assert len(analytics.weakest(limit=2)) == 2


NOTES = (
    "Photosynthesis turns light into chemical energy. Chlorophyll absorbs the light. "
    "The process makes glucose and oxygen."
)
HISTORY = "The Roman empire built long roads. Roads helped armies move quickly."


def settings_for(tmp_path, **changes):
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        settings_file=ROOT / "configs" / "settings.yaml",
        prompts_dir=ROOT / "configs" / "prompts",
        **changes,
    )


@pytest.fixture
def workspace(tmp_path):
    opened = Workspace.open(settings_for(tmp_path))
    yield opened
    opened.close()


class TestWorkspace:
    def test_a_file_can_be_added_and_is_remembered(self, workspace):
        document, chunks = workspace.add_document("biology.txt", NOTES.encode())
        assert document.name == "biology.txt"
        assert len(chunks) == 1
        assert workspace.library.chunk_count() == 1

    def test_chat_answers_from_the_notes_with_citations(self, workspace):
        workspace.add_document("biology.txt", NOTES.encode())
        reply = workspace.chat("What does chlorophyll absorb?")
        assert "[1]" in reply.text
        assert reply.citations == ("biology.txt, section 1",)
        assert reply.grounded
        assert reply.support > 0.5

    def test_chat_picks_the_right_document(self, workspace):
        workspace.add_document("biology.txt", NOTES.encode())
        workspace.add_document("history.txt", HISTORY.encode())
        reply = workspace.chat("Who built long roads for armies?")
        assert "Roman" in reply.text

    def test_chat_with_no_notes_says_so(self, workspace):
        reply = workspace.chat("anything")
        assert reply.citations == ()
        assert not reply.grounded
        assert "could not find" in reply.text

    def test_chat_does_not_show_sources_when_the_notes_have_no_answer(self, workspace):
        workspace.add_document("biology.txt", NOTES.encode())
        reply = workspace.chat("quantum gravity")
        assert "could not find" in reply.text
        assert reply.citations == ()
        assert not reply.grounded

    def test_chat_flags_an_answer_the_notes_do_not_support(self, tmp_path):
        class Inventor(FixedReply):
            def complete(self, system, prompt, max_tokens=400, temperature=0.2):
                return "Dragons guard golden treasure beneath volcanic mountains [1]"

        opened = Workspace.open(settings_for(tmp_path), client=Inventor())
        opened.add_document("biology.txt", NOTES.encode())
        reply = opened.chat("chlorophyll light")
        assert not reply.grounded
        assert reply.support == 0.0
        opened.close()

    def test_a_lesson_explains_the_topic_and_gives_an_example(self, workspace):
        _, chunks = workspace.add_document("biology.txt", NOTES.encode())
        lesson = workspace.lesson(chunks[0].id)
        assert lesson.explanation.startswith("In short:")
        assert lesson.worked_example.startswith("Worked example")
        assert lesson.citation == "biology.txt, section 1"
        assert lesson.title.startswith("Photosynthesis")

    def test_a_lesson_for_an_unknown_topic_is_an_error(self, workspace):
        with pytest.raises(KeyError):
            workspace.lesson(999)

    def test_notes_survive_closing_and_reopening(self, tmp_path):
        first = Workspace.open(settings_for(tmp_path))
        first.add_document("biology.txt", NOTES.encode())
        first.close()
        second = Workspace.open(settings_for(tmp_path))
        assert [d.name for d in second.library.documents()] == ["biology.txt"]
        assert "Chlorophyll" in second.chat("chlorophyll light").text
        second.close()

    def test_removing_a_document_clears_notes_index_and_progress(self, workspace):
        document, chunks = workspace.add_document("biology.txt", NOTES.encode())
        asked = workspace.agent.ask()
        workspace.agent.answer(asked, asked.question.answer_key, Confidence.SURE, 5.0)
        assert workspace.remove_document(document.id) == len(chunks)
        assert workspace.library.documents() == []
        assert len(workspace.retriever.index) == 0
        assert workspace.progress.attempt_count() == 0
        assert workspace.agent.meta.ratings == ()

    def test_a_full_study_round_works_through_the_workspace(self, workspace):
        workspace.add_document("biology.txt", NOTES.encode())
        asked = workspace.agent.ask()
        outcome = workspace.agent.answer(asked, asked.question.answer_key, Confidence.SURE, 8.0)
        assert outcome.correct
        assert workspace.analytics.summary(workspace.clock()).attempts == 1

    def test_the_privacy_setting_blocks_hosted_models(self, tmp_path):
        chosen = settings_for(tmp_path, llm_provider="hf_inference", hf_token="abc")
        with pytest.raises(PrivacyError):
            Workspace.open(chosen)

    def test_a_bad_embedder_choice_is_reported(self, tmp_path):
        with pytest.raises(ValueError, match="EMBEDDER"):
            Workspace.open(settings_for(tmp_path, embedder="magic"))

    def test_a_bad_index_choice_is_reported(self, tmp_path):
        with pytest.raises(ValueError, match="INDEX"):
            Workspace.open(settings_for(tmp_path, index="magic"))
