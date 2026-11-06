from datetime import datetime, timedelta

import numpy as np
import pytest

from calibrayn.models.calibration import Confidence
from calibrayn.models.documents import ChunkDraft
from calibrayn.models.learning import Attempt
from calibrayn.scheduler.sm2 import Schedule, new_schedule
from calibrayn.storage.database import Database
from calibrayn.storage.library import Library
from calibrayn.storage.progress import Progress

NOW = datetime(2026, 11, 2, 10, 0)


@pytest.fixture
def database():
    opened = Database(":memory:")
    yield opened
    opened.close()


@pytest.fixture
def library(database):
    return Library(database)


@pytest.fixture
def progress(database):
    return Progress(database)


def drafts(count):
    return [ChunkDraft(i, i + 1, f"Title {i}", f"Text of chunk {i}.") for i in range(count)]


def vectors(count, size=4):
    return np.arange(count * size, dtype=np.float32).reshape(count, size)


def attempt(chunk_id, score=1.0, level=3, confidence=Confidence.SURE, when=NOW):
    return Attempt(
        chunk_id=chunk_id,
        asked_at=when,
        level=level,
        confidence=confidence,
        score=score,
        correct=score >= 0.6,
        response_seconds=12.5,
        delayed_recall=False,
    )


class TestLibrary:
    def test_adding_a_document_returns_it_with_numbered_chunks(self, library):
        document, chunks = library.add_document("notes.pdf", "page", drafts(3), vectors(3))
        assert document.name == "notes.pdf"
        assert document.unit == "page"
        assert [chunk.position for chunk in chunks] == [0, 1, 2]
        assert len({chunk.id for chunk in chunks}) == 3
        assert all(chunk.document_id == document.id for chunk in chunks)

    def test_documents_are_listed_in_the_order_they_were_added(self, library):
        library.add_document("a.txt", "section", drafts(1), vectors(1))
        library.add_document("b.txt", "section", drafts(1), vectors(1))
        assert [d.name for d in library.documents()] == ["a.txt", "b.txt"]

    def test_chunks_come_back_in_reading_order(self, library):
        document, _ = library.add_document("a.txt", "section", drafts(4), vectors(4))
        assert [c.position for c in library.chunks(document.id)] == [0, 1, 2, 3]

    def test_chunks_can_be_filtered_by_document(self, library):
        first, _ = library.add_document("a.txt", "section", drafts(2), vectors(2))
        library.add_document("b.txt", "section", drafts(3), vectors(3))
        assert len(library.chunks(first.id)) == 2
        assert len(library.chunks()) == 5
        assert library.chunk_count() == 5

    def test_a_single_chunk_can_be_looked_up(self, library):
        _, chunks = library.add_document("a.txt", "section", drafts(2), vectors(2))
        assert library.chunk(chunks[1].id) == chunks[1]
        assert library.chunk(9999) is None

    def test_a_single_document_can_be_looked_up(self, library):
        document, _ = library.add_document("a.txt", "section", drafts(1), vectors(1))
        assert library.document(document.id) == document
        assert library.document(9999) is None

    def test_vectors_round_trip_as_float32(self, library):
        _, chunks = library.add_document("a.txt", "section", drafts(3), vectors(3))
        ids, matrix = library.vectors()
        assert ids == [chunk.id for chunk in chunks]
        assert matrix.dtype == np.float32
        assert np.array_equal(matrix, vectors(3))

    def test_vectors_of_an_empty_library_are_empty(self, library):
        ids, matrix = library.vectors()
        assert ids == []
        assert matrix.shape[0] == 0

    def test_vectors_can_be_replaced(self, library):
        _, chunks = library.add_document("a.txt", "section", drafts(2), vectors(2))
        replacement = np.ones((2, 6), dtype=np.float32)
        library.replace_vectors([c.id for c in chunks], replacement)
        _, matrix = library.vectors()
        assert matrix.shape == (2, 6)

    def test_deleting_a_document_removes_its_chunks_and_progress(self, library, progress):
        document, chunks = library.add_document("a.txt", "section", drafts(2), vectors(2))
        progress.save_schedule(chunks[0].id, new_schedule())
        progress.record_attempt(attempt(chunks[0].id))
        removed = library.delete_document(document.id)
        assert removed == [chunk.id for chunk in chunks]
        assert library.documents() == []
        assert library.chunk_count() == 0
        assert progress.schedules() == {}
        assert progress.attempt_count() == 0

    def test_a_failed_insert_stores_nothing(self, library):
        with pytest.raises(ValueError):
            library.add_document("a.txt", "section", drafts(3), vectors(2))
        assert library.documents() == []
        assert library.chunk_count() == 0

    def test_embedder_name_is_remembered(self, library):
        assert library.embedder_name() is None
        library.set_embedder_name("hashing-384")
        library.set_embedder_name("bge")
        assert library.embedder_name() == "bge"


class TestSchedules:
    def test_an_unknown_chunk_gets_a_new_schedule(self, progress):
        assert progress.schedule_for(5) == new_schedule()
        assert not progress.has_schedule(5)

    def test_a_schedule_round_trips(self, library, progress):
        _, chunks = library.add_document("a.txt", "section", drafts(1), vectors(1))
        schedule = Schedule(
            easiness=2.2,
            interval_days=6,
            repetitions=2,
            due_at=NOW + timedelta(days=6),
            last_reviewed_at=NOW,
        )
        progress.save_schedule(chunks[0].id, schedule)
        assert progress.schedule_for(chunks[0].id) == schedule
        assert progress.has_schedule(chunks[0].id)

    def test_saving_again_replaces_the_schedule(self, library, progress):
        _, chunks = library.add_document("a.txt", "section", drafts(1), vectors(1))
        progress.save_schedule(chunks[0].id, Schedule(easiness=2.0))
        progress.save_schedule(chunks[0].id, Schedule(easiness=1.5))
        assert progress.schedule_for(chunks[0].id).easiness == 1.5
        assert len(progress.schedules()) == 1

    def test_a_schedule_without_dates_round_trips(self, library, progress):
        _, chunks = library.add_document("a.txt", "section", drafts(1), vectors(1))
        progress.save_schedule(chunks[0].id, new_schedule())
        loaded = progress.schedule_for(chunks[0].id)
        assert loaded.due_at is None
        assert loaded.last_reviewed_at is None


class TestAttempts:
    def test_an_attempt_is_stored_with_an_id(self, library, progress):
        _, chunks = library.add_document("a.txt", "section", drafts(1), vectors(1))
        stored = progress.record_attempt(attempt(chunks[0].id, score=0.8))
        assert stored.id is not None
        assert progress.attempts() == [stored]

    def test_all_fields_survive_the_round_trip(self, library, progress):
        _, chunks = library.add_document("a.txt", "section", drafts(1), vectors(1))
        original = attempt(chunks[0].id, score=0.3, level=4, confidence=Confidence.GUESSING)
        stored = progress.record_attempt(original)
        loaded = progress.attempts()[0]
        assert loaded == stored
        assert loaded.confidence is Confidence.GUESSING
        assert loaded.correct is False
        assert loaded.level == 4

    def test_attempts_keep_their_order(self, library, progress):
        _, chunks = library.add_document("a.txt", "section", drafts(1), vectors(1))
        for score in (0.1, 0.5, 0.9):
            progress.record_attempt(attempt(chunks[0].id, score=score))
        assert [a.score for a in progress.attempts()] == [0.1, 0.5, 0.9]

    def test_a_limit_returns_the_latest_in_order(self, library, progress):
        _, chunks = library.add_document("a.txt", "section", drafts(1), vectors(1))
        for score in (0.1, 0.2, 0.3, 0.4):
            progress.record_attempt(attempt(chunks[0].id, score=score))
        assert [a.score for a in progress.attempts(limit=2)] == [0.3, 0.4]

    def test_attempts_are_counted(self, library, progress):
        _, chunks = library.add_document("a.txt", "section", drafts(1), vectors(1))
        assert progress.attempt_count() == 0
        progress.record_attempt(attempt(chunks[0].id))
        assert progress.attempt_count() == 1


class TestState:
    def test_missing_state_gives_the_default(self, progress):
        assert progress.get_state("level") is None
        assert progress.get_state("level", "3") == "3"

    def test_state_is_saved_as_text_and_replaced(self, progress):
        progress.set_state("level", 4)
        progress.set_state("level", 5)
        assert progress.get_state("level") == "5"


class TestRecalls:
    def test_a_recall_is_due_only_after_its_time(self, library, progress):
        _, chunks = library.add_document("a.txt", "section", drafts(2), vectors(2))
        progress.queue_recall(chunks[0].id, NOW + timedelta(minutes=30))
        assert progress.due_recalls(NOW) == []
        assert progress.due_recalls(NOW + timedelta(minutes=30)) == [chunks[0].id]
        assert progress.pending_recalls() == [chunks[0].id]

    def test_queueing_again_moves_the_time(self, library, progress):
        _, chunks = library.add_document("a.txt", "section", drafts(1), vectors(1))
        progress.queue_recall(chunks[0].id, NOW + timedelta(hours=2))
        progress.queue_recall(chunks[0].id, NOW)
        assert progress.due_recalls(NOW) == [chunks[0].id]
        assert len(progress.pending_recalls()) == 1

    def test_due_recalls_come_oldest_first(self, library, progress):
        _, chunks = library.add_document("a.txt", "section", drafts(2), vectors(2))
        progress.queue_recall(chunks[1].id, NOW - timedelta(minutes=5))
        progress.queue_recall(chunks[0].id, NOW - timedelta(minutes=1))
        assert progress.due_recalls(NOW) == [chunks[1].id, chunks[0].id]

    def test_a_recall_can_be_cleared(self, library, progress):
        _, chunks = library.add_document("a.txt", "section", drafts(1), vectors(1))
        progress.queue_recall(chunks[0].id, NOW)
        progress.clear_recall(chunks[0].id)
        assert progress.pending_recalls() == []
