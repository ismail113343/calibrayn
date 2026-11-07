from datetime import datetime, timedelta

import pytest

from calibrayn.config import ROOT, Settings
from calibrayn.evaluation.dataset import build_dataset
from calibrayn.evaluation.retrieval import evaluate_retrieval
from calibrayn.evaluation.simulation import Setup, run_personas
from calibrayn.models.calibration import CalibrationState, Confidence
from calibrayn.services.workspace import Workspace
from calibrayn.ui.components.samples import sample_files

START = datetime(2026, 11, 9, 9, 0)


class Clock:
    def __init__(self):
        self.moment = START

    def __call__(self):
        return self.moment

    def advance(self, **delta):
        self.moment += timedelta(**delta)


def open_workspace(tmp_path, clock):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        llm_provider="mock",
        embedder="hashing",
        settings_file=ROOT / "configs" / "settings.yaml",
        prompts_dir=ROOT / "configs" / "prompts",
    )
    return Workspace.open(settings, clock=clock)


def study(workspace, clock, answer, confidence=Confidence.SURE):
    asked = workspace.agent.ask()
    text = asked.question.answer_key if answer == "key" else answer
    outcome = workspace.agent.answer(asked, text, confidence, 8.0)
    clock.advance(minutes=15)
    return asked, outcome


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def workspace(tmp_path, clock):
    opened = open_workspace(tmp_path, clock)
    for name, content in sample_files():
        opened.add_document(name, content)
    yield opened
    opened.close()


def test_notes_become_topics_that_can_be_studied_and_measured(workspace, clock):
    assert workspace.library.chunk_count() == 5
    assert workspace.analytics.summary(clock()).due_now == 5

    for _ in range(6):
        study(workspace, clock, "bananas are yellow")
    for _ in range(6):
        study(workspace, clock, "key")

    summary = workspace.analytics.summary(clock())
    assert summary.attempts == 12
    assert summary.started == 5
    assert workspace.analytics.calibration_trend()
    assert workspace.analytics.weakest(limit=1)[0].accuracy < 1.0
    assert len(workspace.analytics.daily()) == 1


def test_search_and_questions_can_be_evaluated_on_the_same_notes(workspace):
    items = build_dataset(workspace.library, workspace.writer, 12, workspace.client.name)
    assert len(items) >= 5
    scores = evaluate_retrieval(workspace.retriever, items)
    assert scores.hit_at[5] == 1.0
    assert scores.hit_at[1] > 0.5


def test_the_simulation_runs_with_the_live_settings(workspace):
    results = run_personas(["overconfident"], 3, Setup(topics=10, days=21), workspace.tuning)
    assert results[0].persona == "overconfident"
    assert len(results[0].metrics) == 7


def test_repeated_confident_mistakes_bring_a_recall_check_with_no_help(workspace, clock):
    for _ in range(10):
        study(workspace, clock, "bananas are yellow")
    assert workspace.agent.meta.reading().state is CalibrationState.OVERCONFIDENT
    assert workspace.progress.pending_recalls()

    clock.advance(minutes=31)
    recall = workspace.agent.ask()
    assert recall.question.delayed_recall
    assert not recall.question.worked_example


def test_everything_survives_a_restart(tmp_path, clock):
    first = open_workspace(tmp_path, clock)
    for name, content in sample_files():
        first.add_document(name, content)
    for _ in range(8):
        study(first, clock, "key")
    level = first.agent.zpd.level
    reading = first.agent.meta.reading()
    first.close()

    second = open_workspace(tmp_path, clock)
    assert second.library.chunk_count() == 5
    assert second.progress.attempt_count() == 8
    assert second.agent.zpd.level == level
    assert second.agent.meta.reading() == reading
    assert second.chat("chlorophyll light").citations
    second.close()
