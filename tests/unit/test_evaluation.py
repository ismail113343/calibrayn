import csv
import json
import random
from dataclasses import replace

import numpy as np
import pytest

from calibrayn.config import ROOT, Settings
from calibrayn.evaluation import report
from calibrayn.evaluation.cli import main
from calibrayn.evaluation.dataset import (
    DatasetError,
    QAItem,
    build_dataset,
    load_dataset,
    save_dataset,
)
from calibrayn.evaluation.personas import PERSONAS, Persona
from calibrayn.evaluation.retrieval import evaluate_retrieval
from calibrayn.evaluation.simulation import (
    ADJUSTED,
    METRICS,
    PLAIN,
    Setup,
    compare,
    confidence_from,
    grade_from,
    recall_probability,
    run_personas,
    simulate,
    sweep,
)
from calibrayn.evaluation.stats import bootstrap_interval
from calibrayn.ingestion.pipeline import ingest
from calibrayn.llm.mock import MockClient
from calibrayn.models.calibration import Confidence
from calibrayn.models.documents import ChunkDraft
from calibrayn.rag.embeddings import HashingEmbedder
from calibrayn.rag.index import NumpyIndex
from calibrayn.rag.prompts import PromptBook
from calibrayn.rag.retriever import Retriever
from calibrayn.rag.writer import GeneratedQuestion, Writer
from calibrayn.services.workspace import Workspace
from calibrayn.storage.database import Database
from calibrayn.storage.library import Library


class TestBootstrapInterval:
    def test_the_interval_surrounds_the_mean(self):
        interval = bootstrap_interval([1.0, 2.0, 3.0, 4.0, 5.0])
        assert interval.mean == 3.0
        assert interval.low <= interval.mean <= interval.high

    def test_the_same_seed_gives_the_same_interval(self):
        values = [0.1, 0.4, 0.2, 0.9, 0.5]
        assert bootstrap_interval(values, seed=3) == bootstrap_interval(values, seed=3)

    def test_identical_values_give_a_zero_width_interval(self):
        interval = bootstrap_interval([2.0, 2.0, 2.0])
        assert interval.low == interval.mean == interval.high == 2.0

    def test_one_value_is_its_own_interval(self):
        interval = bootstrap_interval([7.0])
        assert (interval.low, interval.mean, interval.high) == (7.0, 7.0, 7.0)

    def test_more_spread_gives_a_wider_interval(self):
        narrow = bootstrap_interval([5.0, 5.1, 4.9, 5.0, 5.05])
        wide = bootstrap_interval([1.0, 9.0, 2.0, 8.0, 5.0])
        assert wide.high - wide.low > narrow.high - narrow.low

    def test_no_values_are_refused(self):
        with pytest.raises(ValueError):
            bootstrap_interval([])

    @pytest.mark.parametrize("level", [0.0, 1.0, 1.5])
    def test_the_level_must_be_between_zero_and_one(self, level):
        with pytest.raises(ValueError):
            bootstrap_interval([1.0, 2.0], level=level)


class TestPersonas:
    def test_the_four_learners_exist(self):
        assert set(PERSONAS) == {"crammer", "steady", "overconfident", "underconfident"}

    def test_each_persona_has_a_summary(self):
        assert all(persona.summary for persona in PERSONAS.values())

    def test_confidence_bias_points_the_right_way(self):
        assert PERSONAS["overconfident"].confidence_bias > 0
        assert PERSONAS["underconfident"].confidence_bias < 0
        assert PERSONAS["steady"].confidence_bias == 0

    def test_the_crammer_studies_once_a_week(self):
        assert PERSONAS["crammer"].study_days == (6,)

    @pytest.mark.parametrize(
        "bad_settings",
        [
            {"study_days": ()},
            {"study_days": (7,)},
            {"reviews_per_session": 0},
            {"minutes_per_review": 0},
            {"memory_gain": 0},
            {"starting_half_life": -1},
            {"new_topic_recall": 1.5},
            {"confidence_bias": 2.0},
            {"confidence_noise": -0.1},
        ],
    )
    def test_bad_settings_are_rejected(self, bad_settings):
        with pytest.raises(ValueError):
            replace(PERSONAS["steady"], **bad_settings)

    def test_a_persona_can_be_built_by_hand(self):
        assert Persona("x", "s", (1,), 5, 1.0, 5.0, 3.0, 0.2, 0.0, 0.0).name == "x"


class TestMemoryModel:
    def test_an_unseen_topic_uses_the_new_topic_chance(self):
        assert recall_probability(None, None, PERSONAS["steady"]) == 0.25

    def test_recall_halves_after_one_half_life(self):
        from datetime import datetime, timedelta

        seen = datetime(2026, 1, 5)
        persona = PERSONAS["steady"]
        assert recall_probability((4.0, seen), seen, persona) == 1.0
        assert recall_probability((4.0, seen), seen + timedelta(days=4), persona) == pytest.approx(
            0.5
        )
        assert recall_probability((4.0, seen), seen + timedelta(days=8), persona) == pytest.approx(
            0.25
        )

    @pytest.mark.parametrize(
        ("success", "probability", "expected"),
        [
            (True, 0.9, 5),
            (True, 0.85, 5),
            (True, 0.7, 4),
            (True, 0.6, 4),
            (True, 0.4, 3),
            (False, 0.6, 2),
            (False, 0.3, 1),
            (False, 0.1, 0),
        ],
    )
    def test_outcomes_map_to_sm2_grades(self, success, probability, expected):
        assert grade_from(success, probability) == expected

    def test_a_fair_learner_feels_what_they_know(self):
        steady = replace(PERSONAS["steady"], confidence_noise=0.0)
        rng = random.Random(0)
        assert confidence_from(steady, 0.9, rng) is Confidence.SURE
        assert confidence_from(steady, 0.5, rng) is Confidence.UNSURE
        assert confidence_from(steady, 0.2, rng) is Confidence.GUESSING

    def test_an_overconfident_learner_feels_surer(self):
        over = replace(PERSONAS["overconfident"], confidence_noise=0.0)
        assert confidence_from(over, 0.45, random.Random(0)) is Confidence.SURE

    def test_an_underconfident_learner_feels_less_sure(self):
        under = replace(PERSONAS["underconfident"], confidence_noise=0.0)
        assert confidence_from(under, 0.9, random.Random(0)) is Confidence.UNSURE


class TestSetup:
    @pytest.mark.parametrize(
        "bad_settings", [{"topics": 0}, {"days": 0}, {"known_from": 0.0}, {"known_from": 1.5}]
    )
    def test_bad_settings_are_rejected(self, bad_settings):
        with pytest.raises(ValueError):
            Setup(**bad_settings)


SHORT = Setup(topics=20, days=30)


class TestSimulate:
    def test_the_same_seed_gives_the_same_run(self):
        first = simulate(PERSONAS["steady"], ADJUSTED, 4, SHORT)
        assert first == simulate(PERSONAS["steady"], ADJUSTED, 4, SHORT)

    def test_different_seeds_give_different_runs(self):
        first = simulate(PERSONAS["overconfident"], ADJUSTED, 1, SHORT)
        second = simulate(PERSONAS["overconfident"], ADJUSTED, 2, SHORT)
        assert first != second

    def test_a_learner_who_never_studies_in_the_window_learns_nothing(self):
        run = simulate(PERSONAS["crammer"], PLAIN, 0, Setup(topics=10, days=3))
        assert run.reviews == 0
        assert run.retention == 0.0
        assert run.known_share == 0.0
        assert run.known_per_100_reviews == 0.0
        assert run.mean_easiness == 2.5
        assert run.weekly_retention == ()

    def test_the_measures_stay_in_range(self):
        run = simulate(PERSONAS["steady"], PLAIN, 0, SHORT)
        assert 0.0 <= run.retention <= 1.0
        assert 0.0 <= run.known_share <= 1.0
        assert run.reviews > 0
        assert 1.3 <= run.mean_easiness

    def test_one_weekly_point_is_kept_for_each_full_week(self):
        assert (
            len(simulate(PERSONAS["steady"], PLAIN, 0, Setup(topics=10, days=21)).weekly_retention)
            == 3
        )

    def test_a_steady_learner_remembers_more_than_a_crammer(self):
        steady = simulate(PERSONAS["steady"], PLAIN, 0, Setup(topics=20, days=60))
        crammer = simulate(PERSONAS["crammer"], PLAIN, 0, Setup(topics=20, days=60))
        assert steady.retention > crammer.retention

    def test_plain_sm2_never_asks_for_recall_checks(self):
        for name in PERSONAS:
            assert simulate(PERSONAS[name], PLAIN, 0, SHORT).recalls == 0

    def test_overconfidence_leads_to_recall_checks_only_when_adjusted(self):
        adjusted = sum(
            simulate(PERSONAS["overconfident"], ADJUSTED, s, SHORT).recalls for s in range(5)
        )
        plain = sum(simulate(PERSONAS["overconfident"], PLAIN, s, SHORT).recalls for s in range(5))
        assert adjusted > 0
        assert plain == 0

    def test_confident_errors_are_counted_in_both_conditions(self):
        plain = simulate(PERSONAS["overconfident"], PLAIN, 0, SHORT)
        adjusted = simulate(PERSONAS["overconfident"], ADJUSTED, 0, SHORT)
        assert plain.confident_errors > 0
        assert adjusted.confident_errors > 0

    def test_the_adjustment_lowers_easiness_for_an_overconfident_learner(self):
        seeds = range(6)
        plain = [simulate(PERSONAS["overconfident"], PLAIN, s, SHORT).mean_easiness for s in seeds]
        adjusted = [
            simulate(PERSONAS["overconfident"], ADJUSTED, s, SHORT).mean_easiness for s in seeds
        ]
        assert sum(adjusted) < sum(plain)

    def test_the_adjustment_barely_touches_a_fair_learner(self):
        seeds = range(6)
        plain = [simulate(PERSONAS["underconfident"], PLAIN, s, SHORT).mean_easiness for s in seeds]
        adjusted = [
            simulate(PERSONAS["underconfident"], ADJUSTED, s, SHORT).mean_easiness for s in seeds
        ]
        assert sum(adjusted) == pytest.approx(sum(plain), abs=0.2)


class TestCompare:
    def test_every_measure_is_compared_with_intervals(self):
        comparison = compare(PERSONAS["overconfident"], range(4), SHORT)
        assert [item.metric for item in comparison.metrics] == list(METRICS)
        for item in comparison.metrics:
            assert item.difference.low <= item.difference.mean <= item.difference.high
        assert comparison.seeds == 4
        assert comparison.persona == "overconfident"

    def test_the_curves_have_one_point_per_week(self):
        comparison = compare(PERSONAS["steady"], range(3), Setup(topics=10, days=28))
        assert len(comparison.plain_curve) == len(comparison.adjusted_curve) == 4

    def test_no_seeds_are_refused(self):
        with pytest.raises(ValueError):
            compare(PERSONAS["steady"], [], SHORT)

    def test_the_difference_is_adjusted_minus_plain(self):
        comparison = compare(PERSONAS["overconfident"], range(4), SHORT)
        item = next(entry for entry in comparison.metrics if entry.metric == "mean_easiness")
        assert item.difference.mean == pytest.approx(item.adjusted.mean - item.plain.mean)
        assert item.difference.mean < 0


class TestRunPersonas:
    def test_chosen_personas_are_run_in_order(self):
        results = run_personas(["steady", "crammer"], 2, SHORT)
        assert [result.persona for result in results] == ["steady", "crammer"]

    def test_an_unknown_persona_lists_the_choices(self):
        with pytest.raises(KeyError, match="crammer"):
            run_personas(["nobody"], 2, SHORT)


class TestSweep:
    def test_one_row_comes_back_for_each_value(self):
        rows = sweep(
            PERSONAS["overconfident"], "confident_error_penalty", [0.0, 0.25, 0.5], range(3), SHORT
        )
        assert [row.value for row in rows] == [0.0, 0.25, 0.5]
        assert all(row.parameter == "confident_error_penalty" for row in rows)
        assert all(0.0 <= row.retention.mean <= 1.0 for row in rows)

    def test_a_bigger_penalty_lowers_the_mean_easiness_curve_effect(self):
        rows = sweep(
            PERSONAS["overconfident"], "confident_error_penalty", [0.0, 0.5], range(4), SHORT
        )
        assert rows[0].difference.mean != rows[1].difference.mean

    def test_the_other_settings_can_be_swept_too(self):
        for parameter, value in (
            ("persistent_overconfidence_extra", 0.2),
            ("overconfidence_threshold", 0.2),
            ("delayed_recall_minutes", 60),
        ):
            assert sweep(PERSONAS["overconfident"], parameter, [value], range(2), SHORT)

    def test_an_unknown_parameter_lists_the_choices(self):
        with pytest.raises(KeyError, match="confident_error_penalty"):
            sweep(PERSONAS["steady"], "luck", [1], range(2), SHORT)

    def test_a_value_the_settings_refuse_is_reported(self):
        with pytest.raises(ValueError):
            sweep(PERSONAS["steady"], "confident_error_penalty", [-0.5], range(2), SHORT)


TEXTS = [
    "Photosynthesis turns light into chemical energy. Chlorophyll absorbs the light.",
    "Mitochondria release energy from glucose. This process is called respiration.",
    "The Roman empire built long roads. Roads helped armies move quickly.",
]


@pytest.fixture
def library():
    database = Database(":memory:")
    shelf = Library(database)
    drafts = [ChunkDraft(i, i + 1, f"Topic {i}", text) for i, text in enumerate(TEXTS)]
    shelf.add_document("notes.txt", "section", drafts, np.zeros((3, 4)))
    yield shelf
    database.close()


@pytest.fixture
def writer():
    return Writer(MockClient(), PromptBook(ROOT / "configs" / "prompts"))


class CountingWriter:
    def __init__(self):
        self.count = 0

    def question(self, chunk, level, support):
        self.count += 1
        return GeneratedQuestion(f"Question number {self.count}", "An answer.")


class TestDataset:
    def test_questions_are_written_from_every_topic(self, library, writer):
        items = build_dataset(library, writer, 9, "mock")
        assert len(items) == 9
        assert {item.chunk_id for item in items} == {chunk.id for chunk in library.chunks()}
        assert [item.id for item in items] == list(range(1, 10))
        assert all(item.document == "notes.txt" and item.written_by == "mock" for item in items)

    def test_repeated_questions_are_dropped(self, library, writer):
        items = build_dataset(library, writer, 30, "mock")
        texts = [item.question.lower() for item in items]
        assert len(texts) == len(set(texts))
        assert len(items) < 30

    def test_the_count_is_met_when_questions_are_different(self, library):
        items = build_dataset(library, CountingWriter(), 10, "test")
        assert len(items) == 10
        per_topic = [sum(1 for item in items if item.chunk_id == c.id) for c in library.chunks()]
        assert max(per_topic) - min(per_topic) <= 1

    def test_the_levels_are_spread_out(self, library):
        items = build_dataset(library, CountingWriter(), 12, "test")
        assert {item.level for item in items} == {2, 3, 4, 5}

    def test_an_empty_library_is_refused(self, writer):
        empty = Database(":memory:")
        with pytest.raises(DatasetError, match="notes"):
            build_dataset(Library(empty), writer, 5, "mock")

    def test_a_count_below_one_is_refused(self, library, writer):
        with pytest.raises(ValueError):
            build_dataset(library, writer, 0, "mock")

    def test_a_dataset_round_trips_through_a_file(self, library, tmp_path):
        items = build_dataset(library, CountingWriter(), 6, "test")
        path = save_dataset(items, tmp_path / "deep" / "qa.jsonl")
        assert load_dataset(path) == items

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path / "none.jsonl")

    def test_a_bad_line_is_reported_with_its_number(self, tmp_path):
        good = json.dumps(
            {
                "id": 1,
                "question": "q",
                "answer": "a",
                "chunk_id": 1,
                "document": "d",
                "page": 1,
                "level": 2,
                "written_by": "m",
            }
        )
        path = tmp_path / "bad.jsonl"
        path.write_text(good + "\n{not json}\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="Line 2"):
            load_dataset(path)

    def test_a_line_with_missing_fields_is_reported(self, tmp_path):
        path = tmp_path / "short.jsonl"
        path.write_text('{"id": 1}\n', encoding="utf-8")
        with pytest.raises(DatasetError, match="Line 1"):
            load_dataset(path)

    def test_an_empty_file_is_reported(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="no questions"):
            load_dataset(path)


NOTES = {
    "bio.txt": b"Photosynthesis turns light into chemical energy. Chlorophyll absorbs the light.",
    "hist.txt": b"The Roman empire built long roads. Roads helped armies and trade move quickly.",
    "chem.txt": b"Atoms bond by sharing electrons. Covalent bonds hold molecules together.",
}


@pytest.fixture
def stocked():
    database = Database(":memory:")
    shelf = Library(database)
    finder = Retriever(shelf, HashingEmbedder(), NumpyIndex())
    chunks = {}
    for name, content in NOTES.items():
        _, made = ingest(shelf, finder, name, content)
        chunks[name] = made[0].id
    yield finder, chunks
    database.close()


def item(number, question, chunk_id, level=2):
    return QAItem(number, question, "answer", chunk_id, "d", 1, level, "test")


class TestRetrievalScores:
    def test_the_right_passage_is_found_first(self, stocked):
        finder, chunks = stocked
        items = [
            item(1, "which pigment absorbs light for photosynthesis", chunks["bio.txt"]),
            item(2, "who built long roads for armies", chunks["hist.txt"]),
            item(3, "how do atoms share electrons in bonds", chunks["chem.txt"]),
        ]
        scores = evaluate_retrieval(finder, items)
        assert scores.questions == 3
        assert scores.hit_at == {1: 1.0, 3: 1.0, 5: 1.0}
        assert scores.mrr == 1.0
        assert scores.misses == 0

    def test_a_wrong_topic_counts_as_a_miss_or_a_low_rank(self, stocked):
        finder, chunks = stocked
        items = [
            item(1, "which pigment absorbs light for photosynthesis", chunks["bio.txt"]),
            item(2, "which pigment absorbs light for photosynthesis", chunks["hist.txt"]),
            item(3, "which pigment absorbs light for photosynthesis", 999),
        ]
        scores = evaluate_retrieval(finder, items, ks=(1, 3))
        assert scores.hit_at[1] == pytest.approx(1 / 3)
        assert scores.hit_at[3] == pytest.approx(2 / 3)
        assert scores.misses == 1
        assert 1 / 3 < scores.mrr < 2 / 3

    def test_the_first_hit_rate_is_reported_for_each_level(self, stocked):
        finder, chunks = stocked
        items = [
            item(1, "which pigment absorbs light for photosynthesis", chunks["bio.txt"], level=2),
            item(2, "which pigment absorbs light for photosynthesis", chunks["hist.txt"], level=3),
        ]
        scores = evaluate_retrieval(finder, items)
        assert scores.hit_at_1_by_level == {2: 1.0, 3: 0.0}

    def test_the_depth_chosen_covers_the_whole_library(self, stocked):
        finder, chunks = stocked
        scores = evaluate_retrieval(finder, [item(1, "roman roads", chunks["bio.txt"])], ks=(1,))
        assert scores.misses == 0
        assert 0 < scores.mrr <= 1

    def test_no_questions_are_refused(self, stocked):
        finder, _ = stocked
        with pytest.raises(ValueError):
            evaluate_retrieval(finder, [])


class TestReports:
    def test_a_markdown_table_has_a_header_and_rows(self):
        text = report.markdown_table([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        assert text.splitlines() == ["| a | b |", "| --- | --- |", "| 1 | x |", "| 2 | y |"]

    def test_an_empty_table_is_empty_text(self):
        assert report.markdown_table([]) == ""

    def test_comparison_rows_cover_every_persona_and_measure(self):
        results = run_personas(["steady", "crammer"], 2, SHORT)
        rows = report.comparison_rows(results)
        assert len(rows) == 2 * len(METRICS)
        assert {row["persona"] for row in rows} == {"steady", "crammer"}
        assert set(rows[0]) == {
            "persona",
            "metric",
            "plain",
            "adjusted",
            "difference",
            "low",
            "high",
        }

    def test_the_curve_rows_number_the_weeks(self):
        results = run_personas(["steady"], 2, Setup(topics=10, days=14))
        rows = report.curve_rows(results)
        assert [row["week"] for row in rows] == [1, 2]

    def test_comparisons_are_written_as_csv_and_markdown(self, tmp_path):
        results = run_personas(["steady"], 2, SHORT)
        written = report.write_comparison(results, tmp_path / "out")
        names = sorted(path.name for path in written)
        assert names == [
            "persona_comparison.csv",
            "persona_comparison.md",
            "retention_curves.csv",
            "retention_curves.md",
        ]
        with (tmp_path / "out" / "persona_comparison.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(METRICS)
        assert rows[0]["persona"] == "steady"

    def test_a_sweep_is_written_with_the_parameter_in_the_name(self, tmp_path):
        rows = sweep(
            PERSONAS["overconfident"], "confident_error_penalty", [0.1, 0.3], range(2), SHORT
        )
        written = report.write_sensitivity(rows, "overconfident", tmp_path)
        assert [path.name for path in written] == [
            "sensitivity_confident_error_penalty.csv",
            "sensitivity_confident_error_penalty.md",
        ]

    def test_retrieval_scores_are_written_as_json_and_markdown(self, stocked, tmp_path):
        finder, chunks = stocked
        scores = evaluate_retrieval(finder, [item(1, "photosynthesis light", chunks["bio.txt"])])
        written = report.write_retrieval(scores, tmp_path)
        loaded = json.loads((tmp_path / "retrieval.json").read_text(encoding="utf-8"))
        assert loaded["questions"] == 1
        assert "hit@1" in (tmp_path / "retrieval.md").read_text(encoding="utf-8")
        assert len(written) == 2


@pytest.fixture
def workspace_env(tmp_path, monkeypatch):
    values = {
        "LLM_PROVIDER": "mock",
        "EMBEDDER": "hashing",
        "INDEX": "numpy",
        "PRIVACY_MODE": "local",
        "DATA_DIR": str(tmp_path / "data"),
        "DATABASE_PATH": str(tmp_path / "data" / "cli.db"),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    opened = Workspace.open(Settings())
    for name, content in NOTES.items():
        opened.add_document(name, content)
    opened.close()
    return tmp_path


class TestCommandLine:
    def test_questions_are_written_to_a_file(self, workspace_env, capsys):
        target = workspace_env / "qa.jsonl"
        assert main(["questions", "--count", "5", "--out", str(target)]) == 0
        assert 1 <= len(load_dataset(target)) <= 5
        shown = capsys.readouterr().out
        assert "Wrote" in shown
        assert "offline model" in shown

    def test_a_shortfall_is_explained(self, workspace_env, capsys):
        target = workspace_env / "qa.jsonl"
        main(["questions", "--count", "50", "--out", str(target)])
        assert "Add more notes" in capsys.readouterr().out

    def test_retrieval_is_scored_from_a_dataset(self, workspace_env, capsys):
        target = workspace_env / "qa.jsonl"
        main(["questions", "--count", "6", "--out", str(target)])
        reports = workspace_env / "reports"
        assert main(["retrieval", "--dataset", str(target), "--out", str(reports)]) == 0
        assert (reports / "retrieval.json").is_file()
        assert "hit@1" in capsys.readouterr().out

    def test_a_missing_dataset_gives_a_clear_error(self, workspace_env, capsys):
        missing = str(workspace_env / "none.jsonl")
        assert main(["retrieval", "--dataset", missing]) == 1
        assert "not found" in capsys.readouterr().err

    def test_the_simulation_writes_its_tables(self, workspace_env, capsys):
        reports = workspace_env / "reports"
        arguments = ["simulate", "--personas", "steady", "crammer", "--seeds", "2"]
        arguments += ["--days", "14", "--topics", "10", "--out", str(reports)]
        assert main(arguments) == 0
        assert (reports / "persona_comparison.csv").is_file()
        shown = capsys.readouterr().out
        assert "steady: retention" in shown
        assert "crammer: retention" in shown

    def test_the_sensitivity_run_writes_its_table(self, workspace_env, capsys):
        reports = workspace_env / "reports"
        arguments = ["sensitivity", "--persona", "overconfident", "--values", "0", "0.2"]
        arguments += ["--seeds", "2", "--days", "14", "--topics", "10", "--out", str(reports)]
        assert main(arguments) == 0
        assert (reports / "sensitivity_confident_error_penalty.csv").is_file()
        assert "confident_error_penalty = 0.2" in capsys.readouterr().out

    def test_a_value_the_settings_refuse_gives_a_clear_error(self, workspace_env, capsys):
        arguments = [
            "sensitivity",
            "--values",
            "-1",
            "--seeds",
            "2",
            "--days",
            "14",
            "--topics",
            "10",
        ]
        arguments += ["--out", str(workspace_env / "reports")]
        assert main(arguments) == 1
        assert "Could not finish" in capsys.readouterr().err

    def test_an_unknown_persona_is_refused_by_the_parser(self):
        with pytest.raises(SystemExit):
            main(["simulate", "--personas", "nobody"])

    def test_a_step_is_required(self):
        with pytest.raises(SystemExit):
            main([])
