import pytest

from calibrayn.evaluation.benchmark import BenchmarkError
from calibrayn.evaluation.groundedness import (
    accuracy_at,
    as_flag,
    auroc,
    evaluate_groundedness,
)
from calibrayn.hub.cli import main

DOCUMENTS = ["Chlorophyll absorbs light for photosynthesis in the chloroplasts of plant cells."]


def row(response, faithful, documents=None):
    return {"response": response, "documents": documents or DOCUMENTS, "adherence": faithful}


FAITHFUL = "Chlorophyll absorbs light in the chloroplasts of plant cells."
INVENTED = "Dragons guard golden treasure beneath volcanic mountains."


class TestAsFlag:
    @pytest.mark.parametrize("value", [True, 1, 1.0, "True", " yes ", "supported", "FAITHFUL", "1"])
    def test_true_values(self, value):
        assert as_flag(value) is True

    @pytest.mark.parametrize(
        "value", [False, 0, 0.0, "False", "no", "unsupported", "hallucinated", "0"]
    )
    def test_false_values(self, value):
        assert as_flag(value) is False

    @pytest.mark.parametrize("value", [2, "maybe", None, 0.5, ""])
    def test_anything_else_is_refused(self, value):
        with pytest.raises(BenchmarkError, match="not true or false"):
            as_flag(value)


class TestAuroc:
    def test_a_perfect_separation_scores_one(self):
        assert auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0

    def test_a_reversed_separation_scores_zero(self):
        assert auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0

    def test_ties_count_as_half(self):
        assert auroc([0.5, 0.5, 0.5, 0.5], [False, False, True, True]) == pytest.approx(0.5)

    def test_a_known_example(self):
        scores = [0.1, 0.4, 0.35, 0.8]
        assert auroc(scores, [False, False, True, True]) == pytest.approx(0.75)

    def test_one_class_alone_is_refused(self):
        with pytest.raises(BenchmarkError, match="Both"):
            auroc([0.1, 0.2], [True, True])


class TestAccuracy:
    def test_it_counts_decisions_at_a_threshold(self):
        assert accuracy_at([0.1, 0.5, 0.9], [False, True, True], 0.4) == 1.0
        assert accuracy_at([0.1, 0.5, 0.9], [False, True, True], 0.6) == pytest.approx(2 / 3)


class TestEvaluate:
    def rows(self):
        return [
            row(FAITHFUL, True),
            row(INVENTED, False),
            row(FAITHFUL, "yes"),
            row(INVENTED, "no"),
        ]

    def test_a_clear_split_is_found(self):
        scores = evaluate_groundedness(self.rows(), "response", "documents", "adherence", 0.35)
        assert scores.count == 4
        assert scores.faithful_share == 0.5
        assert scores.auroc == 1.0
        assert scores.default_accuracy == 1.0
        assert scores.best_accuracy == 1.0

    def test_the_best_threshold_is_reported(self):
        scores = evaluate_groundedness(self.rows(), "response", "documents", "adherence", 1.5)
        assert scores.default_accuracy == 0.5
        assert scores.default_accuracy < scores.best_accuracy
        assert 0.0 < scores.best_threshold < 1.0

    def test_unusable_rows_are_skipped(self):
        rows = self.rows() + [row("", True), row(FAITHFUL, True, documents=[" "])]
        assert evaluate_groundedness(rows, "response", "documents", "adherence", 0.35).count == 4

    def test_missing_columns_are_named(self):
        with pytest.raises(
            BenchmarkError, match="label were not found.*adherence, documents, response"
        ):
            evaluate_groundedness(self.rows(), "response", "documents", "label", 0.35)

    def test_rows_with_nothing_usable_are_refused(self):
        with pytest.raises(BenchmarkError, match="No usable"):
            evaluate_groundedness([row("", True)], "response", "documents", "adherence", 0.35)


class TestCommand:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        for name, value in {
            "LLM_PROVIDER": "mock",
            "EMBEDDER": "hashing",
            "DATA_DIR": str(tmp_path),
        }.items():
            monkeypatch.setenv(name, value)
        return tmp_path

    def arguments(self, out):
        return [
            "groundedness",
            "org/rag",
            "--response-field",
            "response",
            "--documents-field",
            "documents",
            "--label-field",
            "adherence",
            "--out",
            str(out),
        ]

    def test_the_scores_are_printed_and_written(self, env, monkeypatch, capsys):
        rows = [
            row(FAITHFUL, True),
            row(INVENTED, False),
            row(FAITHFUL, True),
            row(INVENTED, False),
        ]
        monkeypatch.setattr("calibrayn.hub.rows.default_loader", lambda *args: rows)
        assert main(self.arguments(env / "reports")) == 0
        shown = capsys.readouterr().out
        assert "4 answers, 50% of them faithful" in shown
        assert "area under the ROC curve: 1.000" in shown
        assert (env / "reports" / "groundedness_org_rag.csv").is_file()

    def test_a_dataset_with_one_kind_of_answer_gives_a_clear_error(self, env, monkeypatch, capsys):
        monkeypatch.setattr(
            "calibrayn.hub.rows.default_loader", lambda *args: [row(FAITHFUL, True)]
        )
        assert main(self.arguments(env / "reports")) == 1
        assert "Both faithful and unfaithful" in capsys.readouterr().err

    def test_a_wrong_label_column_gives_a_clear_error(self, env, monkeypatch, capsys):
        monkeypatch.setattr(
            "calibrayn.hub.rows.default_loader", lambda *args: [row(FAITHFUL, True)]
        )
        arguments = self.arguments(env / "reports")
        arguments[arguments.index("adherence")] = "label"
        assert main(arguments) == 1
        assert "label" in capsys.readouterr().err
