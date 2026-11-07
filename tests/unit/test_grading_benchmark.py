import pytest

from calibrayn.config import ROOT
from calibrayn.evaluation.benchmark import BenchmarkError
from calibrayn.evaluation.grading_benchmark import (
    Example,
    class_f1,
    collect_examples,
    evaluate_grader,
    is_correct_label,
    model_grade_function,
    pearson,
    report_rows,
    score_predictions,
)
from calibrayn.hub.cli import main
from calibrayn.llm.mock import MockClient
from calibrayn.rag.prompts import PromptBook
from calibrayn.services.grading import Grader

FIELDS = ("question", "reference_answer", "student_answer")


def row(answer, label, question="Why does ice float?", reference=None):
    return {
        "question": question,
        "reference_answer": reference or "Ice is less dense than liquid water so it floats.",
        "student_answer": answer,
        "label": label,
    }


ROWS = [
    row("Ice is less dense than liquid water so it floats.", 0),
    row("It floats because it is less dense than liquid water.", 0),
    row("Because it is cold outside.", 1),
    row("I do not know.", 2),
]


class TestLabels:
    @pytest.mark.parametrize(
        ("value", "labels", "expected"),
        [
            (0, ["0"], True),
            ("0", [0], True),
            ("Correct", ["correct"], True),
            (" correct ", ["correct"], True),
            (1, ["0"], False),
            ("incorrect", ["correct"], False),
        ],
    )
    def test_labels_match_by_text(self, value, labels, expected):
        assert is_correct_label(value, labels) is expected


class TestCollect:
    def test_binary_labels_become_correct_or_not(self):
        examples, skipped = collect_examples(ROWS, *FIELDS, "label", ["0"])
        assert [example.correct for example in examples] == [True, True, False, False]
        assert skipped == 0
        assert examples[0].reference.startswith("Ice is less dense")

    def test_unusable_rows_are_counted(self):
        rows = ROWS + [
            row("", 0),
            row("Fine.", 0, question=""),
            {**row("Fine.", 0), "label": 0, "student_answer": 5},
        ]
        examples, skipped = collect_examples(rows, *FIELDS, "label", ["0"])
        assert len(examples) == 4
        assert skipped == 3

    def test_a_numeric_grade_gives_a_score_and_a_label(self):
        rows = [
            {**row("Good answer here.", 0), "grade": 4.5},
            {**row("Poor answer.", 0), "grade": 1.0},
        ]
        examples, _ = collect_examples(rows, *FIELDS, score_field="grade", max_score=5)
        assert [example.correct for example in examples] == [True, False]
        assert [round(example.human_score, 2) for example in examples] == [0.9, 0.2]

    def test_a_bad_numeric_grade_is_skipped(self):
        rows = [{**row("Fine answer.", 0), "grade": "n/a"}, {**row("Good answer.", 0), "grade": 5}]
        examples, skipped = collect_examples(rows, *FIELDS, score_field="grade", max_score=5)
        assert len(examples) == 1
        assert skipped == 1

    def test_missing_columns_are_named_with_the_real_ones(self):
        with pytest.raises(BenchmarkError, match="nope were not found.*label"):
            collect_examples(ROWS, "question", "reference_answer", "nope", "label", ["0"])

    def test_a_label_or_a_score_is_needed(self):
        with pytest.raises(BenchmarkError, match="label column"):
            collect_examples(ROWS, *FIELDS)

    def test_the_correct_labels_must_be_named(self):
        with pytest.raises(BenchmarkError, match="correct-labels"):
            collect_examples(ROWS, *FIELDS, "label", [])

    def test_a_score_column_needs_its_top_score(self):
        with pytest.raises(BenchmarkError, match="highest possible"):
            collect_examples(ROWS, *FIELDS, score_field="label")

    def test_rows_with_nothing_usable_are_refused(self):
        with pytest.raises(BenchmarkError, match="No usable"):
            collect_examples([row("", 0)], *FIELDS, "label", ["0"])


class TestMaths:
    def test_pearson(self):
        assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
        assert pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)

    def test_pearson_needs_spread_and_two_points(self):
        assert pearson([1, 1, 1], [1, 2, 3]) is None
        assert pearson([1], [1]) is None

    def test_class_f1(self):
        precision, recall, f1 = class_f1(
            [True, True, False, False], [True, False, True, False], True
        )
        assert (precision, recall) == (0.5, 0.5)
        assert f1 == 0.5

    def test_class_f1_with_nothing_claimed(self):
        assert class_f1([False, False], [True, False], True) == (0.0, 0.0, 0.0)


def examples(labels):
    return [
        Example(f"Q{n}?", "A reference.", f"Answer {n}.", flag) for n, flag in enumerate(labels)
    ]


class TestScorePredictions:
    def test_perfect_predictions(self):
        data = examples([True, False, True, False])
        scores = score_predictions([True, False, True, False], data, [0.9, 0.1, 0.8, 0.2])
        assert scores.accuracy == 1.0
        assert scores.f1 == 1.0
        assert scores.macro_f1 == 1.0
        assert scores.kappa == 1.0
        assert scores.auroc == 1.0
        assert scores.pearson is None

    def test_completely_wrong_predictions(self):
        data = examples([True, False, True, False])
        scores = score_predictions([False, True, False, True], data, [0.1, 0.9, 0.2, 0.8])
        assert scores.accuracy == 0.0
        assert scores.kappa < 0
        assert scores.auroc == 0.0

    def test_the_auroc_is_left_out_when_only_one_kind_of_answer_exists(self):
        data = examples([True, True])
        assert score_predictions([True, True], data, [0.9, 0.8]).auroc is None

    def test_pearson_is_reported_when_human_scores_exist(self):
        data = [
            Example("q", "r", "a", True, 0.9),
            Example("q", "r", "b", False, 0.2),
            Example("q", "r", "c", True, 0.7),
        ]
        scores = score_predictions([True, False, True], data, [0.8, 0.1, 0.6])
        assert scores.pearson > 0.9


class TestEvaluateGrader:
    def data(self):
        return [
            Example("Why?", "Ice is less dense than water.", "Ice is less dense than water.", True),
            Example("Why?", "Ice is less dense than water.", "It is less dense than water.", True),
            Example("Why?", "Ice is less dense than water.", "Because it is cold.", False),
            Example("Why?", "Ice is less dense than water.", "I do not know.", False),
        ]

    def test_three_systems_are_compared(self):
        truth = {item.answer: item.correct for item in self.data()}
        report = evaluate_grader(self.data(), lambda q, r, a: 1.0 if truth[a] else 0.0, 0.6)
        assert list(report.systems) == [
            "model grader",
            "keyword overlap",
            "always the majority label",
        ]
        assert report.systems["model grader"].accuracy == 1.0
        assert report.count == 4
        assert report.correct_share == 0.5

    def test_the_keyword_baseline_scores_from_word_overlap(self):
        report = evaluate_grader(self.data(), lambda q, r, a: 0.0, 0.6)
        assert report.systems["keyword overlap"].accuracy >= 0.5

    def test_the_majority_label_scores_the_majority_share(self):
        data = self.data()[:3]
        report = evaluate_grader(data, lambda q, r, a: 0.0, 0.6)
        assert report.systems["always the majority label"].accuracy == pytest.approx(2 / 3)

    def test_the_threshold_decides_what_counts_as_correct(self):
        always_half = lambda q, r, a: 0.5  # noqa: E731
        low = evaluate_grader(self.data(), always_half, 0.4).systems["model grader"]
        high = evaluate_grader(self.data(), always_half, 0.6).systems["model grader"]
        assert low.recall == 1.0
        assert high.recall == 0.0

    def test_the_name_and_skipped_count_are_kept(self):
        report = evaluate_grader(self.data(), lambda q, r, a: 1.0, 0.6, skipped=7, name="my model")
        assert "my model" in report.systems
        assert report.skipped == 7


class TestModelGrader:
    def make(self):
        return model_grade_function(Grader(MockClient(), PromptBook(ROOT / "configs" / "prompts")))

    def test_the_reference_is_used_as_the_source(self):
        grade = self.make()
        reference = "Ice is less dense than liquid water so it floats."
        assert grade("Why does ice float?", reference, reference) == 1.0

    def test_an_unrelated_answer_scores_low(self):
        reference = "Ice is less dense than liquid water so it floats."
        assert self.make()("Why does ice float?", reference, "Bananas are yellow.") < 0.3


class TestReportRows:
    def test_numbers_are_rounded_and_missing_values_are_blank(self):
        report = evaluate_grader(
            [
                Example("q", "a reference", "a reference", True),
                Example("q", "a reference", "other", True),
            ],
            lambda q, r, a: 1.0,
            0.6,
        )
        rows = report_rows(report)
        assert rows[0]["system"] == "model grader"
        assert rows[0]["auroc"] == ""
        assert rows[0]["pearson"] == ""
        assert rows[0]["accuracy"] == 1.0


class TestCommand:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        values = {"LLM_PROVIDER": "mock", "EMBEDDER": "hashing", "DATA_DIR": str(tmp_path)}
        for name, value in values.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr("calibrayn.hub.rows.default_loader", lambda *args: ROWS)
        return tmp_path

    def arguments(self, out, *extra):
        base = ["grader", "org/sra", "--question-field", "question"]
        base += ["--reference-field", "reference_answer", "--answer-field", "student_answer"]
        return base + ["--out", str(out), *extra]

    def test_the_grader_is_scored_against_the_baselines(self, env, capsys):
        arguments = self.arguments(
            env / "reports", "--label-field", "label", "--correct-labels", "0"
        )
        assert main(arguments) == 0
        shown = capsys.readouterr().out
        assert "offline grader is being tested" in shown
        assert "4 answers (50% correct" in shown
        assert "keyword overlap" in shown
        assert "always the majority label" in shown
        assert (env / "reports" / "grader_org_sra.csv").is_file()

    def test_the_offline_grader_gets_the_clear_cases_right(self, env, capsys):
        arguments = self.arguments(
            env / "reports", "--label-field", "label", "--correct-labels", "0"
        )
        main(arguments)
        line = next(x for x in capsys.readouterr().out.splitlines() if "grader (mock)" in x)
        assert "accuracy 1.0" in line

    def test_a_threshold_can_be_chosen(self, env, capsys):
        arguments = self.arguments(
            env / "reports",
            "--label-field",
            "label",
            "--correct-labels",
            "0",
            "--threshold",
            "0.99",
        )
        assert main(arguments) == 0
        assert "threshold 0.99" in capsys.readouterr().out

    def test_a_missing_correct_label_gives_a_clear_error(self, env, capsys):
        assert main(self.arguments(env / "reports", "--label-field", "label")) == 1
        assert "correct-labels" in capsys.readouterr().err

    def test_a_wrong_column_gives_a_clear_error(self, env, capsys):
        arguments = self.arguments(
            env / "reports", "--label-field", "grade", "--correct-labels", "0"
        )
        assert main(arguments) == 1
        assert "grade were not found" in capsys.readouterr().err

    def test_a_numeric_score_column_works(self, env, monkeypatch, capsys):
        rows = [{**item, "grade": 5 if item["label"] == 0 else 1} for item in ROWS]
        monkeypatch.setattr("calibrayn.hub.rows.default_loader", lambda *args: rows)
        arguments = self.arguments(env / "reports", "--score-field", "grade", "--max-score", "5")
        assert main(arguments) == 0
        assert "pearson" in (env / "reports" / "grader_org_sra.csv").read_text(encoding="utf-8")
