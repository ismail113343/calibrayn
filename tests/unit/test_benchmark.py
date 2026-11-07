import csv
import math

import numpy as np
import pytest

from calibrayn.evaluation import quality
from calibrayn.evaluation.baselines import Bm25
from calibrayn.evaluation.benchmark import (
    BenchmarkError,
    Query,
    RetrievalBenchmark,
    embedder_ranks,
    evaluate_benchmark,
    flatten_passages,
    from_library,
    from_rows,
    table_rows,
)
from calibrayn.evaluation.cli import main
from calibrayn.evaluation.dataset import QAItem, load_dataset, save_dataset
from calibrayn.evaluation.metrics import chance_hit_at, chance_scores, ndcg, score_rankings
from calibrayn.models.documents import ChunkDraft
from calibrayn.rag.embeddings import HashingEmbedder
from calibrayn.storage.database import Database
from calibrayn.storage.library import Library


class TestChance:
    def test_one_pick_among_ten_with_one_right_answer(self):
        assert chance_hit_at(10, 1, 1) == pytest.approx(0.1)

    def test_three_picks_among_ten(self):
        assert chance_hit_at(10, 1, 3) == pytest.approx(0.3)

    def test_two_right_answers_double_the_chance_for_one_pick(self):
        assert chance_hit_at(10, 2, 1) == pytest.approx(0.2)

    def test_picking_everything_always_hits(self):
        assert chance_hit_at(5, 1, 5) == 1.0
        assert chance_hit_at(5, 1, 50) == 1.0

    def test_everything_being_right_always_hits(self):
        assert chance_hit_at(3, 3, 1) == 1.0

    @pytest.mark.parametrize(("size", "gold"), [(0, 1), (5, 0)])
    def test_empty_inputs_are_refused(self, size, gold):
        with pytest.raises(ValueError):
            chance_hit_at(size, gold, 1)

    def test_scores_are_averaged_over_queries(self):
        scores = chance_scores(10, [1, 2], (1, 5))
        assert scores.hit_at[1] == pytest.approx(0.15)
        assert scores.recall_at == {1: 0.1, 5: 0.5}


class TestNdcg:
    def test_a_first_place_answer_is_perfect(self):
        assert ndcg((1,), 1, 10) == 1.0

    def test_a_second_place_answer_is_discounted(self):
        assert ndcg((2,), 1, 10) == pytest.approx(1 / math.log2(3))

    def test_an_answer_beyond_the_cut_off_scores_nothing(self):
        assert ndcg((11,), 1, 10) == 0.0

    def test_finding_every_answer_first_is_perfect(self):
        assert ndcg((1, 2), 2, 10) == pytest.approx(1.0)

    def test_finding_only_one_of_two_scores_less(self):
        expected = 1.0 / (1.0 + 1.0 / math.log2(3))
        assert ndcg((1, None), 2, 10) == pytest.approx(expected)


class TestScoreRankings:
    def test_a_mixed_set_of_queries(self):
        scores = score_rankings([(1,), (3,), (None,), (2, 5)], (1, 3))
        assert scores.queries == 4
        assert scores.hit_at == {1: 0.25, 3: 0.75}
        assert scores.recall_at[3] == pytest.approx(0.625)
        assert scores.mrr == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)
        assert scores.misses == 1

    def test_perfect_rankings_score_one_everywhere(self):
        scores = score_rankings([(1,), (1,)], (1, 5, 10))
        assert all(value == 1.0 for value in scores.hit_at.values())
        assert all(value == pytest.approx(1.0) for value in scores.ndcg_at.values())
        assert scores.mrr == 1.0
        assert scores.misses == 0

    def test_no_queries_are_refused(self):
        with pytest.raises(ValueError):
            score_rankings([], (1,))


CORPUS = [
    "Photosynthesis turns light into chemical energy using chlorophyll.",
    "The Roman empire built long roads across Europe for armies.",
    "Atoms bond by sharing electrons in covalent bonds.",
    "Photosynthesis photosynthesis photosynthesis " + "filler " * 40,
]


class TestBm25:
    def test_the_matching_document_comes_first(self):
        ranks = Bm25(CORPUS).ranks_of("who built roads for armies", (1,))
        assert ranks == (1,)

    def test_a_document_without_any_query_word_is_not_ranked(self):
        assert Bm25(CORPUS).ranks_of("who built roads for armies", (2,)) == (None,)

    def test_a_query_of_stop_words_ranks_nothing(self):
        assert Bm25(CORPUS).ranks_of("the and of", (0,)) == (None,)

    def test_a_short_document_beats_a_long_one_for_the_same_word(self):
        ranks = Bm25(CORPUS).ranks_of("photosynthesis", (0, 3))
        assert ranks[0] < ranks[1]

    def test_several_answers_are_ranked_separately(self):
        assert Bm25(CORPUS).ranks_of("roads armies electrons", (1, 2)) == (1, 2)

    def test_an_empty_corpus_gives_no_scores(self):
        assert Bm25([]).scores("anything") == {}


class TestFlatten:
    def test_text_becomes_one_passage(self):
        assert flatten_passages("  hello  ") == ["hello"]

    def test_blank_text_is_dropped(self):
        assert flatten_passages("   ") == []

    def test_lists_are_flattened_to_any_depth(self):
        assert flatten_passages(["a", ["b", ["c"]], " "]) == ["a", "b", "c"]

    def test_other_types_are_refused(self):
        with pytest.raises(BenchmarkError, match="dict"):
            flatten_passages({"a": 1})


class TestFromRows:
    def test_each_row_becomes_a_query_with_its_passage_as_the_answer(self):
        rows = [
            {"q": "Who built roads?", "c": "The Romans built roads."},
            {"q": "What is a bond?", "c": "Atoms share electrons."},
        ]
        benchmark = from_rows(rows, "q", "c")
        assert benchmark.corpus == ("The Romans built roads.", "Atoms share electrons.")
        assert [query.gold for query in benchmark.queries] == [(0,), (1,)]

    def test_a_repeated_passage_is_stored_once(self):
        rows = [{"q": "one?", "c": "Same text."}, {"q": "two?", "c": "Same text."}]
        benchmark = from_rows(rows, "q", "c")
        assert len(benchmark.corpus) == 1
        assert [query.gold for query in benchmark.queries] == [(0,), (0,)]

    def test_a_row_with_several_passages_has_several_answers(self):
        benchmark = from_rows([{"q": "What?", "c": ["First.", "Second.", "First."]}], "q", "c")
        assert benchmark.queries[0].gold == (0, 1)

    def test_unusable_rows_are_counted_and_skipped(self):
        rows = [
            {"q": "Real question?", "c": "Real passage."},
            {"q": "", "c": "No question."},
            {"q": "No passage?", "c": ["  "]},
            {"q": 5, "c": "Not text."},
        ]
        benchmark = from_rows(rows, "q", "c")
        assert len(benchmark.queries) == 1
        assert benchmark.skipped == 3

    def test_a_wrong_column_name_lists_the_real_ones(self):
        with pytest.raises(BenchmarkError, match="question, response"):
            from_rows([{"question": "x", "response": "y"}], "q", "c")

    def test_rows_with_nothing_usable_are_refused(self):
        with pytest.raises(BenchmarkError, match="No usable"):
            from_rows([{"q": "", "c": ""}], "q", "c")


def item(number, question, chunk_id, level=2):
    return QAItem(number, question, "answer", chunk_id, "notes.txt", 1, level, "test")


@pytest.fixture
def library():
    database = Database(":memory:")
    shelf = Library(database)
    drafts = [ChunkDraft(i, 1, f"Topic {i}", text) for i, text in enumerate(CORPUS[:3])]
    _, chunks = shelf.add_document("notes.txt", "section", drafts, np.zeros((3, 4)))
    shelf.test_chunks = chunks
    yield shelf
    database.close()


class TestFromLibrary:
    def test_questions_point_at_their_topic_by_position(self, library):
        chunks = library.test_chunks
        benchmark = from_library(library, [item(1, "roads armies", chunks[1].id, level=3)])
        assert benchmark.corpus[1] == CORPUS[1]
        assert benchmark.queries == (Query("roads armies", (1,), 3),)

    def test_questions_about_missing_topics_are_skipped(self, library):
        chunks = library.test_chunks
        benchmark = from_library(
            library, [item(1, "roads armies", chunks[1].id), item(2, "lost", 999)]
        )
        assert len(benchmark.queries) == 1
        assert benchmark.skipped == 1

    def test_no_matching_topics_is_an_error(self, library):
        with pytest.raises(BenchmarkError):
            from_library(library, [item(1, "lost", 999)])


class TestEvaluateBenchmark:
    def make(self):
        queries = (
            Query("which pigment absorbs light in photosynthesis", (0,)),
            Query("who built roads for armies", (1,)),
            Query("how do atoms share electrons", (2,)),
        )
        return RetrievalBenchmark(tuple(CORPUS[:3]), queries)

    def test_embedding_ranks_put_the_right_passage_first(self):
        assert embedder_ranks(self.make(), HashingEmbedder()) == [(1,), (1,), (1,)]

    def test_both_systems_and_chance_are_reported(self):
        report = evaluate_benchmark(self.make(), HashingEmbedder())
        assert set(report.systems) == {"embeddings (hashing-384)", "keyword search (BM25)"}
        assert report.corpus_size == 3
        assert report.queries == 3
        assert report.systems["embeddings (hashing-384)"].hit_at[1] == 1.0
        assert report.chance.hit_at[1] == pytest.approx(1 / 3)

    def test_the_table_has_one_row_per_system_and_a_chance_row(self):
        rows = table_rows(evaluate_benchmark(self.make(), HashingEmbedder()))
        assert [row["system"] for row in rows][-1] == "random guessing"
        assert len(rows) == 3
        assert rows[0]["hit@1"] == 1.0
        assert rows[-1]["mrr"] == ""

    def test_the_table_works_with_other_cut_offs(self):
        report = evaluate_benchmark(self.make(), HashingEmbedder(), ks=(1, 3))
        rows = table_rows(report, ks=(1, 3))
        assert "hit@3" in rows[0]
        assert "hit@10" not in rows[0]


def labelled(number, question="What is mitosis in cells?", document="a.txt", chunk_id=1):
    return QAItem(number, question, "answer", chunk_id, document, 1, 2, "test")


class TestOverlap:
    def test_the_share_of_question_words_found_in_the_source(self):
        assert quality.question_overlap("roads armies bananas", CORPUS[1]) == pytest.approx(2 / 3)

    def test_a_question_with_no_content_words_scores_zero(self):
        assert quality.question_overlap("what is the", CORPUS[1]) == 0.0

    def test_easy_questions_are_flagged(self, library):
        chunks = library.test_chunks
        items = [
            item(1, "photosynthesis chlorophyll light energy", chunks[0].id),
            item(2, "quantum bananas elephants", chunks[0].id),
        ]
        report = quality.overlap_report(items, library)
        assert report.easy_ids == (1,)
        assert report.easy_share == 0.5
        assert 0.0 < report.mean < 1.0

    def test_questions_about_missing_topics_are_ignored_and_none_left_is_an_error(self, library):
        with pytest.raises(quality.ReviewError):
            quality.overlap_report([item(1, "lost", 999)], library)


class TestDuplicates:
    def test_the_same_question_twice_is_a_pair(self):
        items = [labelled(1), labelled(2), labelled(3, "Why do volcanoes erupt violently?")]
        assert quality.near_duplicates(items) == [(1, 2)]

    def test_different_questions_are_not_pairs(self):
        assert quality.near_duplicates([labelled(1), labelled(2, "Why do volcanoes erupt?")]) == []

    def test_a_lower_threshold_catches_looser_matches(self):
        items = [
            labelled(1, "What is mitosis in cells?"),
            labelled(2, "What is mitosis in plants?"),
        ]
        assert quality.near_duplicates(items, threshold=0.9) == []
        assert quality.near_duplicates(items, threshold=0.3) == [(1, 2)]


class TestSplits:
    def test_the_same_seed_gives_the_same_split(self):
        items = [labelled(number) for number in range(1, 51)]
        assert quality.assign_splits(items, seed=1) == quality.assign_splits(items, seed=1)

    def test_a_different_seed_gives_a_different_split(self):
        items = [labelled(number) for number in range(1, 51)]
        assert quality.assign_splits(items, seed=1) != quality.assign_splits(items, seed=2)

    def test_the_validation_share_is_about_right(self):
        items = [labelled(number) for number in range(1, 201)]
        split = quality.assign_splits(items, validation_share=0.3)
        share = sum(1 for entry in split if entry.split == "validation") / len(split)
        assert 0.2 < share < 0.4
        assert {entry.split for entry in split} == {"validation", "test"}

    def test_splitting_by_document_keeps_a_document_together(self):
        items = [labelled(n, document=f"doc{n % 6}.txt") for n in range(1, 61)]
        split = quality.assign_splits(items, by="document")
        for name in {entry.document for entry in split}:
            assert len({entry.split for entry in split if entry.document == name}) == 1

    @pytest.mark.parametrize(
        "kwargs", [{"validation_share": 0.0}, {"validation_share": 1.0}, {"by": "page"}]
    )
    def test_bad_settings_are_refused(self, kwargs):
        with pytest.raises(ValueError):
            quality.assign_splits([labelled(1)], **kwargs)


class TestKappa:
    def test_perfect_agreement(self):
        assert quality.cohen_kappa(["keep", "reject"], ["keep", "reject"]) == 1.0

    def test_everyone_saying_the_same_thing_counts_as_agreement(self):
        assert quality.cohen_kappa(["keep", "keep"], ["keep", "keep"]) == 1.0

    def test_a_known_example(self):
        assert quality.cohen_kappa(["k", "k", "r", "r"], ["k", "r", "r", "r"]) == pytest.approx(0.5)

    def test_complete_disagreement_is_negative(self):
        assert quality.cohen_kappa(["k", "r"], ["r", "k"]) < 0

    def test_different_lengths_or_nothing_are_refused(self):
        with pytest.raises(ValueError):
            quality.cohen_kappa(["a"], ["a", "b"])
        with pytest.raises(ValueError):
            quality.cohen_kappa([], [])


class TestSummary:
    def test_counts_are_grouped(self):
        items = [
            labelled(1),
            QAItem(2, "q", "a", 1, "b.txt", 1, 3, "t", verified=True, kind="explain", split="test"),
        ]
        summary = quality.review_summary(items)
        assert summary["questions"] == 2
        assert summary["verified"] == 1
        assert summary["documents"] == 2
        assert summary["by_level"] == {2: 1, 3: 1}
        assert summary["by_split"] == {"none": 1, "test": 1}


def fill(path, verdicts, extra=None):
    extra = extra or {}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["verdict"] = verdicts.get(int(row["id"]), "")
        row.update(extra.get(int(row["id"]), {}))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=quality.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class TestReviewSheet:
    def items(self):
        return [labelled(number, f"Question number {number}?") for number in range(1, 5)]

    def test_the_sheet_lists_every_question_with_blank_verdicts(self, tmp_path):
        path = quality.export_review_sheet(self.items(), tmp_path / "sheet.csv")
        rows = quality.read_sheet(path)
        assert [row["question"] for row in rows][0] == "Question number 1?"
        assert all(row["verdict"] == "" for row in rows)

    def test_verdicts_are_applied(self, tmp_path):
        path = quality.export_review_sheet(self.items(), tmp_path / "sheet.csv")
        fill(
            path,
            {1: "keep", 2: "fix", 3: "reject"},
            extra={2: {"question_fixed": "A better question?", "answer_fixed": "A better answer."}},
        )
        result = quality.import_review_sheet(self.items(), path, "Sam")
        assert (result.kept, result.fixed, result.rejected, result.pending) == (1, 1, 1, 1)
        by_id = {entry.id: entry for entry in result.items}
        assert set(by_id) == {1, 2, 4}
        assert by_id[1].verified and by_id[1].reviewer == "Sam"
        assert by_id[2].question == "A better question?"
        assert by_id[2].answer == "A better answer."
        assert by_id[2].verified
        assert not by_id[4].verified

    def test_a_fix_without_new_text_keeps_the_old_text(self, tmp_path):
        path = quality.export_review_sheet(self.items(), tmp_path / "sheet.csv")
        fill(path, {1: "fix"})
        result = quality.import_review_sheet(self.items(), path, "Sam")
        assert result.items[0].question == "Question number 1?"
        assert result.items[0].verified

    def test_verdicts_are_not_case_sensitive(self, tmp_path):
        path = quality.export_review_sheet(self.items(), tmp_path / "sheet.csv")
        fill(path, {1: " KEEP "})
        assert quality.import_review_sheet(self.items(), path, "Sam").kept == 1

    def test_an_unknown_verdict_names_the_question(self, tmp_path):
        path = quality.export_review_sheet(self.items(), tmp_path / "sheet.csv")
        fill(path, {2: "maybe"})
        with pytest.raises(quality.ReviewError, match="Question 2"):
            quality.import_review_sheet(self.items(), path, "Sam")

    def test_a_reviewer_name_is_required(self, tmp_path):
        path = quality.export_review_sheet(self.items(), tmp_path / "sheet.csv")
        with pytest.raises(quality.ReviewError, match="reviewer"):
            quality.import_review_sheet(self.items(), path, "  ")

    def test_a_missing_sheet_is_reported(self, tmp_path):
        with pytest.raises(quality.ReviewError, match="not found"):
            quality.read_sheet(tmp_path / "none.csv")

    def test_a_sheet_with_missing_columns_is_reported(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("id,question\n1,Why?\n", encoding="utf-8")
        with pytest.raises(quality.ReviewError, match="missing columns"):
            quality.read_sheet(path)


class TestAgreement:
    def test_two_reviewers_are_compared_on_what_both_marked(self, tmp_path):
        items = [labelled(number, f"Question number {number}?") for number in range(1, 7)]
        first = quality.export_review_sheet(items, tmp_path / "a.csv")
        second = quality.export_review_sheet(items, tmp_path / "b.csv")
        fill(first, {1: "keep", 2: "keep", 3: "reject", 4: "reject", 5: "keep"})
        fill(second, {1: "keep", 2: "reject", 3: "reject", 4: "reject", 6: "keep"})
        result = quality.review_agreement(first, second)
        assert result.count == 4
        assert result.observed == 0.75
        assert result.kappa == pytest.approx(0.5)

    def test_sheets_with_nothing_in_common_are_reported(self, tmp_path):
        items = [labelled(1), labelled(2, "Another question here?")]
        first = quality.export_review_sheet(items, tmp_path / "a.csv")
        second = quality.export_review_sheet(items, tmp_path / "b.csv")
        fill(first, {1: "keep"})
        fill(second, {2: "keep"})
        with pytest.raises(quality.ReviewError, match="no questions"):
            quality.review_agreement(first, second)


class TestBenchmarkCommands:
    def make_dataset(self, tmp_path, capsys):
        target = tmp_path / "qa.jsonl"
        assert main(["questions", "--count", "9", "--out", str(target)]) == 0
        capsys.readouterr()
        return target

    def test_retrieval_also_compares_with_keyword_search_and_chance(self, sample_env, capsys):
        target = self.make_dataset(sample_env, capsys)
        reports = sample_env / "reports"
        assert main(["retrieval", "--dataset", str(target), "--out", str(reports)]) == 0
        shown = capsys.readouterr().out
        assert "keyword search (BM25)" in shown
        assert "random guessing" in shown
        assert "With so few topics" in shown
        assert (reports / "retrieval_baselines.csv").is_file()

    def test_the_quality_check_summarises_the_questions(self, sample_env, capsys):
        target = self.make_dataset(sample_env, capsys)
        assert main(["quality", "--dataset", str(target)]) == 0
        shown = capsys.readouterr().out
        assert "verified by a person: 0" in shown
        assert "near-duplicate pairs" in shown
        assert "not been reviewed" in shown

    def test_a_review_round_trip_marks_questions_as_verified(self, sample_env, capsys):
        target = self.make_dataset(sample_env, capsys)
        sheet = sample_env / "sheet.csv"
        assert main(["review-export", "--dataset", str(target), "--out", str(sheet)]) == 0
        rows = quality.read_sheet(sheet)
        fill(sheet, {int(row["id"]): "keep" for row in rows})
        reviewed = sample_env / "reviewed.jsonl"
        arguments = ["review-import", "--dataset", str(target), "--sheet", str(sheet)]
        arguments += ["--reviewer", "Sam", "--out", str(reviewed)]
        assert main(arguments) == 0
        assert all(entry.verified and entry.reviewer == "Sam" for entry in load_dataset(reviewed))
        capsys.readouterr()
        main(["quality", "--dataset", str(reviewed)])
        assert "not been reviewed" not in capsys.readouterr().out

    def test_agreement_is_printed(self, sample_env, capsys):
        target = self.make_dataset(sample_env, capsys)
        first = sample_env / "a.csv"
        second = sample_env / "b.csv"
        main(["review-export", "--dataset", str(target), "--out", str(first)])
        main(["review-export", "--dataset", str(target), "--out", str(second)])
        rows = quality.read_sheet(first)
        fill(first, {int(row["id"]): "keep" for row in rows})
        fill(second, {int(row["id"]): "keep" for row in rows})
        capsys.readouterr()
        assert main(["agreement", "--a", str(first), "--b", str(second)]) == 0
        assert "kappa 1.00" in capsys.readouterr().out

    def test_agreement_without_shared_marks_gives_a_clear_error(self, sample_env, capsys):
        target = self.make_dataset(sample_env, capsys)
        sheet = sample_env / "a.csv"
        main(["review-export", "--dataset", str(target), "--out", str(sheet)])
        capsys.readouterr()
        assert main(["agreement", "--a", str(sheet), "--b", str(sheet)]) == 1
        assert "no questions" in capsys.readouterr().err

    def test_splits_are_written_back_to_the_file(self, sample_env, capsys):
        target = self.make_dataset(sample_env, capsys)
        output = sample_env / "split.jsonl"
        assert main(["split", "--dataset", str(target), "--out", str(output)]) == 0
        splits = {entry.split for entry in load_dataset(output)}
        assert splits and splits <= {"validation", "test"}
        assert "Wrote" in capsys.readouterr().out

    def test_a_dataset_can_be_saved_and_reloaded_with_review_fields(self, sample_env, tmp_path):
        items = [
            QAItem(
                1,
                "q?",
                "a",
                1,
                "d",
                1,
                2,
                "m",
                answerable=False,
                kind="apply",
                evidence="e",
                verified=True,
                reviewer="Sam",
                split="test",
            )
        ]
        assert load_dataset(save_dataset(items, tmp_path / "x.jsonl")) == items
