import argparse
import sys

from calibrayn.config import Settings, load_tuning
from calibrayn.evaluation import quality, report
from calibrayn.evaluation.benchmark import (
    BenchmarkError,
    evaluate_benchmark,
    from_library,
    table_rows,
)
from calibrayn.evaluation.dataset import DatasetError, build_dataset, load_dataset, save_dataset
from calibrayn.evaluation.personas import PERSONAS
from calibrayn.evaluation.retrieval import evaluate_retrieval
from calibrayn.evaluation.simulation import PARAMETERS, Setup, run_personas, sweep
from calibrayn.llm.base import LLMError
from calibrayn.services.workspace import Workspace

DATASET_FILE = "evaluation/datasets/qa_100.jsonl"
REPORT_FOLDER = "evaluation/reports"


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m calibrayn.evaluation")
    steps = parser.add_subparsers(dest="step", required=True)

    questions = steps.add_parser("questions", help="write questions from the notes in the library")
    questions.add_argument("--count", type=int, default=100)
    questions.add_argument("--out", default=DATASET_FILE)

    retrieval = steps.add_parser("retrieval", help="measure how often the right passage is found")
    retrieval.add_argument("--dataset", default=DATASET_FILE)
    retrieval.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    retrieval.add_argument("--out", default=REPORT_FOLDER)

    check = steps.add_parser(
        "quality", help="check a question set for easy, repeated or unreviewed questions"
    )
    check.add_argument("--dataset", default=DATASET_FILE)

    export = steps.add_parser(
        "review-export", help="write a sheet for a person to review the questions"
    )
    export.add_argument("--dataset", default=DATASET_FILE)
    export.add_argument("--out", default="evaluation/datasets/review_sheet.csv")

    imported = steps.add_parser("review-import", help="apply a reviewed sheet to the question set")
    imported.add_argument("--dataset", default=DATASET_FILE)
    imported.add_argument("--sheet", required=True)
    imported.add_argument("--reviewer", required=True)
    imported.add_argument("--out", default=DATASET_FILE)

    agree = steps.add_parser("agreement", help="measure how well two reviewers agree")
    agree.add_argument("--a", required=True)
    agree.add_argument("--b", required=True)

    split = steps.add_parser("split", help="divide the questions into validation and test sets")
    split.add_argument("--dataset", default=DATASET_FILE)
    split.add_argument("--out", default=DATASET_FILE)
    split.add_argument("--validation-share", type=float, default=0.3)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--by", choices=["question", "document"], default="question")

    simulate = steps.add_parser(
        "simulate", help="compare plain SM-2 with the confidence-adjusted schedule"
    )
    simulate.add_argument(
        "--personas", nargs="+", choices=sorted(PERSONAS), default=sorted(PERSONAS)
    )
    simulate.add_argument("--seeds", type=int, default=20)
    simulate.add_argument("--days", type=int, default=60)
    simulate.add_argument("--topics", type=int, default=40)
    simulate.add_argument("--out", default=REPORT_FOLDER)

    sensitivity = steps.add_parser("sensitivity", help="vary one setting and watch the result")
    sensitivity.add_argument("--persona", choices=sorted(PERSONAS), default="overconfident")
    sensitivity.add_argument(
        "--parameter", choices=sorted(PARAMETERS), default="confident_error_penalty"
    )
    sensitivity.add_argument("--values", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.4])
    sensitivity.add_argument("--seeds", type=int, default=20)
    sensitivity.add_argument("--days", type=int, default=60)
    sensitivity.add_argument("--topics", type=int, default=40)
    sensitivity.add_argument("--out", default=REPORT_FOLDER)
    return parser


def say(text):
    print(text)


def run_questions(arguments):
    workspace = Workspace.open(Settings())
    try:
        name = workspace.client.name
        items = build_dataset(workspace.library, workspace.writer, arguments.count, name)
    finally:
        workspace.close()
    target = save_dataset(items, arguments.out)
    say(f"Wrote {len(items)} questions to {target} using {name}.")
    if len(items) < arguments.count:
        say(
            f"Only {len(items)} different questions could be made. "
            f"Add more notes to reach {arguments.count}."
        )
    if name == "mock":
        say("The offline model only writes template questions. Use a real model for the report.")
    return 0


def run_retrieval(arguments):
    items = load_dataset(arguments.dataset)
    workspace = Workspace.open(Settings())
    try:
        scores = evaluate_retrieval(workspace.retriever, items, tuple(arguments.ks))
        embedder = workspace.retriever.embedder.name
    finally:
        workspace.close()
    written = report.write_retrieval(scores, arguments.out)
    say(f"Checked {scores.questions} questions with {embedder}.")
    for k, value in scores.hit_at.items():
        say(f"  hit@{k}: {value:.3f}")
    say(f"  mrr: {scores.mrr:.3f}, misses: {scores.misses}")
    written += show_baselines(arguments, items)
    say("Wrote " + ", ".join(str(path) for path in written))
    return 0


def show_baselines(arguments, items):
    workspace = Workspace.open(Settings())
    try:
        benchmark = from_library(workspace.library, items)
        comparison = evaluate_benchmark(benchmark, workspace.retriever.embedder)
    finally:
        workspace.close()
    rows = table_rows(comparison)
    say(f"Compared with a keyword search and chance, over {comparison.corpus_size} topics:")
    for row in rows:
        say(f"  {row['system']}: hit@1 {row['hit@1']}, hit@5 {row['hit@5']}")
    if comparison.corpus_size <= 10:
        say("With so few topics, every system looks good. Use many more topics for the report.")
    return report.write_table(arguments.out, "retrieval_baselines", rows)


def run_quality(arguments):
    items = load_dataset(arguments.dataset)
    summary = quality.review_summary(items)
    say(f"{summary['questions']} questions from {summary['documents']} documents.")
    say(f"  verified by a person: {summary['verified']}")
    say(f"  by level: {summary['by_level']}, by kind: {summary['by_kind']}")
    say(f"  by split: {summary['by_split']}")
    workspace = Workspace.open(Settings())
    try:
        overlap = quality.overlap_report(items, workspace.library)
    finally:
        workspace.close()
    say(f"  question words found in the source: {overlap.mean:.0%} on average")
    say(f"  very easy questions (80 percent or more): {len(overlap.easy_ids)}")
    pairs = quality.near_duplicates(items)
    say(f"  near-duplicate pairs: {len(pairs)}")
    if summary["verified"] < summary["questions"]:
        say("Some questions have not been reviewed by a person. Use review-export.")
    return 0


def run_review_export(arguments):
    target = quality.export_review_sheet(load_dataset(arguments.dataset), arguments.out)
    say(f"Wrote the review sheet to {target}. Fill the verdict column with keep, fix or reject.")
    return 0


def run_review_import(arguments):
    items = load_dataset(arguments.dataset)
    result = quality.import_review_sheet(items, arguments.sheet, arguments.reviewer)
    target = save_dataset(list(result.items), arguments.out)
    say(
        f"Kept {result.kept}, fixed {result.fixed}, rejected {result.rejected}, "
        f"not reviewed {result.pending}. Wrote {target}."
    )
    return 0


def run_agreement(arguments):
    result = quality.review_agreement(arguments.a, arguments.b)
    say(
        f"{result.count} questions marked by both. Agreement {result.observed:.0%}, "
        f"kappa {result.kappa:.2f}."
    )
    return 0


def run_split(arguments):
    items = quality.assign_splits(
        load_dataset(arguments.dataset), arguments.validation_share, arguments.seed, arguments.by
    )
    target = save_dataset(items, arguments.out)
    say(f"Wrote {target}: {quality.review_summary(items)['by_split']}")
    return 0


def run_simulation(arguments):
    tuning = load_tuning(Settings().settings_file)
    setup = Setup(topics=arguments.topics, days=arguments.days)
    comparisons = run_personas(arguments.personas, arguments.seeds, setup, tuning)
    written = report.write_comparison(comparisons, arguments.out)
    for comparison in comparisons:
        retention = next(item for item in comparison.metrics if item.metric == "retention")
        say(
            f"{comparison.persona}: retention {retention.plain.mean:.3f} plain, "
            f"{retention.adjusted.mean:.3f} adjusted, difference {retention.difference.mean:+.3f} "
            f"({retention.difference.low:+.3f} to {retention.difference.high:+.3f})"
        )
    say("Wrote " + ", ".join(str(path) for path in written))
    return 0


def run_sensitivity(arguments):
    tuning = load_tuning(Settings().settings_file)
    setup = Setup(topics=arguments.topics, days=arguments.days)
    rows = sweep(
        PERSONAS[arguments.persona],
        arguments.parameter,
        arguments.values,
        range(arguments.seeds),
        setup,
        tuning,
    )
    written = report.write_sensitivity(rows, arguments.persona, arguments.out)
    for row in rows:
        say(
            f"  {row.parameter} = {row.value}: retention {row.retention.mean:.3f}, "
            f"difference {row.difference.mean:+.3f}"
        )
    say("Wrote " + ", ".join(str(path) for path in written))
    return 0


STEPS = {
    "questions": run_questions,
    "retrieval": run_retrieval,
    "quality": run_quality,
    "review-export": run_review_export,
    "review-import": run_review_import,
    "agreement": run_agreement,
    "split": run_split,
    "simulate": run_simulation,
    "sensitivity": run_sensitivity,
}


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    try:
        return STEPS[arguments.step](arguments)
    except (
        DatasetError,
        BenchmarkError,
        quality.ReviewError,
        LLMError,
        ValueError,
        KeyError,
    ) as error:
        print(f"Could not finish: {error}", file=sys.stderr)
        return 1
