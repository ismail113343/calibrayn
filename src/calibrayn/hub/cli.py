import argparse
import re
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from calibrayn.config import ROOT, Settings, load_tuning
from calibrayn.evaluation import report
from calibrayn.evaluation.benchmark import (
    BenchmarkError,
    evaluate_benchmark,
    from_rows,
    table_rows,
)
from calibrayn.evaluation.dataset import DatasetError, load_dataset
from calibrayn.evaluation.grading_benchmark import (
    collect_examples,
    evaluate_grader,
    model_grade_function,
    report_rows,
)
from calibrayn.evaluation.groundedness import evaluate_groundedness
from calibrayn.hub import publish, use_token
from calibrayn.hub.models import download_model
from calibrayn.hub.rows import HubError, describe_rows, load_rows
from calibrayn.llm.base import LLMError
from calibrayn.llm.factory import build_client
from calibrayn.rag.embeddings import EmbedderError, build_embedder
from calibrayn.rag.prompts import PromptBook
from calibrayn.services.grading import Grader

BIG_CORPUS = 20000


def make_api(token):
    return HfApi(token=token or None)


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m calibrayn.hub")
    steps = parser.add_subparsers(dest="step", required=True)

    inspect = steps.add_parser("inspect", help="show the columns of a dataset on the Hub")
    inspect.add_argument("repo")
    inspect.add_argument("--config")
    inspect.add_argument("--split", default="test")

    bench = steps.add_parser("benchmark", help="score search on a public dataset")
    bench.add_argument("repo")
    bench.add_argument("--question-field", required=True)
    bench.add_argument("--context-field", required=True)
    bench.add_argument("--config")
    bench.add_argument("--split", default="test")
    bench.add_argument("--limit", type=int, default=500)
    bench.add_argument("--out", default="evaluation/reports")

    ground = steps.add_parser("groundedness", help="test the grounding check on labelled answers")
    ground.add_argument("repo")
    ground.add_argument("--response-field", required=True)
    ground.add_argument("--documents-field", required=True)
    ground.add_argument("--label-field", required=True)
    ground.add_argument("--config")
    ground.add_argument("--split", default="test")
    ground.add_argument("--limit", type=int, default=500)
    ground.add_argument("--out", default="evaluation/reports")

    grade = steps.add_parser("grader", help="test the answer grader on a short-answer benchmark")
    grade.add_argument("repo")
    grade.add_argument("--question-field", required=True)
    grade.add_argument("--reference-field", required=True)
    grade.add_argument("--answer-field", required=True)
    grade.add_argument("--label-field")
    grade.add_argument("--correct-labels", nargs="+", default=[])
    grade.add_argument("--score-field")
    grade.add_argument("--max-score", type=float)
    grade.add_argument("--config")
    grade.add_argument("--split", default="test")
    grade.add_argument("--limit", type=int, default=200)
    grade.add_argument("--threshold", type=float)
    grade.add_argument("--out", default="evaluation/reports")

    share = steps.add_parser("publish", help="publish your question set as a dataset")
    share.add_argument("repo")
    share.add_argument("--dataset", default="evaluation/datasets/qa_100.jsonl")
    share.add_argument("--license", required=True)
    share.add_argument("--author", required=True)
    share.add_argument("--description", default="")
    share.add_argument("--public", action="store_true")
    share.add_argument("--confirm-rights", action="store_true")

    deploy = steps.add_parser("deploy", help="deploy the app as a Space")
    deploy.add_argument("repo")
    deploy.add_argument("--private", action="store_true")

    fetch = steps.add_parser("download", help="download the models chosen in the settings")
    fetch.add_argument("--llm", action="store_true")
    fetch.add_argument("--embedding", action="store_true")
    return parser


def say(text):
    print(text)


def run_inspect(arguments):
    rows = load_rows(arguments.repo, arguments.config, arguments.split, 3)
    say(f"{arguments.repo} ({arguments.split}) has these columns:")
    for entry in describe_rows(rows):
        say(f"  {entry['column']} ({entry['type']}): {entry['example']}")
    return 0


def run_benchmark(arguments):
    rows = load_rows(arguments.repo, arguments.config, arguments.split, arguments.limit)
    benchmark = from_rows(rows, arguments.question_field, arguments.context_field)
    embedder = build_embedder(Settings())
    if embedder.name.startswith("hashing"):
        say("Using the offline hashing embedder. Set EMBEDDER=huggingface for real results.")
    if len(benchmark.corpus) > BIG_CORPUS:
        say(f"The corpus has {len(benchmark.corpus)} passages. This may take a while.")
    comparison = evaluate_benchmark(benchmark, embedder)
    rows_out = table_rows(comparison)
    say(
        f"{comparison.queries} questions over {comparison.corpus_size} passages "
        f"({comparison.skipped} rows skipped)."
    )
    for row in rows_out:
        say(
            f"  {row['system']}: hit@1 {row['hit@1']}, hit@5 {row['hit@5']}, hit@10 {row['hit@10']}"
        )
    name = "benchmark_" + re.sub(r"[^A-Za-z0-9]+", "_", arguments.repo).strip("_")
    written = report.write_table(arguments.out, name, rows_out)
    say("Wrote " + ", ".join(str(path) for path in written))
    return 0


def run_groundedness(arguments):
    rows = load_rows(arguments.repo, arguments.config, arguments.split, arguments.limit)
    tuning = load_tuning(Settings().settings_file)
    scores = evaluate_groundedness(
        rows,
        arguments.response_field,
        arguments.documents_field,
        arguments.label_field,
        tuning.grounding.min_support,
    )
    say(f"{scores.count} answers, {scores.faithful_share:.0%} of them faithful to their documents.")
    say(f"  area under the ROC curve: {scores.auroc:.3f} (0.5 is no better than chance)")
    say(f"  accuracy at the app's threshold {scores.default_threshold}: ")
    say(f"    {scores.default_accuracy:.3f}")
    say(f"  best accuracy {scores.best_accuracy:.3f} at threshold {scores.best_threshold}")
    table = [
        {"measure": "answers", "value": scores.count},
        {"measure": "faithful share", "value": round(scores.faithful_share, 4)},
        {"measure": "auroc", "value": round(scores.auroc, 4)},
        {"measure": "accuracy at app threshold", "value": round(scores.default_accuracy, 4)},
        {"measure": "best accuracy", "value": round(scores.best_accuracy, 4)},
        {"measure": "best threshold", "value": scores.best_threshold},
    ]
    name = "groundedness_" + re.sub(r"[^A-Za-z0-9]+", "_", arguments.repo).strip("_")
    written = report.write_table(arguments.out, name, table)
    say("Wrote " + ", ".join(str(path) for path in written))
    return 0


def run_grader(arguments):
    settings = Settings()
    rows = load_rows(arguments.repo, arguments.config, arguments.split, arguments.limit)
    examples, skipped = collect_examples(
        rows,
        arguments.question_field,
        arguments.reference_field,
        arguments.answer_field,
        arguments.label_field,
        arguments.correct_labels,
        arguments.score_field,
        arguments.max_score,
    )
    tuning = load_tuning(settings.settings_file)
    threshold = (
        arguments.threshold if arguments.threshold is not None else tuning.agent.correct_from
    )
    client = build_client(settings)
    if client.name == "mock":
        say("No language model is connected, so the offline grader is being tested.")
    grader = Grader(client, PromptBook(settings.prompts_dir))
    say(f"Grading {len(examples)} answers with {client.name}. This can take a while.")
    comparison = evaluate_grader(
        examples, model_grade_function(grader), threshold, skipped, f"grader ({client.name})"
    )
    say(
        f"{comparison.count} answers ({comparison.correct_share:.0%} correct, "
        f"{comparison.skipped} rows skipped), threshold {threshold}."
    )
    table = report_rows(comparison)
    for row in table:
        say(
            f"  {row['system']}: accuracy {row['accuracy']}, macro F1 {row['macro_f1']}, "
            f"kappa {row['kappa']}"
        )
    if comparison.model_share is not None:
        say(f"The model itself marked {comparison.model_share:.0%} of the answers.")
        if comparison.model_share < 1.0:
            say("The rest fell back to keyword overlap, so the grader row mixes both.")
    name = "grader_" + re.sub(r"[^A-Za-z0-9]+", "_", arguments.repo).strip("_")
    written = report.write_table(arguments.out, name, table)
    say("Wrote " + ", ".join(str(path) for path in written))
    return 0


def run_publish(arguments):
    items = load_dataset(arguments.dataset)
    api = make_api(Settings().hf_token)
    url = publish.publish_dataset(
        api,
        arguments.repo,
        items,
        arguments.license,
        arguments.author,
        private=not arguments.public,
        rights_confirmed=arguments.confirm_rights,
        description=arguments.description,
    )
    visibility = "public" if arguments.public else "private"
    say(f"Published {len(items)} questions as a {visibility} dataset: {url}")
    return 0


def run_deploy(arguments):
    api = make_api(Settings().hf_token)
    with tempfile.TemporaryDirectory() as folder:
        staged = publish.stage_space(ROOT, Path(folder))
        url = publish.deploy_space(api, arguments.repo, folder, private=arguments.private)
    say(f"Uploaded {len(staged)} files to {url}")
    say("Add HF_TOKEN as a secret in the Space settings if the app should use hosted inference.")
    say("The first build takes a few minutes.")
    return 0


def run_download(arguments):
    settings = Settings()
    wanted = []
    if arguments.embedding or not arguments.llm:
        wanted.append(settings.hf_embedding_model)
    if arguments.llm or not arguments.embedding:
        wanted.append(settings.hf_llm_model)
    for repo in wanted:
        say(f"Downloading {repo} ...")
        say(f"  saved in {download_model(repo, settings.hf_token)}")
    return 0


STEPS = {
    "inspect": run_inspect,
    "benchmark": run_benchmark,
    "groundedness": run_groundedness,
    "grader": run_grader,
    "publish": run_publish,
    "deploy": run_deploy,
    "download": run_download,
}


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    try:
        use_token(Settings().hf_token)
        return STEPS[arguments.step](arguments)
    except (
        HubError,
        BenchmarkError,
        DatasetError,
        EmbedderError,
        LLMError,
        ValueError,
        KeyError,
    ) as error:
        print(f"Could not finish: {error}", file=sys.stderr)
        return 1
