from dataclasses import dataclass
from math import sqrt
from types import SimpleNamespace

from calibrayn.evaluation.benchmark import BenchmarkError
from calibrayn.evaluation.groundedness import auroc
from calibrayn.evaluation.quality import cohen_kappa
from calibrayn.services.grading import keyword_verdict


@dataclass(frozen=True)
class Example:
    question: str
    reference: str
    answer: str
    correct: bool
    human_score: float | None = None


@dataclass(frozen=True)
class GradingScores:
    count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    kappa: float
    auroc: float | None
    pearson: float | None


@dataclass(frozen=True)
class GradingReport:
    count: int
    correct_share: float
    skipped: int
    systems: dict


def is_correct_label(value, correct_labels):
    wanted = {str(label).strip().lower() for label in correct_labels}
    return str(value).strip().lower() in wanted


def collect_examples(
    rows,
    question_field,
    reference_field,
    answer_field,
    label_field=None,
    correct_labels=(),
    score_field=None,
    max_score=None,
):
    if not label_field and not score_field:
        raise BenchmarkError("Give a label column with its correct labels, or a score column.")
    if label_field and not score_field and not correct_labels:
        raise BenchmarkError("Say which label values mean correct, for example --correct-labels 0.")
    if score_field and not max_score:
        raise BenchmarkError("A score column needs the highest possible score.")
    examples = []
    skipped = 0
    needed = [question_field, reference_field, answer_field] + [
        name for name in (label_field, score_field) if name
    ]
    for row in rows:
        missing = [name for name in needed if name not in row]
        if missing:
            raise BenchmarkError(
                f"The columns {', '.join(missing)} were not found. "
                f"This dataset has: {', '.join(sorted(row))}."
            )
        texts = [row[question_field], row[reference_field], row[answer_field]]
        if not all(isinstance(text, str) and text.strip() for text in texts):
            skipped += 1
            continue
        human = None
        if score_field:
            try:
                human = float(row[score_field]) / max_score
            except (TypeError, ValueError):
                skipped += 1
                continue
            correct = human >= 0.5
        else:
            correct = is_correct_label(row[label_field], correct_labels)
        examples.append(Example(*[text.strip() for text in texts], correct, human))
    if not examples:
        raise BenchmarkError("No usable answers were found in these rows.")
    return examples, skipped


def pearson(first, second):
    if len(first) < 2:
        return None
    mean_a = sum(first) / len(first)
    mean_b = sum(second) / len(second)
    spread_a = sqrt(sum((a - mean_a) ** 2 for a in first))
    spread_b = sqrt(sum((b - mean_b) ** 2 for b in second))
    if spread_a == 0 or spread_b == 0:
        return None
    covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(first, second, strict=True))
    return covariance / (spread_a * spread_b)


def class_f1(predicted, actual, positive):
    hits = sum(1 for p, a in zip(predicted, actual, strict=True) if p == positive and a == positive)
    claimed = sum(1 for p in predicted if p == positive)
    real = sum(1 for a in actual if a == positive)
    precision = hits / claimed if claimed else 0.0
    recall = hits / real if real else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_predictions(predicted, examples, scores):
    actual = [example.correct for example in examples]
    precision, recall, f1 = class_f1(predicted, actual, True)
    _, _, negative_f1 = class_f1(predicted, actual, False)
    both = any(actual) and not all(actual)
    human = [example.human_score for example in examples]
    numeric = pearson(scores, human) if all(value is not None for value in human) else None
    return GradingScores(
        count=len(examples),
        accuracy=sum(1 for p, a in zip(predicted, actual, strict=True) if p == a) / len(examples),
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=(f1 + negative_f1) / 2,
        kappa=cohen_kappa([str(p) for p in predicted], [str(a) for a in actual]),
        auroc=auroc(scores, actual) if both else None,
        pearson=numeric,
    )


def evaluate_grader(examples, grade, threshold, skipped=0, name="model grader"):
    model_scores = [grade(item.question, item.reference, item.answer) for item in examples]
    keyword_scores = [keyword_verdict(item.reference, item.answer).score for item in examples]
    majority = sum(1 for item in examples if item.correct) * 2 >= len(examples)
    systems = {
        name: score_predictions([s >= threshold for s in model_scores], examples, model_scores),
        "keyword overlap": score_predictions(
            [s >= threshold for s in keyword_scores], examples, keyword_scores
        ),
        "always the majority label": score_predictions(
            [majority] * len(examples), examples, [1.0 if majority else 0.0] * len(examples)
        ),
    }
    return GradingReport(
        count=len(examples),
        correct_share=sum(1 for item in examples if item.correct) / len(examples),
        skipped=skipped,
        systems=systems,
    )


def model_grade_function(grader):
    def grade(question, reference, answer):
        return grader.grade(question, reference, answer, SimpleNamespace(text=reference)).score

    return grade


def report_rows(report):
    rows = []
    for name, scores in report.systems.items():
        rows.append(
            {
                "system": name,
                "accuracy": round(scores.accuracy, 4),
                "precision": round(scores.precision, 4),
                "recall": round(scores.recall, 4),
                "f1": round(scores.f1, 4),
                "macro_f1": round(scores.macro_f1, 4),
                "kappa": round(scores.kappa, 4),
                "auroc": "" if scores.auroc is None else round(scores.auroc, 4),
                "pearson": "" if scores.pearson is None else round(scores.pearson, 4),
            }
        )
    return rows
