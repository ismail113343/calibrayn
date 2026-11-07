from dataclasses import dataclass

from calibrayn.evaluation.benchmark import BenchmarkError, flatten_passages
from calibrayn.rag.grounding import support_ratio

TRUE_WORDS = {"true", "yes", "1", "supported", "faithful"}
FALSE_WORDS = {"false", "no", "0", "unsupported", "hallucinated"}
THRESHOLDS = tuple(step / 20 for step in range(1, 20))


@dataclass(frozen=True)
class GroundednessScores:
    count: int
    faithful_share: float
    auroc: float
    default_threshold: float
    default_accuracy: float
    best_threshold: float
    best_accuracy: float


def as_flag(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in TRUE_WORDS:
            return True
        if word in FALSE_WORDS:
            return False
    raise BenchmarkError(f"The label {value!r} is not true or false.")


def auroc(scores, flags):
    positives = sum(1 for flag in flags if flag)
    negatives = len(flags) - positives
    if positives == 0 or negatives == 0:
        raise BenchmarkError("Both faithful and unfaithful answers are needed to score this.")
    order = sorted(range(len(scores)), key=lambda place: scores[place])
    ranks = [0.0] * len(scores)
    place = 0
    while place < len(order):
        end = place
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[place]]:
            end += 1
        for member in order[place : end + 1]:
            ranks[member] = (place + end) / 2.0 + 1.0
        place = end + 1
    rank_sum = sum(rank for rank, flag in zip(ranks, flags, strict=True) if flag)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def accuracy_at(scores, flags, threshold):
    correct = sum(
        1 for score, flag in zip(scores, flags, strict=True) if (score >= threshold) == flag
    )
    return correct / len(flags)


def evaluate_groundedness(rows, response_field, documents_field, label_field, default_threshold):
    scores = []
    flags = []
    for row in rows:
        missing = [
            name for name in (response_field, documents_field, label_field) if name not in row
        ]
        if missing:
            raise BenchmarkError(
                f"The columns {', '.join(missing)} were not found. "
                f"This dataset has: {', '.join(sorted(row))}."
            )
        response = row[response_field]
        passages = flatten_passages(row[documents_field])
        if not isinstance(response, str) or not response.strip() or not passages:
            continue
        scores.append(support_ratio(response, passages))
        flags.append(as_flag(row[label_field]))
    if not scores:
        raise BenchmarkError("No usable answers were found in these rows.")
    best = max(THRESHOLDS, key=lambda step: accuracy_at(scores, flags, step))
    return GroundednessScores(
        count=len(scores),
        faithful_share=sum(1 for flag in flags if flag) / len(flags),
        auroc=auroc(scores, flags),
        default_threshold=default_threshold,
        default_accuracy=accuracy_at(scores, flags, default_threshold),
        best_threshold=best,
        best_accuracy=accuracy_at(scores, flags, best),
    )
