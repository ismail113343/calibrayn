import csv
import hashlib
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from calibrayn.text import content_words

SHEET_COLUMNS = (
    "id",
    "question",
    "answer",
    "evidence",
    "document",
    "page",
    "level",
    "kind",
    "verdict",
    "question_fixed",
    "answer_fixed",
    "evidence_fixed",
    "reviewer",
    "comments",
)
VERDICTS = ("keep", "fix", "reject")
EASY_FROM = 0.8


class ReviewError(ValueError):
    pass


@dataclass(frozen=True)
class OverlapReport:
    mean: float
    easy_share: float
    easy_ids: tuple


@dataclass(frozen=True)
class ReviewResult:
    items: tuple
    kept: int
    fixed: int
    rejected: int
    pending: int


@dataclass(frozen=True)
class Agreement:
    count: int
    observed: float
    kappa: float


def question_overlap(question, source):
    asked = set(content_words(question))
    if not asked:
        return 0.0
    return len(asked & set(content_words(source))) / len(asked)


def overlap_report(items, library):
    shares = {}
    for item in items:
        chunk = library.chunk(item.chunk_id)
        if chunk is not None:
            shares[item.id] = question_overlap(item.question, chunk.text)
    if not shares:
        raise ReviewError("None of the questions match a topic in the library.")
    easy = tuple(sorted(number for number, share in shares.items() if share >= EASY_FROM))
    return OverlapReport(sum(shares.values()) / len(shares), len(easy) / len(shares), easy)


def near_duplicates(items, threshold=0.9):
    words = {item.id: set(content_words(item.question)) for item in items}
    pairs = []
    ordered = sorted(words)
    for place, first in enumerate(ordered):
        for second in ordered[place + 1 :]:
            union = words[first] | words[second]
            if union and len(words[first] & words[second]) / len(union) >= threshold:
                pairs.append((first, second))
    return pairs


def bucket(seed, key):
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(digest, 16) % 10000 / 10000


def assign_splits(items, validation_share=0.3, seed=0, by="question"):
    if not 0.0 < validation_share < 1.0:
        raise ValueError("validation_share must be between 0 and 1")
    if by not in ("question", "document"):
        raise ValueError("by must be 'question' or 'document'")
    assigned = []
    for item in items:
        key = item.document if by == "document" else item.id
        name = "validation" if bucket(seed, key) < validation_share else "test"
        assigned.append(replace(item, split=name))
    return assigned


def cohen_kappa(first, second):
    if len(first) != len(second) or not first:
        raise ValueError("Both reviewers must label the same, non-empty set of questions.")
    total = len(first)
    observed = sum(1 for a, b in zip(first, second, strict=True) if a == b) / total
    counts_a = Counter(first)
    counts_b = Counter(second)
    expected = sum(counts_a[label] * counts_b[label] for label in counts_a) / (total * total)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def review_summary(items):
    return {
        "questions": len(items),
        "verified": sum(1 for item in items if item.verified),
        "answerable": sum(1 for item in items if item.answerable),
        "by_level": dict(sorted(Counter(item.level for item in items).items())),
        "by_kind": dict(sorted(Counter(item.kind for item in items).items())),
        "by_split": dict(sorted(Counter(item.split or "none" for item in items).items())),
        "documents": len({item.document for item in items}),
    }


def export_review_sheet(items, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for item in items:
            row = {name: "" for name in SHEET_COLUMNS}
            row.update(
                id=item.id,
                question=item.question,
                answer=item.answer,
                evidence=item.evidence,
                document=item.document,
                page=item.page,
                level=item.level,
                kind=item.kind,
            )
            writer.writerow(row)
    return target


def read_sheet(path):
    source = Path(path)
    if not source.is_file():
        raise ReviewError(f"The review sheet {source} was not found.")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(SHEET_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ReviewError(f"The review sheet is missing columns: {', '.join(sorted(missing))}.")
        return list(reader)


def verdict_of(row):
    verdict = (row.get("verdict") or "").strip().lower()
    if verdict and verdict not in VERDICTS:
        raise ReviewError(
            f"Question {row['id']} has the verdict '{verdict}'. Use keep, fix or reject."
        )
    return verdict


def import_review_sheet(items, path, reviewer):
    if not reviewer.strip():
        raise ReviewError("A reviewer name is needed.")
    rows = {int(row["id"]): row for row in read_sheet(path)}
    result = []
    kept = fixed = rejected = pending = 0
    for item in items:
        row = rows.get(item.id)
        verdict = verdict_of(row) if row else ""
        if verdict == "reject":
            rejected += 1
            continue
        if verdict == "keep":
            kept += 1
            item = replace(item, verified=True, reviewer=reviewer)
        elif verdict == "fix":
            fixed += 1
            item = replace(
                item,
                question=(row["question_fixed"] or item.question).strip(),
                answer=(row["answer_fixed"] or item.answer).strip(),
                evidence=(row["evidence_fixed"] or item.evidence).strip(),
                verified=True,
                reviewer=reviewer,
            )
        else:
            pending += 1
        result.append(item)
    return ReviewResult(tuple(result), kept, fixed, rejected, pending)


def review_agreement(path_a, path_b):
    first = {int(row["id"]): verdict_of(row) for row in read_sheet(path_a)}
    second = {int(row["id"]): verdict_of(row) for row in read_sheet(path_b)}
    shared = sorted(number for number in first if first[number] and second.get(number))
    if not shared:
        raise ReviewError("The two sheets have no questions that both reviewers marked.")
    labels_a = [first[number] for number in shared]
    labels_b = [second[number] for number in shared]
    observed = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / len(shared)
    return Agreement(len(shared), observed, cohen_kappa(labels_a, labels_b))
