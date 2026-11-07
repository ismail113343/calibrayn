import json
from dataclasses import asdict, dataclass
from pathlib import Path

from calibrayn.models.learning import Support
from calibrayn.text import sentences

LEVELS = (2, 3, 4, 5)
KINDS = {2: "recall", 3: "explain", 4: "reason", 5: "apply"}
FIELDS = ("id", "question", "answer", "chunk_id", "document", "page", "level", "written_by")
OPTIONAL = {
    "answerable": True,
    "kind": "recall",
    "evidence": "",
    "verified": False,
    "reviewer": "",
    "split": "",
}


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class QAItem:
    id: int
    question: str
    answer: str
    chunk_id: int
    document: str
    page: int
    level: int
    written_by: str
    answerable: bool = True
    kind: str = "recall"
    evidence: str = ""
    verified: bool = False
    reviewer: str = ""
    split: str = ""


def evidence_from(chunk_text, answer):
    flat = " ".join(chunk_text.split())
    for sentence in sentences(answer):
        if sentence in flat:
            return sentence
    return ""


def build_dataset(library, writer, count, written_by, levels=LEVELS):
    if count < 1:
        raise ValueError("count must be at least 1")
    chunks = library.chunks()
    if not chunks:
        raise DatasetError("Add some notes first, then build the questions.")
    names = {document.id: document.name for document in library.documents()}
    items = []
    seen = set()
    attempts = 0
    limit = count * 6
    while len(items) < count and attempts < limit:
        chunk = chunks[attempts % len(chunks)]
        level = levels[(attempts // len(chunks)) % len(levels)]
        attempts += 1
        made = writer.question(chunk, level, Support.DIRECT_RECALL)
        key = made.text.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(
            QAItem(
                id=len(items) + 1,
                question=made.text.strip(),
                answer=made.answer_key,
                chunk_id=chunk.id,
                document=names.get(chunk.document_id, ""),
                page=chunk.page,
                level=level,
                written_by=written_by,
                kind=KINDS.get(level, "recall"),
                evidence=evidence_from(chunk.text, made.answer_key),
            )
        )
    return items


def save_dataset(items, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(item), ensure_ascii=False) for item in items]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def load_dataset(path):
    source = Path(path)
    if not source.is_file():
        raise DatasetError(f"The dataset file {source} was not found.")
    items = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
            required = {name: loaded[name] for name in FIELDS}
            optional = {name: loaded.get(name, default) for name, default in OPTIONAL.items()}
            items.append(QAItem(**required, **optional))
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise DatasetError(f"Line {number} of {source} is not a valid question.") from error
    if not items:
        raise DatasetError(f"The dataset file {source} has no questions.")
    return items
