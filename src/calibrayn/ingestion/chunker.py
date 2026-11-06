import re
from dataclasses import dataclass

from calibrayn.models.documents import ChunkDraft
from calibrayn.text import shorten

BREAK = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
TITLE_LIMIT = 60


@dataclass(frozen=True)
class ChunkConfig:
    target_words: int = 500
    overlap_words: int = 50

    def __post_init__(self) -> None:
        if self.target_words < 10:
            raise ValueError("target_words must be at least 10")
        if not 0 <= self.overlap_words < self.target_words:
            raise ValueError("overlap_words must be at least 0 and below target_words")


DEFAULT_CONFIG = ChunkConfig()


def pieces(pages, limit):
    for page in pages:
        for part in BREAK.split(page.text):
            tokens = " ".join(part.split()).split(" ")
            if tokens == [""]:
                continue
            for start in range(0, len(tokens), limit):
                yield page.number, " ".join(tokens[start : start + limit])


def overlap_tail(current, limit):
    tail = []
    total = 0
    for number, text in reversed(current):
        size = len(text.split())
        if total + size > limit:
            break
        tail.insert(0, (number, text))
        total += size
    return tail


def build_draft(position, current, fresh):
    first = len(current) - fresh
    return ChunkDraft(
        position=position,
        page=current[first][0],
        title=shorten(current[first][1], TITLE_LIMIT),
        text=" ".join(text for _, text in current),
    )


def make_chunks(pages, config=DEFAULT_CONFIG):
    drafts = []
    current = []
    fresh = 0
    count = 0
    for number, text in pieces(pages, config.target_words):
        size = len(text.split())
        if fresh and count + size > config.target_words:
            drafts.append(build_draft(len(drafts), current, fresh))
            current = overlap_tail(current, config.overlap_words)
            count = sum(len(piece.split()) for _, piece in current)
            fresh = 0
        current.append((number, text))
        count += size
        fresh += 1
    if fresh:
        drafts.append(build_draft(len(drafts), current, fresh))
    return drafts
