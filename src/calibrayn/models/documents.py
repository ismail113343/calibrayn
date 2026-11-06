from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class LoadedFile:
    unit: str
    pages: tuple[Page, ...]


@dataclass(frozen=True)
class ChunkDraft:
    position: int
    page: int
    title: str
    text: str


@dataclass(frozen=True)
class Document:
    id: int
    name: str
    unit: str
    added_at: datetime


@dataclass(frozen=True)
class Chunk:
    id: int
    document_id: int
    position: int
    page: int
    title: str
    text: str
