from dataclasses import dataclass

from calibrayn.models.documents import Chunk


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 4

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")


DEFAULT_CONFIG = RetrievalConfig()


@dataclass(frozen=True)
class Passage:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, library, embedder, index, config=DEFAULT_CONFIG):
        self.library = library
        self.embedder = embedder
        self.index = index
        self.config = config

    def refresh(self):
        if self.library.chunk_count() and self.library.embedder_name() != self.embedder.name:
            chunks = self.library.chunks()
            vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
            self.library.replace_vectors([chunk.id for chunk in chunks], vectors)
        self.library.set_embedder_name(self.embedder.name)
        ids, matrix = self.library.vectors()
        self.index.rebuild(ids, matrix)

    def add(self, ids, vectors):
        self.index.add(ids, vectors)

    def remove(self, ids):
        self.index.remove(ids)

    def retrieve(self, query, k=None, skip=()):
        wanted = k or self.config.top_k
        if not query.strip() or not len(self.index):
            return []
        hits = self.index.search(self.embedder.embed_query(query), wanted + len(skip))
        passages = []
        for hit in hits:
            if hit.chunk_id in skip:
                continue
            chunk = self.library.chunk(hit.chunk_id)
            if chunk is not None:
                passages.append(Passage(chunk, hit.score))
        return passages[:wanted]
