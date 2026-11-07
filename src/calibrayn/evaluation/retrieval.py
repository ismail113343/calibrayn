from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class RetrievalScores:
    questions: int
    hit_at: dict
    mrr: float
    misses: int
    hit_at_1_by_level: dict


def rank_of(retriever, item, depth):
    found = retriever.retrieve(item.question, k=depth)
    ids = [passage.chunk.id for passage in found]
    return ids.index(item.chunk_id) + 1 if item.chunk_id in ids else None


def evaluate_retrieval(retriever, items, ks=(1, 3, 5)):
    if not items:
        raise ValueError("There are no questions to check.")
    depth = max(max(ks), retriever.library.chunk_count())
    ranks = [rank_of(retriever, item, depth) for item in items]

    hit_at = {
        k: sum(1 for rank in ranks if rank is not None and rank <= k) / len(items) for k in ks
    }
    reciprocal = [1.0 / rank if rank else 0.0 for rank in ranks]

    by_level = {}
    for level in sorted({item.level for item in items}):
        wanted = [rank for item, rank in zip(items, ranks, strict=True) if item.level == level]
        by_level[level] = sum(1 for rank in wanted if rank == 1) / len(wanted)

    return RetrievalScores(
        questions=len(items),
        hit_at=hit_at,
        mrr=fmean(reciprocal),
        misses=sum(1 for rank in ranks if rank is None),
        hit_at_1_by_level=by_level,
    )
