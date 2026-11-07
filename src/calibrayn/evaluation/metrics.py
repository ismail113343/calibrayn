import math
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class RankedScores:
    queries: int
    hit_at: dict
    recall_at: dict
    ndcg_at: dict
    mrr: float
    misses: int


@dataclass(frozen=True)
class ChanceScores:
    hit_at: dict
    recall_at: dict


def chance_hit_at(corpus_size, gold_count, k):
    if corpus_size < 1 or gold_count < 1:
        raise ValueError("corpus_size and gold_count must be at least 1")
    picks = min(k, corpus_size)
    if gold_count >= corpus_size:
        return 1.0
    missed = 1.0
    for step in range(picks):
        missed *= max(0.0, (corpus_size - gold_count - step) / (corpus_size - step))
    return 1.0 - missed


def chance_scores(corpus_size, gold_counts, ks):
    return ChanceScores(
        hit_at={k: fmean(chance_hit_at(corpus_size, g, k) for g in gold_counts) for k in ks},
        recall_at={k: min(k, corpus_size) / corpus_size for k in ks},
    )


def ndcg(ranks, gold_count, k):
    gains = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank is not None and rank <= k)
    ideal = sum(1.0 / math.log2(place + 1) for place in range(1, min(gold_count, k) + 1))
    return gains / ideal


def score_rankings(gold_ranks, ks):
    if not gold_ranks:
        raise ValueError("There are no queries to score.")
    found = [[rank for rank in ranks if rank is not None] for ranks in gold_ranks]
    hit_at = {
        k: fmean(1.0 if any(rank <= k for rank in ranks) else 0.0 for ranks in found) for k in ks
    }
    recall_at = {
        k: fmean(
            sum(1 for rank in ranks if rank <= k) / len(all_ranks)
            for ranks, all_ranks in zip(found, gold_ranks, strict=True)
        )
        for k in ks
    }
    ndcg_at = {k: fmean(ndcg(all_ranks, len(all_ranks), k) for all_ranks in gold_ranks) for k in ks}
    reciprocal = [1.0 / min(ranks) if ranks else 0.0 for ranks in found]
    return RankedScores(
        queries=len(gold_ranks),
        hit_at=hit_at,
        recall_at=recall_at,
        ndcg_at=ndcg_at,
        mrr=fmean(reciprocal),
        misses=sum(1 for ranks in found if not ranks),
    )
