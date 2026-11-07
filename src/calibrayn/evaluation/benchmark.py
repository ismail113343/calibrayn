from dataclasses import dataclass

import numpy as np

from calibrayn.evaluation.baselines import Bm25
from calibrayn.evaluation.metrics import chance_scores, score_rankings

KS = (1, 3, 5, 10)


class BenchmarkError(ValueError):
    pass


@dataclass(frozen=True)
class Query:
    text: str
    gold: tuple
    level: int | None = None


@dataclass(frozen=True)
class RetrievalBenchmark:
    corpus: tuple
    queries: tuple
    skipped: int = 0


@dataclass(frozen=True)
class BenchmarkReport:
    corpus_size: int
    queries: int
    skipped: int
    systems: dict
    chance: object


def flatten_passages(value):
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list | tuple):
        found = []
        for part in value:
            found.extend(flatten_passages(part))
        return found
    raise BenchmarkError(
        f"A passage field must hold text or a list of text, not {type(value).__name__}."
    )


def from_rows(rows, question_field, context_field):
    corpus = []
    positions = {}
    queries = []
    skipped = 0
    for row in rows:
        if question_field not in row or context_field not in row:
            raise BenchmarkError(
                f"The columns '{question_field}' and '{context_field}' were not both found. "
                f"This dataset has: {', '.join(sorted(row))}."
            )
        question = row[question_field]
        passages = flatten_passages(row[context_field])
        if not isinstance(question, str) or not question.strip() or not passages:
            skipped += 1
            continue
        gold = []
        for passage in passages:
            if passage not in positions:
                positions[passage] = len(corpus)
                corpus.append(passage)
            if positions[passage] not in gold:
                gold.append(positions[passage])
        queries.append(Query(question.strip(), tuple(gold)))
    if not queries:
        raise BenchmarkError("No usable questions were found in these rows.")
    return RetrievalBenchmark(tuple(corpus), tuple(queries), skipped)


def from_library(library, items):
    chunks = library.chunks()
    positions = {chunk.id: place for place, chunk in enumerate(chunks)}
    queries = []
    skipped = 0
    for item in items:
        if item.chunk_id not in positions:
            skipped += 1
            continue
        queries.append(Query(item.question, (positions[item.chunk_id],), item.level))
    if not queries:
        raise BenchmarkError("None of the questions match a topic in the library.")
    return RetrievalBenchmark(tuple(chunk.text for chunk in chunks), tuple(queries), skipped)


def embedder_ranks(benchmark, embedder):
    documents = embedder.embed_documents(list(benchmark.corpus))
    questions = np.vstack([embedder.embed_query(query.text) for query in benchmark.queries])
    similarity = questions @ documents.T
    ranks = []
    for row, query in enumerate(benchmark.queries):
        ranks.append(
            tuple(int(1 + np.sum(similarity[row] > similarity[row, g])) for g in query.gold)
        )
    return ranks


def evaluate_benchmark(benchmark, embedder, ks=KS):
    keyword = Bm25(benchmark.corpus)
    systems = {
        f"embeddings ({embedder.name})": score_rankings(embedder_ranks(benchmark, embedder), ks),
        "keyword search (BM25)": score_rankings(
            [keyword.ranks_of(query.text, query.gold) for query in benchmark.queries], ks
        ),
    }
    return BenchmarkReport(
        corpus_size=len(benchmark.corpus),
        queries=len(benchmark.queries),
        skipped=benchmark.skipped,
        systems=systems,
        chance=chance_scores(len(benchmark.corpus), [len(q.gold) for q in benchmark.queries], ks),
    )


def table_rows(report, ks=KS):
    rows = []
    for name, scores in report.systems.items():
        row = {"system": name}
        for k in ks:
            row[f"hit@{k}"] = round(scores.hit_at[k], 4)
        row["ndcg@10"] = round(scores.ndcg_at.get(10, scores.ndcg_at[max(ks)]), 4)
        row["mrr"] = round(scores.mrr, 4)
        rows.append(row)
    chance = {"system": "random guessing"}
    for k in ks:
        chance[f"hit@{k}"] = round(report.chance.hit_at[k], 4)
    chance["ndcg@10"] = ""
    chance["mrr"] = ""
    rows.append(chance)
    return rows
