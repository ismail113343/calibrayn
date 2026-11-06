from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    score: float


class NumpyIndex:
    def __init__(self):
        self._ids = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    def __len__(self):
        return len(self._ids)

    def rebuild(self, ids, matrix):
        self._ids = list(ids)
        if self._ids:
            self._matrix = np.asarray(matrix, dtype=np.float32).reshape(len(self._ids), -1)
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._changed()

    def add(self, ids, matrix):
        ids = list(ids)
        if not ids:
            return
        matrix = np.asarray(matrix, dtype=np.float32).reshape(len(ids), -1)
        if not self._ids:
            self.rebuild(ids, matrix)
            return
        if matrix.shape[1] != self._matrix.shape[1]:
            raise ValueError("The new vectors do not match the index size.")
        self._ids.extend(ids)
        self._matrix = np.vstack([self._matrix, matrix])
        self._changed()

    def remove(self, ids):
        drop = set(ids)
        keep = [position for position, chunk_id in enumerate(self._ids) if chunk_id not in drop]
        self._ids = [self._ids[position] for position in keep]
        self._matrix = self._matrix[keep]
        self._changed()

    def search(self, vector, k):
        if not self._ids:
            return []
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape[0] != self._matrix.shape[1]:
            raise ValueError("The query vector does not match the index size.")
        scores = self._matrix @ vector
        best = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self._ids[position], float(scores[position])) for position in best]

    def _changed(self):
        return None


class FaissIndex(NumpyIndex):
    def __init__(self):
        super().__init__()
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                "FaissIndex needs faiss-cpu. Run: pip install -r requirements-ml.txt"
            ) from error
        self._faiss = faiss
        self._engine = None

    def _changed(self):
        if not self._ids:
            self._engine = None
            return
        engine = self._faiss.IndexFlatIP(self._matrix.shape[1])
        engine.add(np.ascontiguousarray(self._matrix))
        self._engine = engine

    def search(self, vector, k):
        if self._engine is None:
            return []
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self._matrix.shape[1]:
            raise ValueError("The query vector does not match the index size.")
        scores, positions = self._engine.search(np.ascontiguousarray(vector), min(k, len(self)))
        return [
            Hit(self._ids[position], float(score))
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]


def build_index(settings):
    choice = settings.index.strip().lower()
    if choice == "numpy":
        return NumpyIndex()
    if choice == "faiss":
        return FaissIndex()
    raise ValueError(f"Unknown INDEX '{settings.index}'. Use numpy or faiss.")
