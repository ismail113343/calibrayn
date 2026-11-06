import hashlib

import numpy as np

from calibrayn.text import content_words

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbedderError(ValueError):
    pass


class HashingEmbedder:
    def __init__(self, dimensions=384):
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions
        self.name = f"hashing-{dimensions}"

    def embed_documents(self, texts):
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text):
        return self._vector(text)

    def _vector(self, text):
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for word in content_words(text):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "little")
            vector[number % self.dimensions] += 1.0 if number >> 63 == 0 else -1.0
        length = np.linalg.norm(vector)
        return vector / length if length else vector


class SentenceTransformerEmbedder:
    def __init__(self, model_name, model=None):
        self.name = model_name
        self._model = model
        self._query_prefix = QUERY_PREFIX if "bge" in model_name.lower() else ""

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise EmbedderError(
                    "The huggingface embedder needs sentence-transformers. "
                    "Run: pip install -r requirements-ml.txt"
                ) from error
            self._model = SentenceTransformer(self.name)
        return self._model

    def embed_documents(self, texts):
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._load().encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text):
        vectors = self._load().encode([self._query_prefix + text], normalize_embeddings=True)
        return np.asarray(vectors[0], dtype=np.float32)


def build_embedder(settings):
    choice = settings.embedder.strip().lower()
    if choice == "hashing":
        return HashingEmbedder()
    if choice == "huggingface":
        return SentenceTransformerEmbedder(settings.hf_embedding_model)
    raise EmbedderError(f"Unknown EMBEDDER '{settings.embedder}'. Use hashing or huggingface.")
