from datetime import datetime

import numpy as np

from calibrayn.models.documents import Chunk, Document


def _document(row):
    return Document(row["id"], row["name"], row["unit"], datetime.fromisoformat(row["added_at"]))


def _chunk(row):
    return Chunk(
        row["id"], row["document_id"], row["position"], row["page"], row["title"], row["text"]
    )


class Library:
    def __init__(self, database):
        self.database = database

    def add_document(self, name, unit, drafts, vectors):
        added_at = datetime.now()
        stored = []
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO documents (name, unit, added_at) VALUES (?, ?, ?)",
                (name, unit, added_at.isoformat()),
            )
            document_id = cursor.lastrowid
            for draft, vector in zip(drafts, vectors, strict=True):
                inserted = connection.execute(
                    "INSERT INTO chunks (document_id, position, page, title, text, vector)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        document_id,
                        draft.position,
                        draft.page,
                        draft.title,
                        draft.text,
                        np.asarray(vector, dtype=np.float32).tobytes(),
                    ),
                )
                stored.append(
                    Chunk(
                        inserted.lastrowid,
                        document_id,
                        draft.position,
                        draft.page,
                        draft.title,
                        draft.text,
                    )
                )
        return Document(document_id, name, unit, added_at), stored

    def documents(self):
        rows = self.database.read("SELECT * FROM documents ORDER BY id")
        return [_document(row) for row in rows]

    def document(self, document_id):
        rows = self.database.read("SELECT * FROM documents WHERE id = ?", (document_id,))
        return _document(rows[0]) if rows else None

    def delete_document(self, document_id):
        ids = [chunk.id for chunk in self.chunks(document_id)]
        self.database.write("DELETE FROM documents WHERE id = ?", (document_id,))
        return ids

    def chunks(self, document_id=None):
        if document_id is None:
            rows = self.database.read("SELECT * FROM chunks ORDER BY document_id, position")
        else:
            rows = self.database.read(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY position", (document_id,)
            )
        return [_chunk(row) for row in rows]

    def chunk(self, chunk_id):
        rows = self.database.read("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        return _chunk(rows[0]) if rows else None

    def chunk_count(self):
        return self.database.read("SELECT COUNT(*) AS total FROM chunks")[0]["total"]

    def vectors(self):
        rows = self.database.read(
            "SELECT id, vector FROM chunks WHERE vector IS NOT NULL ORDER BY document_id, position"
        )
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [row["id"] for row in rows]
        matrix = np.vstack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows])
        return ids, matrix

    def replace_vectors(self, ids, vectors):
        with self.database.transaction() as connection:
            for chunk_id, vector in zip(ids, vectors, strict=True):
                connection.execute(
                    "UPDATE chunks SET vector = ? WHERE id = ?",
                    (np.asarray(vector, dtype=np.float32).tobytes(), chunk_id),
                )

    def embedder_name(self):
        rows = self.database.read("SELECT value FROM state WHERE key = 'embedder'")
        return rows[0]["value"] if rows else None

    def set_embedder_name(self, name):
        self.database.write(
            "INSERT INTO state (key, value) VALUES ('embedder', ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (name,),
        )
