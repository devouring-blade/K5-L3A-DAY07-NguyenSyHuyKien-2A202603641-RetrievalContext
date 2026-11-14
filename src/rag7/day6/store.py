"""Store and search vectors using Chroma (persistent, local).

Thay cho PostgreSQL + pgvector của starter vì máy không có Docker (GUIDE cho phép
Chroma). Mỗi chiến lược chunking có một collection riêng, metric cosine.
"""
import hashlib
from typing import Any

import chromadb

from .models import Chunk


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ChromaVectorStore:
    def __init__(self, path: str, dimensions: int, model_name: str) -> None:
        self.client = chromadb.PersistentClient(path=path)
        self.dimensions = dimensions
        self.model_name = model_name

    def _collection(self, strategy: str):
        col = self.client.get_or_create_collection(
            name=f"day06_{strategy}",
            metadata={"hnsw:space": "cosine", "embedding_model": self.model_name,
                      "dimensions": self.dimensions},
            embedding_function=None,   # vector do MyEmbedder tạo, Chroma không tự embed
        )
        stored_model = (col.metadata or {}).get("embedding_model")
        if stored_model and stored_model != self.model_name:
            raise RuntimeError(
                f"Collection {col.name} được tạo bằng model {stored_model}, "
                f"không dùng chung với {self.model_name}. Xóa thư mục chroma_db để nạp lại.")
        return col

    def existing_hashes(self, strategy: str) -> dict[str, str]:
        """{chunk_id: text_hash} đang có trong store — để bỏ qua chunk không đổi."""
        got = self._collection(strategy).get(include=["metadatas"])
        return {i: (m or {}).get("text_hash", "") for i, m in zip(got["ids"], got["metadatas"])}

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Ghi theo ID ổn định (upsert ⇒ nạp lại không nhân đôi)."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunk nhưng {len(vectors)} vector")
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError(f"Vector {len(v)} chiều, store yêu cầu {self.dimensions}")
        if not chunks:
            return
        col = self._collection(strategy)
        col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{**c.metadata, "text_hash": text_hash(c.text)} for c in chunks],
        )

    def delete_obsolete(self, strategy: str, document_id: str, keep_ids: set[str]) -> list[str]:
        """Khi tài liệu đổi, xóa chunk cũ của tài liệu đó không còn trong lần nạp mới."""
        col = self._collection(strategy)
        old = col.get(where={"document_id": document_id})["ids"]
        stale = [i for i in old if i not in keep_ids]
        if stale:
            col.delete(ids=stale)
        return stale

    def count(self, strategy: str) -> int:
        return self._collection(strategy).count()

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Lọc metadata ngay trong Chroma (where) rồi xếp hạng theo cosine.

        score = 1 - cosine_distance (càng cao càng giống).
        """
        if len(query_vector) != self.dimensions:
            raise ValueError("Query vector sai số chiều")
        where = None
        if filters:
            clauses = [{k: v} for k, v in filters.items()]
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        res = self._collection(strategy).query(
            query_embeddings=[query_vector], n_results=top_k, where=where,
            include=["documents", "metadatas", "distances"])
        hits = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), 1):
            hits.append({"rank": rank, "chunk_id": cid, "score": round(1 - dist, 6),
                         "text": doc, "metadata": meta})
        return hits
