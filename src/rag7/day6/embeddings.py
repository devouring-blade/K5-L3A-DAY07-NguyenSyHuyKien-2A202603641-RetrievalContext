"""Real embedding provider: OpenAI Embeddings API."""
import os
from typing import Protocol

from openai import OpenAI


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


# Số chiều mặc định do OpenAI công bố cho từng model.
_KNOWN_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class MyEmbedder:
    """OpenAI embeddings. Query và document dùng CÙNG model, không tiền tố.

    Không có fallback: nếu API lỗi (sai key, hết quota, mất mạng) thì exception
    được ném ra để pipeline dừng — không bao giờ thay bằng vector giả.
    """

    def __init__(self, model_name: str | None = None, batch_size: int = 100) -> None:
        self.model_name = model_name or os.environ.get("EMBEDDING_MODEL", "")
        if not self.model_name:
            raise RuntimeError("Thiếu EMBEDDING_MODEL trong .env")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Thiếu OPENAI_API_KEY trong .env")
        self.client = OpenAI()
        self.batch_size = batch_size
        self.total_tokens = 0          # cộng dồn để tính chi phí
        self.dimensions = _KNOWN_DIMS.get(self.model_name) or len(self.embed_query("dimension probe"))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            self.total_tokens += resp.usage.total_tokens
            # API trả về đúng thứ tự, vẫn sắp theo index cho chắc chắn.
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question])[0]
