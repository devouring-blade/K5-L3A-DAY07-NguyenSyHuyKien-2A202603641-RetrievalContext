"""Bước 1: dựng lại vector store Day 6 và xuất chunk theo schema Day 7.

Chạy:  python -m src.rag7.build_index

- Đọc PDF theo manifest Day 6, chunk bằng đúng hàm `by_paragraph(max 1000)` của
  chiến lược B/C Day 6 -> 242 chunk, ID ổn định dạng `B:<doc>:p<trang>:<thứ tự>`.
- Embedding bằng OpenAI text-embedding-3-small (1536 chiều) như Day 6, lưu vào
  Chroma (thư mục chroma_db/, collection day06_B, cosine).
- Nạp lại không nhân đôi: chunk nào nội dung không đổi thì bỏ qua (so hash).
- Xuất data/day7/chunks.json (id, text, source_id, source, page, metadata) và
  kiểm tra mọi evidence trong benchmark nằm trọn trong một chunk đúng nguồn/trang.
"""
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version

from . import config
from .day6.chunking import by_paragraph
from .day6.embeddings import MyEmbedder
from .day6.ingestion import load_corpus
from .day6.store import ChromaVectorStore, text_hash
from .evidence import gold_hits


def to_day7(chunk) -> dict:
    m = chunk.metadata
    return {"id": chunk.chunk_id, "text": chunk.text, "source_id": chunk.document_id,
            "source": chunk.source, "page": chunk.page,
            "metadata": {"chapter": m["chapter"], "section": m["section"],
                         "section_title": m["section_title"], "chunk_index": m["chunk_index"]}}


def main() -> None:
    t0 = time.perf_counter()
    pages, report = load_corpus(config.DAY6_MANIFEST)
    chunks = [c for p in pages for c in by_paragraph(p, config.CHUNK_MAX_CHARS)]
    print(f"Đọc {len(pages)} trang nội dung -> {len(chunks)} chunk")

    embedder = MyEmbedder(config.EMBEDDING_MODEL)
    store = ChromaVectorStore(str(config.CHROMA_PATH), embedder.dimensions, embedder.model_name)
    existing = store.existing_hashes(config.COLLECTION)
    todo = [c for c in chunks if existing.get(c.chunk_id) != text_hash(c.text)]
    if todo:
        store.upsert(config.COLLECTION, todo, embedder.embed_documents([c.text for c in todo]))
    removed = []
    for doc_id in sorted({c.document_id for c in chunks}):
        keep = {c.chunk_id for c in chunks if c.document_id == doc_id}
        removed += store.delete_obsolete(config.COLLECTION, doc_id, keep)
    count = store.count(config.COLLECTION)
    if count != len(chunks):
        raise RuntimeError(f"Store có {count} chunk, mong đợi {len(chunks)} — kiểm tra nhân đôi")
    print(f"Embed mới {len(todo)}, bỏ qua (không đổi) {len(chunks) - len(todo)}, "
          f"xóa cũ {len(removed)}, store hiện có {count}")

    day7_chunks = [to_day7(c) for c in chunks]
    config.CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_JSON.write_text(json.dumps(day7_chunks, ensure_ascii=False, indent=1) + "\n",
                                  encoding="utf-8")

    # Kiểm tra gold: mỗi câu có đáp án phải có evidence nằm trọn trong >=1 chunk.
    questions = json.loads(config.QUESTIONS_JSON.read_text(encoding="utf-8"))
    missing = [q["id"] for q in questions if not any(gold_hits(c, q["gold"]) for c in day7_chunks)]

    tokens = embedder.total_tokens
    price = config.price_per_1m(embedder.model_name)
    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pages": len(pages), "chunks": len(chunks),
        "chunking": f"Day 6 strategy B: by_paragraph(max_size={config.CHUNK_MAX_CHARS})",
        "embedding_model": embedder.model_name, "dimensions": embedder.dimensions,
        "vector_store": {"name": "Chroma PersistentClient", "path": "chroma_db/",
                         "collection": f"day06_{config.COLLECTION}", "metric": "cosine",
                         "search": "approximate (HNSW)"},
        "embedded_now": len(todo), "skipped_unchanged": len(chunks) - len(todo),
        "removed_obsolete": len(removed), "embedding_tokens": tokens,
        "embedding_cost_usd": round(tokens / 1e6 * price[0], 6) if price else None,
        "seconds": round(time.perf_counter() - t0, 2),
        "libraries": {n: version(n) for n in ("chromadb", "openai", "pypdf")},
        "gold_not_in_single_chunk": missing,
        "corpus_report": report,
    }
    config.INDEX_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã ghi {config.CHUNKS_JSON.relative_to(config.REPO)} và {config.INDEX_INFO.relative_to(config.REPO)}")
    if missing:
        print(f"CẢNH BÁO: evidence của {missing} không nằm trọn trong một chunk")
    else:
        print(f"Kiểm tra gold: {len(questions)}/{len(questions)} câu có evidence nằm trọn trong một chunk")


if __name__ == "__main__":
    main()
