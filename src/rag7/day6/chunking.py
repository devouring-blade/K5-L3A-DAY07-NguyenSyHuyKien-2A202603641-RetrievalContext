"""Implement fixed-size and structure-aware chunking.

Cả hai hàm làm việc trên từng trang đã chuẩn hóa, nên mỗi chunk luôn thuộc đúng
một trang (truy vết `page` chính xác). Chunk ID ổn định theo
(strategy, document_id, page, thứ tự) để nạp lại không bị nhân đôi.
"""
import re

from .ingestion import detect_heading
from .models import Chunk, DocumentPage


def _base_metadata(page: DocumentPage, section: str, strategy: str, idx: int) -> dict:
    sections = page.metadata.get("known_sections", {})
    return {
        "document_id": page.document_id,
        "source": page.source,
        "page": page.page,
        "chapter": page.metadata["chapter"],
        "section": section,
        "section_title": sections.get(section, ""),
        "strategy": strategy,
        "chunk_index": idx,
    }


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Cửa sổ ký tự cố định có chồng lấn; không quan tâm câu/đoạn.

    `section` = mục hiệu lực ở đầu trang (A không phân tích heading), nên chunk
    nằm sau một heading giữa trang có thể mang mục cũ — đây là hạn chế có chủ đích.
    """
    if overlap >= size:
        raise ValueError("overlap phải nhỏ hơn size")
    text, chunks, step, idx = page.text, [], size - overlap, 0
    section = page.metadata["section_at_start"]
    for start in range(0, max(len(text), 1), step):
        piece = text[start:start + size]
        if not piece.strip():
            continue
        end = start + len(piece)
        meta = _base_metadata(page, section, "A", idx)
        meta.update(char_start=start, char_end=end)
        chunks.append(Chunk(
            chunk_id=f"A:{page.document_id}:p{page.page}:{idx}",
            document_id=page.document_id, source=page.source, page=page.page,
            text=piece, metadata=meta))
        idx += 1
        if end >= len(text):
            break
    return chunks


_SENT_SPLIT = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9•(\"'])")


def _sentences(block: str) -> list[str]:
    """Tách câu thô. PDF ngắt dòng giữa câu nên nối dòng trước khi tách,
    nhưng giữ dòng bắt đầu bằng '•' làm câu riêng (mục liệt kê)."""
    units, buf = [], []
    for line in block.split("\n"):
        if line.startswith("•") and buf:
            units.append(" ".join(buf)); buf = []
        buf.append(line)
    if buf:
        units.append(" ".join(buf))
    out = []
    for u in units:
        out.extend(s.strip() for s in _SENT_SPLIT.split(u) if s.strip())
    return out


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Chunk theo cấu trúc: cắt tại heading cấp mục (X.Y), trong mỗi mục gom
    các câu liền nhau đến tối đa `max_size` ký tự, không bao giờ cắt giữa câu.

    pypdf không giữ dòng trống giữa các đoạn, nên "đoạn" được xấp xỉ bằng ranh
    giới mục + ranh giới câu. Câu dài hơn max_size được giữ nguyên (không cắt).
    """
    chapter = page.metadata["chapter"]
    known = page.metadata.get("known_sections", {})
    detect = not page.metadata.get("is_chapter_opener", False)

    # 1) Chia trang thành các khối theo heading.
    blocks, current, lines = [], page.metadata["section_at_start"], []
    for line in page.text.split("\n"):
        sec = detect_heading(line, chapter, known) if detect else None
        if sec:
            if any(l.strip() for l in lines):
                blocks.append((current, "\n".join(lines)))
            current, lines = sec, [line]      # heading đi cùng nội dung mục mới
        else:
            lines.append(line)
    if any(l.strip() for l in lines):
        blocks.append((current, "\n".join(lines)))

    # 2) Trong mỗi khối, gom câu.
    chunks, idx = [], 0
    for section, block in blocks:
        buf = ""
        for sent in _sentences(block):
            if buf and len(buf) + 1 + len(sent) > max_size:
                chunks.append((section, buf)); buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            chunks.append((section, buf))

    result = []
    for section, text in chunks:
        meta = _base_metadata(page, section, "B", idx)
        result.append(Chunk(
            chunk_id=f"B:{page.document_id}:p{page.page}:{idx}",
            document_id=page.document_id, source=page.source, page=page.page,
            text=text, metadata=meta))
        idx += 1
    return result
