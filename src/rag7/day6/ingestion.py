"""Read and normalize source data. (Copy từ Day 6, không sửa logic.)"""
import json
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page; keep its page number and source.

    Detect and record pages with no text. OCR, when needed, is a separate step.
    """
    reader = PdfReader(path)
    pages = []
    for i, pdf_page in enumerate(reader.pages):
        raw = pdf_page.extract_text() or ""
        pages.append(DocumentPage(document_id=document_id,
                                  source=source,
                                  page=i + 1,
                                  text=raw,
                                  metadata={"empty": not raw.strip()}))
    return pages


# --- Chuẩn hóa văn bản -------------------------------------------------------

# Ký tự "trông giống nhau nhưng mã khác nhau" quy về một dạng ASCII để so khớp
# evidence ổn định. NFKC đã xử lý ligature (ﬁ -> fi) và dấu cách đặc biệt.
_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'",   # ‘ ’
    "“": '"', "”": '"',   # “ ”
    "–": "-", "—": "-",   # – —
    "­": "",                   # soft hyphen (vô hình)
})


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding useful structure.

    - Unicode NFKC + quy dấu nháy cong / gạch dài về ASCII.
    - Mỗi dòng: bỏ khoảng trắng hai đầu, gộp nhiều dấu cách thành một.
    - GIỮ ký tự xuống dòng (cấu trúc heading/danh sách), chỉ gộp các dòng trống liên tiếp.
    - Giữ dấu '•' vì nó đánh dấu mục liệt kê.
    """
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# --- Bỏ header/footer đặc thù sách OpenStax -----------------------------------

# Trang chẵn:  "58     2 • Collecting and Preparing Data" + "Access for free at openstax.org"
# Trang lẻ:    "2.1 • Overview of Data Collection Methods     59"
_EVEN_HEADER = re.compile(r"^\d+\s+\d+ • .+$")
_ACCESS_LINE = re.compile(r"^Access for free at openstax\.org$")
_ODD_FOOTER = re.compile(r"^(\d+\.\d+) • (.+?)\s+\d+$")


def strip_page_furniture(text: str) -> tuple[str, list[str], dict[str, str]]:
    """Bỏ dòng header/footer lặp lại ở mọi trang.

    Trả về (text_đã_bỏ, các_dòng_đã_bỏ, {số_mục: tên_mục}) — tên mục lấy từ
    footer trang lẻ, dùng làm danh sách mục đáng tin để nhận diện heading.
    """
    kept, removed, sections = [], [], {}
    for line in text.split("\n"):
        s = line.strip()
        m = _ODD_FOOTER.match(s)
        if m:
            sections[m.group(1)] = m.group(2).strip()
            removed.append(s)
        elif _EVEN_HEADER.match(s) or _ACCESS_LINE.match(s):
            removed.append(s)
        else:
            kept.append(line)
    return "\n".join(kept), removed, sections


def _strip_chapter_opener_art(text: str, chapter: int) -> tuple[str, list[str]]:
    """Trang mở chương kết thúc bằng tên chương + số chương (chữ trang trí)."""
    lines = text.rstrip().split("\n")
    removed = []
    if lines and lines[-1].strip() == str(chapter):
        removed.append(lines.pop().strip())
        if lines:
            removed.append(lines.pop().strip())
    return "\n".join(lines), removed


HEADING_RE = re.compile(r"^(\d+)\.(\d+) (.+)$")


def detect_heading(line: str, chapter: int, known_sections: dict[str, str]) -> str | None:
    """Trả về số mục (vd '2.3') nếu dòng là heading cấp mục của chương này."""
    m = HEADING_RE.match(line.strip())
    if not m or int(m.group(1)) != chapter:
        return None
    num = f"{m.group(1)}.{m.group(2)}"
    if known_sections.get(num) == m.group(3).strip():
        return num
    return None


# --- Nạp toàn bộ corpus theo manifest ----------------------------------------

def load_corpus(manifest_path: Path) -> tuple[list[DocumentPage], dict]:
    """Đọc PDF theo manifest, chỉ giữ trang được tính, chuẩn hóa và gắn metadata.

    metadata mỗi trang: chapter, section_at_start, section_title_at_start,
    is_chapter_opener, known_sections. Trả về (pages, report) với report ghi
    trang rỗng, dòng header/footer đã bỏ, số ký tự.
    """
    manifest_path = Path(manifest_path)
    base = manifest_path.parent.parent          # starter/
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cache: dict[str, list[DocumentPage]] = {}
    pages: list[DocumentPage] = []
    report = {"documents": []}

    for doc in manifest["documents"]:
        pdf_path = base / doc["local_file"]
        if doc["local_file"] not in cache:
            cache[doc["local_file"]] = load_pdf(pdf_path, "_", doc["local_file"])
        all_pages = cache[doc["local_file"]]
        first, last = doc["pdf_pages_counted"]
        chapter = int(doc["chapter"])

        # Lượt 1: bỏ furniture, thu thập tên mục từ footer.
        stripped, known_sections, removed_count, empty_pages = [], {}, 0, []
        for raw_page in all_pages[first - 1:last]:
            if raw_page.metadata["empty"]:
                empty_pages.append(raw_page.page)
                continue
            text, removed, secs = strip_page_furniture(raw_page.text)
            known_sections.update(secs)
            if raw_page.page == first:
                text, art = _strip_chapter_opener_art(text, chapter)
                removed += art
            removed_count += len(removed)
            stripped.append((raw_page.page, text))

        # Lượt 2: chuẩn hóa, theo dõi mục đang hiệu lực ở đầu mỗi trang.
        current = f"{chapter}.0"
        known_sections.setdefault(current, "Introduction")
        doc_chars = 0
        for page_no, text in stripped:
            clean = normalize_text(text)
            is_opener = page_no == first
            pages.append(DocumentPage(
                document_id=doc["source_id"],
                source=doc["local_file"],
                page=page_no,
                text=clean,
                metadata={
                    "chapter": chapter,
                    "section_at_start": current,
                    "section_title_at_start": known_sections.get(current, ""),
                    "is_chapter_opener": is_opener,
                    "known_sections": known_sections,
                },
            ))
            if not is_opener:   # mục lục ở trang mở chương không phải heading thật
                for line in clean.split("\n"):
                    sec = detect_heading(line, chapter, known_sections)
                    if sec:
                        current = sec
            doc_chars += len(clean)

        report["documents"].append({
            "source_id": doc["source_id"],
            "pages_counted": last - first + 1,
            "pages_loaded": len(stripped),
            "empty_pages": empty_pages,
            "furniture_lines_removed": removed_count,
            "char_count_normalized": doc_chars,
            "sections": dict(sorted(known_sections.items())),
        })
    return pages, report
