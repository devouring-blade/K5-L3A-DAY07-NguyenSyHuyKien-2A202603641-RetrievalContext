"""Đường dẫn, biến môi trường và bảng giá dùng chung cho toàn bộ Day 7.

Mọi lệnh chạy từ THƯ MỤC GỐC repo:  python -m src.rag7.<tên_module>
"""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]          # thư mục gốc repo Day 7
load_dotenv(REPO / ".env")

# Dữ liệu Day 6 (PDF + manifest) và dữ liệu đã chuyển sang schema Day 7
DAY6_MANIFEST = REPO / "data" / "day6" / "manifest.json"
CHUNKS_JSON = REPO / "data" / "day7" / "chunks.json"
INDEX_INFO = REPO / "data" / "day7" / "index_info.json"
QUESTIONS_JSON = REPO / "benchmark" / "questions.json"   # câu có đáp án (gold)
CASES_JSON = REPO / "benchmark" / "cases.json"           # mọi câu, cờ answerable

CHROMA_PATH = REPO / "chroma_db"          # vector store Day 6 (Chroma, bền vững)
COLLECTION = "B"                          # -> collection "day06_B" (chiến lược B/C Day 6)
CHUNK_MAX_CHARS = 1000                    # tham số chunk B ở Day 6

RESULTS = REPO / "results"
CACHE_DIR = RESULTS / "cache"

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL") or "text-embedding-3-small"
os.environ.setdefault("EMBEDDING_MODEL", EMBEDDING_MODEL)   # mã Day 6 đọc biến này
AGENT_MODEL = os.environ.get("AGENT_MODEL") or "gpt-4.1-mini"

# Giá USD / 1 triệu token theo bảng giá công khai của OpenAI (kiểm tra lại khi nộp).
# Có thể ghi đè bằng .env: AGENT_PRICE_IN_PER_1M, AGENT_PRICE_OUT_PER_1M.
_KNOWN_PRICES = {
    "text-embedding-3-small": (0.02, 0.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
}


def price_per_1m(model: str) -> tuple[float, float] | None:
    """(giá input, giá output) USD/1M token; None nếu không biết giá."""
    if model == AGENT_MODEL and os.environ.get("AGENT_PRICE_IN_PER_1M"):
        return (float(os.environ["AGENT_PRICE_IN_PER_1M"]),
                float(os.environ.get("AGENT_PRICE_OUT_PER_1M", "0")))
    return _KNOWN_PRICES.get(model)
