"""Bước 0: kiểm tra môi trường trước khi chạy (key, model, thư viện).

Chạy:  python -m src.rag7.check
Gọi API 2 lần rất nhỏ: 1 embedding + 1 chat. Không in API key.
"""
import os
import sys
from importlib.metadata import version

from . import config


def main() -> None:
    print("Python", sys.version.split()[0])
    for lib in ("openai", "chromadb", "pypdf", "python-dotenv"):
        try:
            print(f"  {lib:14s} {version(lib)}")
        except Exception:  # noqa: BLE001
            print(f"  {lib:14s} CHƯA CÀI -> pip install -r requirements.txt")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Thiếu OPENAI_API_KEY trong file .env ở thư mục gốc repo")
    from openai import OpenAI
    client = OpenAI()
    emb = client.embeddings.create(model=config.EMBEDDING_MODEL, input=["ping"])
    print(f"Embedding OK: {config.EMBEDDING_MODEL}, {len(emb.data[0].embedding)} chiều")
    try:
        r = client.chat.completions.create(model=config.AGENT_MODEL,
                                           messages=[{"role": "user", "content": "Reply with OK"}])
        print(f"Agent model OK: {config.AGENT_MODEL} -> {r.choices[0].message.content!r}")
    except Exception as exc:  # noqa: BLE001
        names = sorted(m.id for m in client.models.list() if m.id.startswith(("gpt-", "o")))
        sys.exit(f"Model {config.AGENT_MODEL} lỗi: {exc}\nĐặt AGENT_MODEL=<tên> trong .env. Model khả dụng: {names}")
    if config.price_per_1m(config.AGENT_MODEL) is None:
        print("Lưu ý: chưa có giá của model này -> đặt AGENT_PRICE_IN_PER_1M / AGENT_PRICE_OUT_PER_1M trong .env")
    print("Môi trường sẵn sàng.")


if __name__ == "__main__":
    main()
