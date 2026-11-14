"""Bọc OpenAI API: embedding câu hỏi (có cache) và chat có tool calling.

Đếm token và chi phí cho từng lời gọi. KHÔNG có fallback: lỗi API (sai key,
hết quota, mất mạng) được ném ra để người chạy thấy — không thay bằng kết quả giả.
"""
import hashlib
import json
import time
from dataclasses import dataclass, field

from openai import OpenAI

from . import config


@dataclass
class Usage:
    """Cộng dồn token theo model để tính chi phí."""
    by_model: dict = field(default_factory=dict)   # model -> {"in": n, "out": n, "calls": n}

    def add(self, model: str, tokens_in: int, tokens_out: int = 0) -> None:
        row = self.by_model.setdefault(model, {"in": 0, "out": 0, "calls": 0})
        row["in"] += tokens_in
        row["out"] += tokens_out
        row["calls"] += 1

    def cost_usd(self) -> float | None:
        total = 0.0
        for model, row in self.by_model.items():
            price = config.price_per_1m(model)
            if price is None:
                return None          # không biết giá -> báo "chưa rõ", không đoán
            total += row["in"] / 1e6 * price[0] + row["out"] / 1e6 * price[1]
        return round(total, 6)

    def snapshot(self) -> dict:
        return {"by_model": json.loads(json.dumps(self.by_model)), "cost_usd": self.cost_usd()}

    def diff(self, before: dict) -> dict:
        """Token/chi phí phát sinh kể từ snapshot `before` (dùng cho từng câu hỏi)."""
        out = Usage()
        for model, row in self.by_model.items():
            old = before["by_model"].get(model, {"in": 0, "out": 0, "calls": 0})
            out.by_model[model] = {k: row[k] - old[k] for k in ("in", "out", "calls")}
        return out.snapshot()


class LLM:
    def __init__(self, agent_model: str | None = None) -> None:
        self.client = OpenAI()        # đọc OPENAI_API_KEY từ môi trường (.env)
        self.agent_model = agent_model or config.AGENT_MODEL
        self.embedding_model = config.EMBEDDING_MODEL
        self.usage = Usage()
        self._cache_path = config.CACHE_DIR / f"query_embeddings_{self.embedding_model}.json"
        self._cache = (json.loads(self._cache_path.read_text(encoding="utf-8"))
                       if self._cache_path.exists() else {})

    # ---- Embedding câu truy vấn (cùng model với index Day 6) --------------
    def embed_query(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in self._cache:
            resp = self.client.embeddings.create(model=self.embedding_model, input=[text])
            self.usage.add(self.embedding_model, resp.usage.total_tokens)
            self._cache[key] = resp.data[0].embedding
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        return self._cache[key]

    # ---- Chat có tool calling --------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str | None = None, json_mode: bool = False):
        kwargs = {"model": self.agent_model, "messages": messages}
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice or "auto",
                          parallel_tool_calls=False)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_error = None
        for attempt in range(3):      # thử lại khi lỗi mạng/tạm thời, rồi mới báo lỗi
            try:
                resp = self.client.chat.completions.create(**kwargs)
                u = resp.usage
                self.usage.add(self.agent_model, u.prompt_tokens, u.completion_tokens)
                return resp.choices[0].message
            except Exception as exc:  # noqa: BLE001 - ghi lại rồi ném tiếp
                last_error = exc
                if "rate" not in str(exc).lower() and "timeout" not in str(exc).lower():
                    break
                time.sleep(2 * (attempt + 1))
        raise last_error
