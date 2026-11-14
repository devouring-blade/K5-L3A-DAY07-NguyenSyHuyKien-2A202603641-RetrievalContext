"""Định nghĩa "hit" dùng chung cho mọi phương pháp và mọi cấu hình.

Một chunk là hit khi: đúng source_id, đúng trang gold, VÀ chứa trọn evidence
(so khớp sau khi gộp khoảng trắng/xuống dòng và không phân biệt hoa thường —
giống Day 6, vì chunk nối các dòng PDF bằng dấu cách).
"""
import re


def ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def gold_hits(chunk: dict, gold: list[dict]) -> bool:
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"]
               and ws(g["evidence"]) in ws(chunk["text"]) for g in gold)
