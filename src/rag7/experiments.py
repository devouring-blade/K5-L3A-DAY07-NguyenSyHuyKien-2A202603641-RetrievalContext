"""Các cấu hình thử nghiệm và tiêu chí chọn — CHỐT TRƯỚC KHI CHẠY (kể cả dev).

Retrieval (đo trên câu hỏi gốc, top-3):  dense | bm25 | hybrid | hybrid_rerank

Agent (cùng model, cùng prompt; chỉ khác cột được ghi trong `changes`):
"""
AGENT_CONFIGS = {
    "ag_base": {
        "retriever": "dense", "top_k": 3, "max_searches": 3, "budget_chars": 800,
        "changes": "Baseline: dense trên vector store Day 6, tham số mặc định của starter (top_k=3, 3 lượt, budget 800 ký tự).",
        "hypothesis": "Mốc so sánh.",
    },
    "ag_budget": {
        "retriever": "dense", "top_k": 3, "max_searches": 3, "budget_chars": 4000,
        "changes": "Cải thiện 1: chỉ tăng context budget 800 -> 4000 ký tự.",
        "hypothesis": "Chunk Day 6 dài trung vị ~900 ký tự > 800, nên với budget 800 nhiều chunk đúng bị loại khỏi context; tăng budget sẽ tăng Context Evidence Rate.",
    },
    "ag_hybrid": {
        "retriever": "hybrid", "top_k": 3, "max_searches": 3, "budget_chars": 4000,
        "changes": "Cải thiện 2: giống ag_budget nhưng tool tìm bằng hybrid (dense + BM25, RRF).",
        "hypothesis": "Câu hỏi chứa thuật ngữ riêng (k-anonymization, Beautiful Soup...) được BM25 bắt tốt hơn; gộp RRF giúp đưa chunk đúng lên top-3 thường hơn dense.",
    },
    "ag_rerank": {
        "retriever": "hybrid_rerank", "top_k": 3, "max_searches": 3, "budget_chars": 4000,
        "changes": "Cải thiện 3: giống ag_hybrid nhưng xếp lại 10 ứng viên bằng LLM.",
        "hypothesis": "Rerank tăng Hit@1 (chunk đúng lên đầu) đổi lại thêm 1 lời gọi LLM mỗi lượt tìm (chậm hơn, tốn hơn).",
    },
}

BASELINE = "ag_base"

SELECTION_RULE = (
    "Chốt trước khi chạy: chọn cấu hình agent có Context Evidence Rate trên dev cao nhất; "
    "hòa thì ít câu có đáp án bị từ chối nhầm hơn (dev); hòa tiếp thì chi phí + thời gian thấp hơn. "
    "Sau khi chọn mới chạy test cho baseline và cấu hình đã chọn; không chỉnh gì sau khi xem test."
)
