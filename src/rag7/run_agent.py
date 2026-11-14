"""Bước 3: chạy agent thật trên một split cho các cấu hình trong experiments.py.

Chạy:  python -m src.rag7.run_agent --split dev                       (mọi cấu hình)
       python -m src.rag7.run_agent --split test --configs ag_base,selected

`selected` = cấu hình đã chọn trên dev (đọc results/agent/selection.json do
select_config tạo). Kết quả: results/agent/<split>/<config>/traces.json + summary.json,
và results/agent/<split>/summary.md (bảng so sánh).

Chỉ số tự động (không cần chấm tay):
- Tool use: số câu có gọi search / số câu chạy thành công; phân bố số lượt search.
- Context Evidence Rate: số câu có đáp án mà context cuối chứa chunk hit / số câu có đáp án.
- Số câu trả lời / từ chối theo loại câu, lỗi, thời gian, token, chi phí.
Chấm câu trả lời, citation, abstention là việc của người: xem review.py.
"""
import argparse
import json
import statistics
from collections import Counter

from . import config
from .agent import run_agent
from .evidence import gold_hits
from .experiments import AGENT_CONFIGS
from .llm import LLM
from .retrievers import ChunkCorpus, make_retriever


def load_cases(split: str) -> list[dict]:
    cases = json.loads(config.CASES_JSON.read_text(encoding="utf-8"))
    return [c for c in cases if c["split"] == split]


def gold_by_id() -> dict:
    return {q["id"]: q["gold"] for q in json.loads(config.QUESTIONS_JSON.read_text(encoding="utf-8"))}


def resolve(names: str) -> list[str]:
    out = []
    for n in names.split(","):
        if n == "selected":
            sel = json.loads((config.RESULTS / "agent" / "selection.json").read_text(encoding="utf-8"))
            n = sel["selected"]
        if n not in AGENT_CONFIGS:
            raise ValueError(f"Không có cấu hình {n}")
        if n not in out:
            out.append(n)
    return out


def summarize(name: str, traces: list[dict], cases: list[dict], llm_usage: dict) -> dict:
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["answerable"]]
    unans = [t for t in ok if not t["answerable"]]
    searches = Counter(t["n_searches"] for t in ok)
    cfg = AGENT_CONFIGS[name]
    return {
        "config": name, **{k: cfg[k] for k in ("retriever", "top_k", "max_searches", "budget_chars")},
        "cite_fix": cfg.get("cite_fix", False),
        "model": config.AGENT_MODEL, "cases": len(cases), "errors": len(traces) - len(ok),
        "error_ids": [t["query_id"] for t in traces if t["status"] == "error"],
        "tool_use": f"{sum(t['n_searches'] > 0 for t in ok)}/{len(ok)}",
        "searches_distribution": dict(sorted(searches.items())),
        "mean_searches": round(statistics.mean(t["n_searches"] for t in ok), 2) if ok else None,
        "context_evidence_rate": f"{sum(t['context_has_evidence'] for t in ans)}/{len(ans)}",
        "context_evidence_rate_value": round(sum(t["context_has_evidence"] for t in ans) / len(ans), 3) if ans else None,
        "context_missing_evidence": [t["query_id"] for t in ans if not t["context_has_evidence"]],
        "answerable_answered": f"{sum(t['status'] == 'answered' for t in ans)}/{len(ans)}",
        "false_abstention": f"{sum(t['status'] == 'abstain' for t in ans)}/{len(ans)}",
        "false_abstention_ids": [t["query_id"] for t in ans if t["status"] == "abstain"],
        "unanswerable_abstained": f"{sum(t['status'] == 'abstain' for t in unans)}/{len(unans)}",
        "answers_with_citation": f"{sum(bool(t['citations']) for t in ok if t['status'] == 'answered')}/"
                                 f"{sum(t['status'] == 'answered' for t in ok)}",
        "mean_context_chars": round(statistics.mean(t["context_chars"] for t in ok), 1) if ok else None,
        "median_seconds": round(statistics.median(t["elapsed_s"] for t in ok), 2) if ok else None,
        "total_seconds": round(sum(t["elapsed_s"] for t in traces), 1),
        "usage": llm_usage,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Chạy agent RAG trên một split")
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", default=",".join(AGENT_CONFIGS))
    args = ap.parse_args()
    cases, gold = load_cases(args.split), gold_by_id()
    corpus = ChunkCorpus.load()
    split_dir = config.RESULTS / "agent" / args.split
    for name in resolve(args.configs):
        cfg = AGENT_CONFIGS[name]
        llm = LLM()                       # usage riêng cho từng cấu hình
        retriever = make_retriever(cfg["retriever"], corpus, llm)
        traces = []
        print(f"== {name}: {cfg['changes']}")
        for case in cases:
            # Chỉ đưa câu hỏi + filter người dùng cung cấp; gold chỉ dùng SAU để chấm.
            out = run_agent(case["question"], llm, retriever, filters=case.get("filter"),
                            top_k=cfg["top_k"], max_searches=cfg["max_searches"],
                            budget_chars=cfg["budget_chars"],
                            cite_fix=cfg.get("cite_fix", False))
            g = gold.get(case["id"])
            has_ev = bool(g) and any(gold_hits(corpus.by_id[i], g) for i in out["context_ids"])
            traces.append({"query_id": case["id"], "split": case["split"], "config": name,
                           "question": case["question"], "answerable": case["answerable"],
                           "filter": case.get("filter"), **out, "context_has_evidence": has_ev})
            print(f"  {case['id']:4s} {out['status']:9s} searches={out['n_searches']} "
                  f"ctx_evidence={has_ev if g else '-'}")
        out_dir = split_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = summarize(name, traces, cases, llm.usage.snapshot())
        (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  -> Context Evidence {summary['context_evidence_rate']}, từ chối nhầm {summary['false_abstention']}, "
              f"từ chối đúng câu thiếu evidence {summary['unanswerable_abstained']}, lỗi {summary['errors']}, "
              f"chi phí {summary['usage']['cost_usd']} USD")
    write_table(split_dir)


def write_table(split_dir) -> None:
    rows = ["| Cấu hình | Retriever | Budget | Sửa citation | Tool use | Lượt search TB | Context Evidence | "
            "Có đáp án được trả lời | Từ chối nhầm | Thiếu evidence bị từ chối | Lỗi | Median s | Chi phí USD |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(split_dir.glob("*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        rows.append(f"| {s['config']} | {s['retriever']} | {s['budget_chars']} | {s.get('cite_fix', False)} | {s['tool_use']} | {s['mean_searches']} | "
                    f"{s['context_evidence_rate']} | {s['answerable_answered']} | {s['false_abstention']} | "
                    f"{s['unanswerable_abstained']} | {s['errors']} | {s['median_seconds']} | {s['usage']['cost_usd']} |")
    (split_dir / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
