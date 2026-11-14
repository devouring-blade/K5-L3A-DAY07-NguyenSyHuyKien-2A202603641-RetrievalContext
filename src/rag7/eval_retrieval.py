"""Bước 2: so sánh các phương pháp retrieval trên câu hỏi gốc (không qua agent).

Chạy:  python -m src.rag7.eval_retrieval --split dev
       python -m src.rag7.eval_retrieval --split dev --methods dense,bm25

Với mỗi câu có đáp án trong split: tìm top-3 bằng câu hỏi gốc + filter người dùng
cung cấp (nếu có), đánh dấu hit theo evidence.py. Câu thiếu evidence không vào
mẫu số. Ghi results/retrieval/<split>/<method>.json và summary.json/.md.
"""
import argparse
import json
import statistics
import time

from . import config
from .evidence import gold_hits
from .llm import LLM
from .retrievers import METHODS, ChunkCorpus, make_retriever

TOP_K = 3


def load_questions(split: str) -> list[dict]:
    qs = json.loads(config.QUESTIONS_JSON.read_text(encoding="utf-8"))
    return [q for q in qs if q["split"] == split]


def run_method(name: str, retriever, questions: list[dict], llm: LLM) -> tuple[list, dict]:
    traces, times = [], []
    usage_before = llm.usage.snapshot()
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], filters=q.get("filter"), top_k=TOP_K)
        times.append((time.perf_counter() - t0) * 1000)
        marked = [{"rank": h["rank"], "id": h["id"], "score": h["score"], "source_id": h["source_id"],
                   "page": h["page"], "metadata": h["metadata"], "hit": gold_hits(h, q["gold"]),
                   "text": h["text"], **({"base_rank": h["base_rank"]} if "base_rank" in h else {})}
                  for h in hits]
        flt = q.get("filter")
        traces.append({"query_id": q["id"], "split": q["split"], "type": q["type"],
                       "question": q["question"], "query_used": q["question"], "filter": flt,
                       "method": name, "top_k": TOP_K, "gold": q["gold"], "results": marked,
                       "hit1": bool(marked) and marked[0]["hit"],
                       "hit3": any(m["hit"] for m in marked),
                       "filter_respected": (all(all(m["metadata"].get(k) == v for k, v in flt.items())
                                                for m in marked) if flt else None),
                       "elapsed_ms": round(times[-1], 1)})
    n = len(traces)
    cost = llm.usage.diff(usage_before)
    summary = {"method": name, "N": n,
               "hit1": f"{sum(t['hit1'] for t in traces)}/{n}",
               "hit1_rate": round(sum(t["hit1"] for t in traces) / n, 3),
               "hit3": f"{sum(t['hit3'] for t in traces)}/{n}",
               "hit3_rate": round(sum(t["hit3"] for t in traces) / n, 3),
               "misses_top3": [t["query_id"] for t in traces if not t["hit3"]],
               "not_top1": [t["query_id"] for t in traces if not t["hit1"]],
               "filter_questions_respected": f"{sum(bool(t['filter_respected']) for t in traces if t['filter'])}/"
                                             f"{sum(1 for t in traces if t['filter'])}",
               "median_ms": round(statistics.median(times), 1),
               "usage": cost}
    return traces, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="So sánh phương pháp retrieval trên một split")
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--methods", default=",".join(METHODS))
    args = ap.parse_args()
    questions = load_questions(args.split)
    corpus, llm = ChunkCorpus.load(), LLM()
    out_dir = config.RESULTS / "retrieval" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in args.methods.split(","):
        traces, summary = run_method(name, make_retriever(name, corpus, llm), questions, llm)
        (out_dir / f"{name}.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n",
                                              encoding="utf-8")
        summaries.append(summary)
        print(f"{name:14s} Hit@1 {summary['hit1']:>6s}  Hit@3 {summary['hit3']:>6s}  "
              f"median {summary['median_ms']} ms  cost {summary['usage']['cost_usd']}")
    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    rows = ["| Phương pháp | Hit@1 | Evidence Hit@3 | Lọt top-3 | Median ms | Chi phí USD |",
            "|---|---|---|---|---|---|"]
    for s in summaries:
        rows.append(f"| {s['method']} | {s['hit1']} ({s['hit1_rate']:.0%}) | {s['hit3']} ({s['hit3_rate']:.0%}) | "
                    f"{', '.join(s['misses_top3']) or '-'} | {s['median_ms']} | {s['usage']['cost_usd']} |")
    (out_dir / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Đã ghi {out_dir.relative_to(config.REPO)}/")


if __name__ == "__main__":
    main()
