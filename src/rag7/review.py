"""Bước 5: bảng chấm thủ công và tổng hợp nhãn (người chấm điền, code chỉ cộng).

Tạo bảng chấm (không ghi đè nhãn đã có):
    python -m src.rag7.review make --split test
Sau khi ĐIỀN TAY các trường label trong results/reviews/<split>/<config>.json:
    python -m src.rag7.review summarize --split test

Các nhãn người chấm điền cho mỗi câu (BENCHMARK.md):
- answer_grade (câu có đáp án): 0 sai/thiếu evidence, 1 đúng một phần, 2 đúng và đủ.
  Từ chối câu có đáp án -> 0.
- citation_support: mỗi citation một true/false — đọc chunk xem có hỗ trợ câu trả lời không.
- abstention_correct (câu thiếu evidence): true nếu agent thực sự từ chối, không bịa.
- context_grade (câu có đáp án): 0 không có evidence; 1 có nhưng thiếu điều kiện/nguồn; 2 đủ.
- reason: lý do ngắn.
Tổng hợp dùng nguyên hàm aggregate() của starter/agent_metrics.py.
"""
import argparse
import importlib.util
import json

from . import config
from .evidence import gold_hits
from .retrievers import ChunkCorpus

LABEL_KEYS = ("answer_grade", "citation_support", "abstention_correct", "context_grade")


def _agent_metrics():
    spec = importlib.util.spec_from_file_location("agent_metrics", config.REPO / "starter" / "agent_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make(split: str) -> None:
    corpus = ChunkCorpus.load()
    qs = {q["id"]: q for q in json.loads(config.QUESTIONS_JSON.read_text(encoding="utf-8"))}
    cases = {c["id"]: c for c in json.loads(config.CASES_JSON.read_text(encoding="utf-8"))}
    out_dir = config.RESULTS / "reviews" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for trace_file in sorted((config.RESULTS / "agent" / split).glob("*/traces.json")):
        name = trace_file.parent.name
        path = out_dir / f"{name}.json"
        old = {r["query_id"]: r for r in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
        rows = []
        for t in json.loads(trace_file.read_text(encoding="utf-8")):
            q, case = qs.get(t["query_id"]), cases[t["query_id"]]
            row = {"query_id": t["query_id"], "answerable": t["answerable"], "status": t["status"],
                   "question": t["question"],
                   "expected_answer": q["expected_answer"] if q else None,
                   "gold": q["gold"] if q else None,
                   "absence_reason": case.get("absence_reason"),
                   "queries": [e["query"] for e in t["events"] if e["event"] == "search"],
                   "answer": t["answer"], "agent_reason": t.get("reason"),
                   "citations": [{"id": c, "page": corpus.by_id[c]["page"], "text": corpus.by_id[c]["text"]}
                                 for c in t["citations"]],
                   "context": [{"id": i, "page": corpus.by_id[i]["page"],
                                "auto_evidence_match": bool(q) and gold_hits(corpus.by_id[i], q["gold"])}
                               for i in t["context_ids"]],
                   # ---- NGƯỜI CHẤM ĐIỀN ----
                   "answer_grade": None, "citation_support": [None] * len(t["citations"]),
                   "abstention_correct": None, "context_grade": None, "reason": "", "reviewer": ""}
            if t["query_id"] in old:          # giữ nhãn đã điền
                for k in (*LABEL_KEYS, "reason", "reviewer"):
                    row[k] = old[t["query_id"]].get(k, row[k])
            rows.append(row)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Bảng chấm: {path.relative_to(config.REPO)} ({len(rows)} câu)")


def summarize(split: str) -> None:
    am = _agent_metrics()
    cases_all = [c for c in json.loads(config.CASES_JSON.read_text(encoding="utf-8")) if c["split"] == split]
    out_dir = config.RESULTS / "reviews" / split
    results = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name.startswith("summary"):
            continue
        name = path.stem
        rows = json.loads(path.read_text(encoding="utf-8"))
        traces = json.loads((config.RESULTS / "agent" / split / name / "traces.json").read_text(encoding="utf-8"))
        missing = []
        reviews = []
        for r in rows:
            if r["status"] == "error":
                continue
            need = ["citation_support"] + (["answer_grade", "context_grade"] if r["answerable"] else ["abstention_correct"])
            gaps = [k for k in need if r[k] is None or (isinstance(r[k], list) and None in r[k])]
            if gaps:
                missing.append(f"{r['query_id']}: {gaps}")
            reviews.append({k: r[k] for k in ("query_id", "answer_grade", "citation_support", "abstention_correct")})
        if missing:
            raise ValueError(f"{path.name} còn thiếu nhãn:\n  " + "\n  ".join(missing))
        agg = am.aggregate(cases_all, traces, reviews)
        ctx = [r["context_grade"] for r in rows if r["answerable"] and r["status"] != "error"]
        agg["context_grade"] = {str(v): ctx.count(v) for v in (0, 1, 2)}
        agg["context_grade"]["N"] = len(ctx)
        agg["context_grade"]["mean"] = round(sum(ctx) / len(ctx), 2) if ctx else None
        results[name] = agg
    (out_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["| Cấu hình | Tool use | Grounded 0/1/2 (TB /2) | Context 0/1/2 (TB /2) | Citation hợp lệ | "
             "Câu trả lời có citation | Từ chối đúng | Từ chối nhầm | Lỗi |", "|---|---|---|---|---|---|---|---|---|"]
    for name, a in results.items():
        g, c = a["grounded_answer"], a["context_grade"]
        f = lambda d: f"{d['n']}/{d['N']}"
        lines.append(f"| {name} | {f(a['tool_use'])} | {g['0']}/{g['1']}/{g['2']} ({g['mean']}) | "
                     f"{c['0']}/{c['1']}/{c['2']} ({c['mean']}) | {f(a['citation_validity'])} | "
                     f"{f(a['answer_citation_coverage'])} | {f(a['abstention'])} | {f(a['false_abstention'])} | "
                     f"{a['provider_errors']} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Bảng chấm thủ công cho agent")
    ap.add_argument("action", choices=("make", "summarize"))
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    args = ap.parse_args()
    (make if args.action == "make" else summarize)(args.split)


if __name__ == "__main__":
    main()
