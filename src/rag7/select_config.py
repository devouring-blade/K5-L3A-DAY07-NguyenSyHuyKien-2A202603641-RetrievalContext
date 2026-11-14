"""Bước 4: áp dụng tiêu chí chọn (đã chốt trong experiments.py) lên kết quả DEV.

Chạy:  python -m src.rag7.select_config
Ghi results/agent/selection.json. Không đọc bất kỳ kết quả test nào.
"""
import json

from . import config
from .experiments import AGENT_CONFIGS, BASELINE, SELECTION_RULE


def _frac(s: str) -> float:
    n, d = s.split("/")
    return int(n) / int(d) if int(d) else 0.0


def main() -> None:
    rows = []
    for name in AGENT_CONFIGS:
        f = config.RESULTS / "agent" / "dev" / name / "summary.json"
        if not f.exists():
            raise FileNotFoundError(f"Chưa có kết quả dev cho {name}: chạy run_agent --split dev trước")
        s = json.loads(f.read_text(encoding="utf-8"))
        rows.append((-_frac(s["context_evidence_rate"]), _frac(s["false_abstention"]),
                     (s["usage"]["cost_usd"] or 0) * 1000 + s["total_seconds"] / 60, name, s))
    rows.sort(key=lambda r: r[:3])
    ranking = [{"config": r[3], "context_evidence_rate": r[4]["context_evidence_rate"],
                "false_abstention": r[4]["false_abstention"], "cost_usd": r[4]["usage"]["cost_usd"],
                "total_seconds": r[4]["total_seconds"]} for r in rows]
    selected = rows[0][3]
    out = {"rule": SELECTION_RULE, "baseline": BASELINE, "selected": selected, "dev_ranking": ranking}
    path = config.RESULTS / "agent" / "selection.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
