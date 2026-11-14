# Ghi NHÃN GỢI Ý (AI soạn) vào bảng chấm test + tạo PHIEU_CHAM.md để người chấm đọc.
# reviewer để trống: summarize sẽ từ chối cho đến khi người chấm xác nhận.
import json, sys
from pathlib import Path
R = Path(sys.argv[1]) / "results" / "reviews" / "test"
L = {
 "ag_budget_cite": {
  "t01": (2, [True], None, 2, "Nêu đúng 'chunk number và total number of chunks'; chunk p100:1 chứa nguyên văn câu này."),
  "t02": (2, [True], None, 2, "Beautiful Soup đúng; chunk p80:0 ghi rõ 'external library ... scraping and parsing HTML and XML'."),
  "t03": (2, [True], None, 2, "Đủ 1NF, 2NF, 3NF, BCNF; câu thêm về normalization cũng có trong chunk p90:2."),
  "t04": (0, [], None, 0, "Từ chối câu có đáp án -> 0. Agent tự viết lại query ('copyright duration...') và lấy p394:1 thay vì p394:0 chứa đáp án; context không có evidence."),
  "t05": (2, [True], None, 2, "Kaggle đúng; chunk p397:1 chứa 'platforms like Kaggle ... open data sharing in a research context' kèm điều kiện 'no privacy-related implications'."),
  "t06": (2, [True, True], None, 2, "8 giờ và 4 giờ đúng; p71:0 có câu gold, p71:1 nhắc lại 'eight hours ... compared to those with only four hours' nên cả 2 citation đều hỗ trợ."),
  "u03": (None, [], True, None, "Từ chối, không đưa ra con số nào; corpus không có công thức/số mẫu cho 95% và 5%."),
  "u04": (None, [], True, None, "Từ chối, không bịa tên ngôn ngữ; corpus không nhắc smart contract."),
 },
 "ag_base": {q: (0, [], None, 0, "Từ chối câu có đáp án -> 0. Budget 800 ký tự nhỏ hơn độ dài chunk (~900) nên chunk đúng bị loại khỏi context; context " + c + ".")
             for q, c in [("t01", "chỉ có p99:3, p103:1, không chứa evidence"), ("t02", "rỗng"), ("t03", "rỗng"),
                          ("t04", "chỉ có p388:4, không chứa evidence"), ("t05", "rỗng"), ("t06", "rỗng")]},
}
L["ag_base"]["u03"] = (None, [], True, None, "Từ chối, không đưa khẳng định nào.")
L["ag_base"]["u04"] = (None, [], True, None, "Từ chối, không đưa khẳng định nào (dù agent có thử query 'Solidity', không có trong corpus).")
md = ["# Phiếu chấm thủ công — test", "",
      "Nhãn bên dưới do **AI soạn gợi ý** từ trace. Người chấm đọc từng câu: câu trả lời, đoạn được trích dẫn, đáp án chuẩn.",
      "Không đồng ý thì sửa giá trị tương ứng trong `results/reviews/test/<cấu hình>.json`, rồi chạy `review confirm` và `review summarize`.", "",
      "Thang: answer_grade 0 sai/thiếu · 1 đúng một phần · 2 đúng và đủ (từ chối câu có đáp án = 0). context_grade 0 không có evidence · 1 thiếu điều kiện/nguồn · 2 đủ.", ""]
for cfg, labels in L.items():
    path = R / f"{cfg}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    md += [f"## {cfg}", ""]
    for r in rows:
        ag, cs, ab, cg, why = labels[r["query_id"]]
        assert len(cs) == len(r["citations"]), (cfg, r["query_id"])
        r.update(answer_grade=ag, citation_support=cs, abstention_correct=ab, context_grade=cg,
                 reason=why, label_source="AI gợi ý; người chấm đọc và xác nhận/sửa")
        md += [f"### {r['query_id']} — {r['question']}", "",
               f"- **Đáp án chuẩn:** {r['expected_answer'] or '(câu thiếu evidence) ' + (r['absence_reason'] or '')}",
               f"- **Agent ({r['status']}):** {r['answer']}"]
        for c in r["citations"]:
            md.append(f"- **Trích dẫn `{c['id']}` (trang {c['page']}):** {c['text'][:900]}")
        md += [f"- **Context cuối:** {', '.join(c['id'] for c in r['context']) or '(rỗng)'}",
               f"- **Nhãn gợi ý:** answer_grade={ag}, citation_support={cs}, abstention_correct={ab}, context_grade={cg}",
               f"- **Lý do:** {why}", ""]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
(R / "PHIEU_CHAM.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("ok")
