# Báo cáo Day 7 — Nguyễn Sỹ Huy Kiên (2A202603641)

> **Về việc dùng AI.** Code, benchmark mới, phân tích và bản nháp báo cáo do trợ lý AI (Claude) viết. Tôi chạy toàn bộ lệnh trên máy mình với API thật; mọi con số dưới đây lấy từ file kết quả trong repo. Nhãn chấm thủ công trên test do AI **soạn gợi ý** từ trace ([PHIEU_CHAM.md](../../results/reviews/test/PHIEU_CHAM.md), script [ai_draft_labels.py](../../results/reviews/test/ai_draft_labels.py)); tôi đọc từng câu và xác nhận bằng `review confirm` (trường `reviewer` trong file nhãn). Cách chạy: [BAI_LAM.md](../../BAI_LAM.md); giải thích code: [GIAI_THICH_CODE.md](../../GIAI_THICH_CODE.md).

## Dữ liệu và benchmark

- **Nguồn, quyền, số trang, số chunk:** *Principles of Data Science* (OpenStax ©2025, **CC BY-NC-SA 4.0**), chương 2 (PDF tr. 67–103) và chương 8 (tr. 387–413) = **64 trang nội dung**, giống Day 6 ([manifest](../../data/day6/manifest.json)). Chunk theo chiến lược B/C đã chọn ở Day 6 (cắt tại heading mục + ranh giới câu, ≤ 1000 ký tự) → **242 chunk**, trung vị ~900 ký tự ([chunks.json](../../data/day7/chunks.json)). Mã Day 6 dùng lại nguyên văn trong `src/rag7/day6/`.
- **Vector store, embedding:** Chroma 1.5.9 `PersistentClient` (`chroma_db/`, collection `day06_B`, cosine, HNSW xấp xỉ); OpenAI `text-embedding-3-small`, **1536 chiều**, cùng model cho query và document. Lần dựng đầu embed 242 chunk (~36 nghìn token theo log Day 6, < 0,001 USD); chạy lại `build_index` → *Embed mới 0, bỏ qua 242, store 242* ([index_info.json](../../data/day7/index_info.json)): nạp lại không nhân đôi. pypdf 7.0.1, openai 3.50.0, Python 3.13.14, Windows 11.
- **Agent model, cách chạy, dev/test:** `gpt-4.1-mini` qua Chat Completions với tool calling. Dev = **20 câu Day 6** (kết quả test Day 6 đã từng xem, nên Day 7 chỉ dùng làm dev) **+ 2 câu thiếu evidence**. Test = **6 câu có đáp án + 2 câu thiếu evidence viết mới**, cố định trước mọi lần chạy ([questions.json](../../benchmark/questions.json), [cases.json](../../benchmark/cases.json)). 5 câu có filter `chapter` do người dùng cung cấp (3 dev, 2 test).
- **Câu có/thiếu evidence, tiêu chí chấm:** câu có đáp án có `gold = {source_id, page, evidence}`; `build_index` kiểm tra 26/26 evidence nằm trọn trong một chunk đúng nguồn/trang. **Hit** = chunk đúng nguồn + trang và chứa trọn evidence (bỏ qua khác biệt khoảng trắng/hoa thường). Câu thiếu evidence là câu gần chủ đề; cách kiểm tra (tìm từ khóa trong cả 64 trang) ghi ở trường `how_checked`, ví dụ u02 "Hadoop/HDFS": 0 kết quả.

## Retrieval và thử nghiệm

Retrieval đo bằng **câu hỏi gốc**, top-3, cùng index ([results/retrieval/](../../results/retrieval/)). Agent đo **Context Evidence Rate** = câu có đáp án mà context cuối gửi model chứa chunk hit / số câu có đáp án ([results/agent/](../../results/agent/)). Budget đo bằng **ký tự** (gồm dòng nguồn). Thời gian là tổng cho cả split, chi phí theo bảng giá OpenAI (gpt-4.1-mini 0,40/1,60 USD/1M token vào/ra; embedding 0,02).

**So sánh phương pháp retrieval (câu hỏi gốc):**

| Phương pháp | Split | Hit@1 | Evidence Hit@3 | Median / chi phí |
|---|---|---|---|---|
| dense (Chroma Day 6) | dev | 14/20 (70%) | 19/20 (95%) | 189 ms / ~0 |
| bm25 (tự cài) | dev | 16/20 (80%) | 19/20 (95%) | 0,3 ms / 0 |
| hybrid (RRF k=60, mỗi bên 20) | dev | 16/20 (80%) | 19/20 (95%) | 1,8 ms / 0 |
| hybrid + LLM rerank (10 ứng viên) | dev | 16/20 (80%) | 19/20 (95%) | 1472 ms / 0,015 USD |
| dense | test | 4/6 (67%) | 6/6 (100%) | 570 ms |
| bm25 · hybrid | test | 6/6 (100%) | 6/6 (100%) | 2–7 ms |
| hybrid + rerank | test | 4/6 (67%) | 6/6 (100%) | 1495 ms / 0,004 USD |

**Agent (cùng model và prompt; chỉ đổi cột ghi trong "thay đổi"):**

| Cấu hình | Phương pháp, top-k, filter, rerank, budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline `ag_base` | dense, top-3, ≤3 lượt, filter người dùng, budget 800 | Mặc định của starter | dev | 14/20 | 19/20 | **0/20 (0%)** | 76 s / 0,014 USD |
| Phương pháp 2 `ag_hybrid` | hybrid RRF, top-3, budget 4000 | Thuật ngữ riêng được BM25 bắt tốt hơn | dev | 16/20 | 19/20 | 19/20 (95%) | 80 s / 0,026 USD |
| Phương pháp 3 `ag_rerank` | hybrid + LLM rerank, budget 4000 | Rerank tăng Hit@1, đổi lại chậm hơn | dev | 16/20 | 19/20 | 18/20 (90%) | 121 s / 0,047 USD |
| Cải thiện 1 `ag_budget` | dense, budget **800 → 4000** | Chunk ~900 ký tự > 800 nên bị loại khỏi context | dev | 14/20 | 19/20 | **20/20 (100%)** | 84 s / 0,025 USD |
| Cải thiện 2 `ag_budget_cite` | như cải thiện 1 + sửa định dạng citation | Giảm từ chối nhầm do lỗi citation | dev | 14/20 | 19/20 | 20/20 (100%) | 78 s / 0,024 USD |
| (thêm) `ag_hybrid_cite` | như `ag_hybrid` + sửa citation | Hybrid có lợi hơn khi đã sửa citation? | dev | 16/20 | 19/20 | 19/20 (95%) | 75 s / 0,025 USD |
| **Cấu hình đã chọn** `ag_budget_cite` | dense, top-3, ≤3 lượt, budget 4000, sửa citation | — | test | 4/6 | 6/6 | **5/6 (83%)** | 27 s / 0,0088 USD |
| Baseline `ag_base` | như trên dev | — | test | 4/6 | 6/6 | 0/6 (0%) | 32 s / 0,0051 USD |

Hit@1/Hit@3 của cấu hình agent là của retriever tương ứng với câu hỏi gốc. Agent tự viết lại query nên context có thể khác top-3 này; Context Evidence Rate đo trên context thật.

`ag_hybrid` / `ag_rerank` vừa là phương pháp mới vừa là thử cải thiện. `ag_budget_cite` và `ag_hybrid_cite` được **thêm sau khi đọc lỗi dev** của 4 cấu hình đầu; vẫn chỉ dùng dev. Tiêu chí chọn chốt trước ([experiments.py](../../src/rag7/experiments.py)): dev Context Evidence Rate cao nhất → ít từ chối nhầm → rẻ/nhanh hơn. Kết quả ([selection.json](../../results/agent/selection.json); vòng đầu: [selection_round1.json](../../results/agent/selection_round1.json)): `ag_budget_cite` (20/20, từ chối nhầm 0/20) > `ag_budget` (20/20, 6/20) > `ag_hybrid_cite` (19/20, 0/20). Test chạy **một lần** sau khi chốt, không chỉnh gì sau đó.

**Hai lỗi dev, nguyên nhân, cách thử:**

1. **Budget 800 làm context rỗng.** `ag_base` từ chối 20/20 câu dev dù dense có Hit@3 19/20. Trace (`results/agent/dev/ag_base/traces.json`, ví dụ q01) cho thấy top-3 có `B:pds-ch02:p73:0` chứa đáp án nhưng nằm trong `dropped_over_budget`. Chỉ chunk ngắn `p76:1` lọt vào context; context trung bình chỉ 240 ký tự. Nguyên nhân: chunk Day 6 trung vị ~900 ký tự, cộng dòng nguồn đã vượt 800. Model còn tìm thêm vô ích (2,3 lượt/câu). Tăng budget lên 4000 → Context Evidence 0/20 → 20/20, trung bình 1,2 lượt/câu.
2. **Lỗi định dạng citation gây từ chối nhầm.** `ag_budget` từ chối 6/20 câu dev (q02, q06, q08, q15, q16, q18) dù context **đã có** evidence. Trace: lần đầu model trả lời đúng (ví dụ q16 "HIPAA") nhưng không ghi `[chunk-id]` trong câu; khi được báo sửa lại thì viết `[pds-ch02:p73:1]`, mất tiền tố `B:`. Hai lần sai nên bị coi là từ chối. Sửa (`cite_fix`): prompt nêu định dạng ID kèm ví dụ, thông báo sửa liệt kê ID hợp lệ, code khôi phục tiền tố `B:` nếu khớp duy nhất. Kết quả: từ chối nhầm 6/20 → 0/20; hybrid 7/20 → 0/20. Trace cho thấy phần khôi phục tiền tố **không được dùng lần nào**: 10 câu trả lời sai định dạng ở lần đầu đều được model tự sửa nhờ thông báo liệt kê ID. Cải thiện đến từ prompt và thông báo, không phải từ code sửa hộ.

**Vì sao dense thay vì hybrid dù hybrid có Hit@1 cao hơn:** tiêu chí là chất lượng context cuối, không phải Hit@1. Trên dev, hybrid để lọt 1 câu (q17 "What is data validation?", filter chương 8): chunk gold `p403:1` không vào top-3, thay bằng `p403:2`/`p404:0` (nói về outlier detection, cross-validation). Rerank làm mất thêm q15. Hit@1 không ảnh hưởng nhiều vì agent đọc cả 3 chunk trong context.

## Agent và đánh giá

- **Tool, lượt gọi, dừng, context, lỗi.** Ba tool: `search_documents(query)` (model tự viết query; filter `chapter` do người dùng cung cấp được hệ thống áp, model không đổi được), `submit_answer(answer, citations)`, `abstain(reason)`. `tool_choice="required"`, tối đa 3 search; lượt thứ 4 bị từ chối và model phải chốt; tối đa 6 lượt hội thoại. Context tích lũy qua các lượt, dedup theo ID, dừng thêm khi hết budget (báo lại cho model số chunk bị loại). Câu trả lời chỉ hợp lệ khi mọi citation có trong context và xuất hiện dạng `[id]`; sai được sửa 1 lần, sai nữa → từ chối. Lỗi API → `status: "error"` (0 lỗi trong mọi lần chạy); mỗi lời gọi có timeout 60 s và thử lại 2 lần (thêm sau khi một lần chạy dev bị treo).

**Kết quả agent trên test (nhãn thủ công: [results/reviews/test/summary.md](../../results/reviews/test/summary.md)):**

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| `ag_base` / test | 8/8 | 1×1, 4×2, 3×3 (TB 2,25) | 6/0/0 · 0,00 | 0/0 (không áp dụng) | 0/0 | 2/2 | 6/6 |
| `ag_budget_cite` / test | 8/8 | 6×1, 2×3 (TB 1,5) | 1/0/5 · 1,67 | 6/6 | 5/5 | 2/2 | 1/6 |

Context grade (5+ câu test có đáp án): `ag_budget_cite` 0/1/2 = 1/0/5 (TB 1,67); `ag_base` 6/0/0 (TB 0).

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t02 | diễn đạt lại | 1 search "Python library for scraping and parsing HTML and XML" → p78:1, p77:0, **p80:0** | "…is Beautiful Soup [B:pds-ch02:p80:0]" | Có bằng chứng: grade 2, citation đúng. p80:0 ghi nguyên văn "external library … scraping and parsing HTML and XML". |
| t05 | filter `chapter=8` | 1 search, filter tự áp → p397:0, p396:3, **p397:1** | "…is Kaggle [B:pds-ch08:p397:1]" | Grade 2; chunk giữ cả điều kiện "no privacy-related implications". |
| t04 | trực tiếp | 3 search tự viết lại ("copyright duration United States"…) → p388:4, p388:3, p389:1, p394:1 | Từ chối | **Lỗi:** grade 0. Câu hỏi gốc đưa `p394:0` lên hạng 1 (dense test Hit@1), nhưng query agent viết lại thì không. Query rewriting làm hại. |
| u04 | thiếu evidence | 1 search → p393:3, p80:1, p77:1 | Từ chối: "do not contain information about … smart contracts" | Đúng: không bịa ngôn ngữ nào. Corpus chỉ nhắc blockchain khi nói về mã hóa. |
| u03 | thiếu evidence | 3 search về sample size 95%/5% | Từ chối | Đúng: corpus không có công thức hay con số này. |

- **Đường dẫn, lệnh, thời gian, chi phí:** trace agent `results/agent/<split>/<config>/traces.json` (mỗi câu: query từng lượt, kết quả và score, chunk được thêm/bị loại, context cuối, câu trả lời, citation, token, thời gian). Trace retrieval `results/retrieval/<split>/<method>.json`; nhãn `results/reviews/test/*.json`. Lệnh ở [BAI_LAM.md](../../BAI_LAM.md). Tổng chi phí mọi lần chạy ≈ **0,22 USD**, cộng từ các file summary (agent dev 6 cấu hình kể cả lần chạy lặp ≈ 0,18; rerank retrieval 0,02; agent test 0,014; embedding < 0,001).
- **Trước/sau, đánh đổi, giới hạn:**
  - Trên test, cấu hình chọn tăng grounded answer từ 0,00 lên 1,67/2 và từ chối nhầm từ 6/6 xuống 1/6, chi phí 0,0051 → 0,0088 USD cho 8 câu (context dài hơn ~9 lần nhưng ít lượt search hơn).
  - Rerank tốn gấp đôi, chậm hơn ~50% mà không tăng Context Evidence trên dev.
  - BM25/hybrid có Hit@1 cao hơn dense trên cả dev lẫn test. Nếu tối ưu cho top-1 (ví dụ budget nhỏ chỉ đủ 1 chunk), hybrid có thể là lựa chọn tốt hơn; điều này chưa được kiểm chứng ở đây.
  - **Giới hạn:** test chỉ 6+2 câu nên mỗi câu = 17 điểm phần trăm; không suy rộng. Dev là câu Day 6 do cùng tác giả viết, câu hỏi dùng thuật ngữ gần với sách. Model không tất định: `ag_budget_cite` dev chạy 2 lần (do lần chạy lặp), hai lần đều 20/20 và 0/20 từ chối nhầm, chi phí 0,0224 và 0,0240 USD (file hiện tại là lần 2). Chỉ một người chấm, nhãn do AI soạn nên có nguy cơ thiên lệch. Lỗi t04 cho thấy nên thử giữ câu hỏi gốc làm lượt search đầu. Ý tưởng này phát hiện trên test nên **không** áp dụng lại; cần tập test mới để đánh giá.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
