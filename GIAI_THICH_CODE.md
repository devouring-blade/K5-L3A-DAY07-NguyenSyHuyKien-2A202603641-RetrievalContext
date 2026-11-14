# Giải thích code Day 7

Đọc theo thứ tự các bước chạy. Mỗi file trong `src/rag7/` có chú thích ở đầu file.

## 0. Dữ liệu đi vào từ Day 6

- `src/rag7/day6/` là **mã Day 6 chép nguyên** (đọc PDF, chunk, embedding, Chroma).
- `build_index.py` gọi lại mã đó để tạo **cùng 242 chunk** như chiến lược B ở Day 6, embedding rồi lưu vào Chroma. Chạy lại lần hai thì không embed lại chunk nào (so hash nội dung).
- Đồng thời xuất `data/day7/chunks.json`: mỗi chunk có `id`, `text`, `source_id`, `source`, `page`, `metadata` (chương, mục). BM25 và phần chấm điểm đọc file này.

## 1. Benchmark

- `benchmark/questions.json`: câu có đáp án. Trường `gold` là danh sách `{source_id, page, evidence}`.
- `benchmark/cases.json`: mọi câu, có cờ `answerable`. Câu thiếu evidence ghi lý do và cách đã kiểm tra.
- `evidence.py`: định nghĩa **hit** = chunk đúng nguồn, đúng trang, và chứa trọn câu evidence (bỏ qua khác biệt khoảng trắng, hoa thường).
- Gold **chỉ dùng để chấm**, không bao giờ được gửi cho agent hay retriever. Có test kiểm tra điều này.

## 2. Bốn phương pháp retrieval (`retrievers.py`)

Tất cả nhận `(query, filters, top_k)` và trả về danh sách chunk có `score` và `rank`.

- **dense**: biến câu hỏi thành vector (`llm.embed_query`), hỏi Chroma lấy các vector gần nhất theo cosine. `score = 1 - khoảng cách`.
- **bm25**: tính điểm theo từ khóa. Từ hiếm (IDF cao) xuất hiện trong chunk thì điểm cao; chunk dài bị trừ bớt. Bỏ từ dừng tiếng Anh.
- **hybrid**: chạy cả dense và bm25 (mỗi bên 20 kết quả), cộng điểm `1/(60 + thứ hạng)`. Chỉ dùng thứ hạng nên không cần quy đổi hai loại score về cùng thang.
- **hybrid_rerank**: lấy 10 ứng viên từ hybrid, gửi cho LLM để sắp xếp lại theo mức hữu ích, lấy 3 cái đầu.
- **filter** (ví dụ `{"chapter": 8}`): do người dùng cung cấp trong benchmark, áp dụng trước khi xếp hạng.

`eval_retrieval.py` chạy từng phương pháp với câu hỏi gốc, top-3, rồi tính **Hit@1** và **Evidence Hit@3**.

## 3. Agent (`agent.py`)

Model nhận câu hỏi và 3 tool:

1. `search_documents(query)`: model tự viết query (có thể viết lại câu hỏi). Code gọi retriever của cấu hình, rồi thêm chunk mới vào **context**:
   - chunk trùng ID bị bỏ;
   - nếu vượt **budget** (số ký tự) thì chunk bị loại và model được báo lại.
2. `submit_answer(answer, citations)`: hợp lệ khi mọi citation là ID chunk **đã có trong context** và xuất hiện dạng `[id]` trong câu trả lời. Sai thì cho sửa 1 lần, sai nữa thì coi như từ chối.
3. `abstain(reason)`: từ chối khi không đủ bằng chứng.

Điều kiện dừng: trả lời, từ chối, hết số lượt search (model bị buộc phải chốt), hoặc quá số lượt hội thoại. Lỗi API được ghi `status: "error"`, không thay bằng kết quả giả.

## 4. Thử nghiệm (`experiments.py`, `run_agent.py`, `select_config.py`)

| Cấu hình | Khác baseline ở chỗ nào |
|---|---|
| `ag_base` | dense, top-3, 3 lượt search, budget 800 ký tự (mặc định của starter) |
| `ag_budget` | chỉ tăng budget lên 4000 |
| `ag_hybrid` | như `ag_budget`, tool tìm bằng hybrid |
| `ag_rerank` | như `ag_hybrid`, thêm LLM rerank |

`run_agent.py` tự đo: tool use, số lượt search, **Context Evidence Rate** (context cuối có chunk hit hay không), số câu trả lời hay từ chối, thời gian, token, chi phí.
`select_config.py` áp tiêu chí đã chốt **chỉ trên dev**. Sau đó mới chạy test.

## 5. Chấm tay (`review.py`)

`make` tạo bảng chấm cho từng cấu hình. Người chấm đọc câu trả lời và chunk được trích dẫn rồi điền: `answer_grade`, `citation_support`, `abstention_correct`, `context_grade`, `reason`. `summarize` kiểm tra đã điền đủ chưa, rồi dùng hàm `aggregate()` của `starter/agent_metrics.py` để tổng hợp.
