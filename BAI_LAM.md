# Day 7 — Agentic RAG · bài làm

Agent trả lời câu hỏi về 2 chương sách *Principles of Data Science* (OpenStax, CC BY-NC-SA 4.0), dùng lại **corpus, chunk, embedding và vector store của Day 6**. Báo cáo: [report/individual/2A202603641_NguyenSyHuyKien.md](report/individual/2A202603641_NguyenSyHuyKien.md). Giải thích code cho người đọc: [GIAI_THICH_CODE.md](GIAI_THICH_CODE.md).

## Thành phần

| Thành phần | Lựa chọn |
|---|---|
| Dữ liệu | Chương 2 + 8, 64 trang nội dung, manifest [data/day6/manifest.json](data/day6/manifest.json) (giống Day 6) |
| Chunk | Chiến lược B Day 6: cắt theo heading mục + câu, tối đa 1000 ký tự → 242 chunk, [data/day7/chunks.json](data/day7/chunks.json) |
| Embedding | OpenAI `text-embedding-3-small`, 1536 chiều, cùng model cho query và document |
| Vector store | Chroma `PersistentClient`, thư mục `chroma_db/`, collection `day06_B`, cosine (HNSW) — như Day 6 |
| Retrieval | `dense` (Chroma) · `bm25` (tự cài) · `hybrid` (RRF dense+BM25) · `hybrid_rerank` (LLM xếp lại 10 ứng viên) |
| Agent | Model OpenAI (`AGENT_MODEL` trong `.env`) với tool calling: `search_documents`, `submit_answer`, `abstain` |
| Benchmark | [benchmark/questions.json](benchmark/questions.json) (câu có đáp án + gold) và [benchmark/cases.json](benchmark/cases.json) (mọi câu): dev 20 + 2 thiếu evidence, test 6 + 2 thiếu evidence |
| Cấu hình & tiêu chí chọn | Chốt trước khi chạy trong [src/rag7/experiments.py](src/rag7/experiments.py) |

Benchmark dev = 20 câu Day 6 (đã từng xem kết quả ở Day 6 nên **chỉ dùng làm dev**). Test = 6 câu có đáp án + 2 câu thiếu evidence **viết mới cho Day 7, trước mọi lần chạy**. Evidence của cả 26 câu được kiểm tra tự động nằm trọn trong một chunk đúng nguồn/trang.

## Cách chạy (Windows PowerShell, từ thư mục gốc repo)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env        # rồi điền OPENAI_API_KEY (không commit .env)
# PDF (37 MB, không commit lại): tải từ https://openstax.org/details/books/principles-data-science
# hoặc lấy từ repo Day 6, đặt tại data/day6/principles-data-science.pdf

python -m src.rag7.check                              # 0. kiểm tra key, model, thư viện
python -m src.rag7.build_index                        # 1. dựng index Day 6 + xuất chunks.json
python -m src.rag7.eval_retrieval --split dev         # 2. so sánh 4 phương pháp retrieval
python -m src.rag7.run_agent --split dev              # 3. chạy agent, mọi cấu hình, trên dev
python -m src.rag7.select_config                      # 4. chọn cấu hình theo tiêu chí đã chốt
python -m src.rag7.eval_retrieval --split test        # 5. sau khi chốt: retrieval trên test
python -m src.rag7.run_agent --split test --configs ag_base,selected
python -m src.rag7.review make --split test           # 6. tạo bảng chấm thủ công
python -m src.rag7.review summarize --split test      #    sau khi người chấm điền nhãn
python -m unittest discover -s tests -v               # test offline (không gọi API)
```

## Kết quả nằm ở đâu

| Đường dẫn | Nội dung |
|---|---|
| `data/day7/index_info.json` | Số trang/chunk, model, vector store, token/chi phí embedding, kiểm tra nạp lại |
| `results/retrieval/<split>/` | Trace top-3 từng câu cho mỗi phương pháp, `summary.md` (Hit@1, Evidence Hit@3) |
| `results/agent/<split>/<config>/traces.json` | Mỗi câu: tool call, query, kết quả, context cuối, câu trả lời, citation, lỗi, thời gian, token |
| `results/agent/<split>/summary.md` | Tool use, Context Evidence Rate, từ chối, chi phí theo cấu hình |
| `results/agent/selection.json` | Áp tiêu chí chọn lên dev |
| `results/reviews/test/` | Nhãn chấm thủ công và bảng tổng hợp |
