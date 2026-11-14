"""Kiểm tra hành vi code Day 7 OFFLINE (không gọi API).

Fake LLM/embedding ở đây CHỈ để kiểm tra logic; không dùng để báo chất lượng.
Chạy:  python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag7.agent import ContextWindow, run_agent  # noqa: E402
from src.rag7.evidence import gold_hits  # noqa: E402
from src.rag7.llm import Usage  # noqa: E402
from src.rag7.retrievers import BM25Retriever, ChunkCorpus, HybridRetriever, RerankRetriever  # noqa: E402


def chunk(cid, text, page=1, chapter=2):
    return {"id": cid, "text": text, "source_id": f"doc{chapter}", "source": "x.pdf", "page": page,
            "metadata": {"chapter": chapter, "section": "2.1"}}


CORPUS = ChunkCorpus([
    chunk("c1", "Stratified sampling takes a random sample from each stratum.", 1, 2),
    chunk("c2", "Cluster sampling selects whole groups such as school districts.", 2, 2),
    chunk("c3", "Pseudonymization replaces identifiers with codewords.", 3, 8),
])


class FixedRetriever:
    """Retriever giả trả về danh sách cố định, đếm số lần gọi."""
    name = "fixed"

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def search(self, query, *, filters=None, top_k=3):
        self.calls.append((query, filters, top_k))
        return [{**r, "score": 1.0, "rank": i} for i, r in enumerate(self.rows[:top_k], 1)]


def tool_msg(name, **args):
    call = SimpleNamespace(id=f"id-{name}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return SimpleNamespace(content=None, tool_calls=[call])


class ScriptedLLM:
    """LLM giả trả lời theo kịch bản; ghi lại messages để kiểm tra."""

    def __init__(self, script):
        self.script, self.usage, self.seen = iter(script), Usage(), []

    def chat(self, messages, tools=None, tool_choice=None, json_mode=False):
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        self.usage.add("fake", 10, 5)
        return next(self.script)


class EvidenceTests(unittest.TestCase):
    def test_hit_ignores_whitespace_and_case_but_checks_page(self):
        c = chunk("c", "Taking a random\nsample from EACH stratum.", page=5)
        self.assertTrue(gold_hits(c, [{"source_id": "doc2", "page": 5, "evidence": "random sample from each"}]))
        self.assertFalse(gold_hits(c, [{"source_id": "doc2", "page": 6, "evidence": "random sample from each"}]))


class RetrieverTests(unittest.TestCase):
    def test_bm25_ranks_keyword_match_and_applies_filter(self):
        bm = BM25Retriever(CORPUS)
        self.assertEqual(bm.search("cluster sampling districts")[0]["id"], "c2")
        self.assertEqual([r["id"] for r in bm.search("sampling", filters={"chapter": 8})], [])

    def test_rrf_rewards_agreement(self):
        a, b = FixedRetriever([CORPUS.by_id["c1"], CORPUS.by_id["c2"]]), FixedRetriever([CORPUS.by_id["c2"], CORPUS.by_id["c3"]])
        self.assertEqual(HybridRetriever([a, b]).search("q")[0]["id"], "c2")

    def test_rerank_keeps_all_candidates_when_llm_omits_ids(self):
        base = FixedRetriever(CORPUS.chunks)
        llm = ScriptedLLM([SimpleNamespace(content='{"ranking": ["c3"]}', tool_calls=None)])
        out = RerankRetriever(base, llm).search("q", top_k=3)
        self.assertEqual([r["id"] for r in out], ["c3", "c1", "c2"])


class AgentTests(unittest.TestCase):
    def run_script(self, script, rows=None, **kw):
        llm, ret = ScriptedLLM(script), FixedRetriever(rows if rows is not None else CORPUS.chunks)
        params = dict(filters={"chapter": 2}, top_k=2, max_searches=2, budget_chars=4000)
        params.update(kw)
        return run_agent("How is stratified sampling done?", llm, ret, **params), llm, ret

    def test_answer_with_valid_citation(self):
        out, llm, ret = self.run_script([tool_msg("search_documents", query="stratified sampling"),
                                         tool_msg("submit_answer", answer="Random sample per stratum [c1].", citations=["c1"])])
        self.assertEqual(out["status"], "answered")
        self.assertEqual(ret.calls, [("stratified sampling", {"chapter": 2}, 2)])   # filter từ người dùng
        self.assertEqual(out["context_ids"], ["c1", "c2"])

    def test_citation_outside_context_is_rejected_after_one_fix(self):
        bad = tool_msg("submit_answer", answer="x [c3]", citations=["c3"])
        out, _, _ = self.run_script([tool_msg("search_documents", query="q"), bad, bad])
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(sum(e["event"] == "invalid_answer" for e in out["events"]), 2)

    def test_search_limit_and_budget(self):
        out, _, ret = self.run_script([tool_msg("search_documents", query="a"), tool_msg("search_documents", query="b"),
                                       tool_msg("search_documents", query="c"), tool_msg("abstain", reason="none")],
                                      budget_chars=90)
        self.assertEqual(len(ret.calls), 2)
        self.assertEqual(out["status"], "abstain")
        self.assertLessEqual(out["context_chars"], 90)
        self.assertTrue(any(e["event"] == "search_limit" for e in out["events"]))

    def test_gold_never_reaches_model(self):
        _, llm, _ = self.run_script([tool_msg("search_documents", query="q"), tool_msg("abstain", reason="x")])
        dumped = json.dumps(llm.seen)
        for key in ('"gold"', '"evidence":', "expected_answer", "gold_page"):
            self.assertNotIn(key, dumped)

    def test_provider_error_is_reported_not_hidden(self):
        class Boom(ScriptedLLM):
            def chat(self, *a, **k):
                raise RuntimeError("quota")
        out = run_agent("q", Boom([]), FixedRetriever([]), filters=None, top_k=3, max_searches=1, budget_chars=100)
        self.assertEqual(out["status"], "error")


class ContextTests(unittest.TestCase):
    def test_dedup_and_budget(self):
        ctx = ContextWindow(120)
        added, dup, over = ctx.add([CORPUS.by_id["c1"], CORPUS.by_id["c1"], CORPUS.by_id["c2"]])
        self.assertEqual([c["id"] for c in added], ["c1"])
        self.assertEqual(dup, ["c1"])
        self.assertEqual(over, ["c2"])


if __name__ == "__main__":
    unittest.main()
