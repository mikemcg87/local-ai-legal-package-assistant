import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bundle_rag.evaluation import (
    load_suite,
    page_hit_and_mrr,
    required_term_coverage,
    summarize_chunk,
)


class EvaluationHelpersTest(unittest.TestCase):
    def test_page_hit_and_mrr_returns_first_matching_rank(self):
        hit, mrr = page_hit_and_mrr([4, 7, 9], (7, 10))

        self.assertTrue(hit)
        self.assertEqual(mrr, 0.5)

    def test_page_hit_and_mrr_returns_zero_for_miss(self):
        hit, mrr = page_hit_and_mrr([4, 7, 9], (10, 11))

        self.assertFalse(hit)
        self.assertEqual(mrr, 0.0)

    def test_required_term_coverage_is_case_insensitive(self):
        matched, total, ratio = required_term_coverage(
            "The patient had chest pain and raised Troponin.",
            ("chest pain", "troponin", "xray"),
        )

        self.assertEqual(matched, 2)
        self.assertEqual(total, 3)
        self.assertAlmostEqual(ratio, 2 / 3)

    def test_load_suite_validates_and_normalises_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite_path = Path(tmp) / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "case": "example_case",
                        "questions": [
                            {
                                "id": "admission",
                                "question": "What was documented on admission?",
                                "expected_pages": [12, "13"],
                                "required_terms": ["admission", "discharge"],
                                "top_k": 7,
                                "max_page": 100,
                            }
                        ],
                    }
                )
            )

            suite = load_suite(suite_path)

        self.assertEqual(suite.case, "example_case")
        self.assertEqual(len(suite.questions), 1)
        question = suite.questions[0]
        self.assertEqual(question.id, "admission")
        self.assertEqual(question.expected_pages, (12, 13))
        self.assertEqual(question.required_terms, ("admission", "discharge"))
        self.assertEqual(question.top_k, 7)
        self.assertEqual(question.max_page, 100)

    def test_summarize_chunk_truncates_preview_and_keeps_scores(self):
        chunk = {
            "id": "chunk-1",
            "page": 42,
            "score": 0.9,
            "semantic_score": 0.8,
            "bm25_score": 0.7,
            "rerank_score": 0.6,
            "dates": ["2025-10-22"],
            "text": "x" * 600,
        }

        summary = summarize_chunk(chunk)

        self.assertEqual(summary["id"], "chunk-1")
        self.assertEqual(summary["page"], 42)
        self.assertEqual(summary["score"], 0.9)
        self.assertEqual(summary["semantic_score"], 0.8)
        self.assertEqual(summary["bm25_score"], 0.7)
        self.assertEqual(summary["rerank_score"], 0.6)
        self.assertEqual(summary["dates"], ["2025-10-22"])
        self.assertEqual(len(summary["preview"]), 500)
        self.assertTrue(summary["preview"].endswith("..."))


if __name__ == "__main__":
    unittest.main()
