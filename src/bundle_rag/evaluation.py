"""Private evaluation harness for bundle retrieval quality.

Eval suites can point at real local cases, but those suites belong under
evals/private/ and must not be committed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from bundle_rag.config import CASES_DIR, TOP_K
from bundle_rag.vectorstore import HybridVectorStore


@dataclass(frozen=True)
class EvalQuestion:
    """One retrieval-quality question."""

    id: str
    question: str
    expected_pages: tuple[int, ...] = ()
    required_terms: tuple[str, ...] = ()
    top_k: int = TOP_K
    max_page: int | None = None
    semantic_weight: float = 0.5
    bm25_weight: float = 0.5


@dataclass(frozen=True)
class EvalSuite:
    """A retrieval eval suite for one local case."""

    case: str
    questions: tuple[EvalQuestion, ...]


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("expected_pages must be a list of page numbers")
    return tuple(int(v) for v in value)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("required_terms must be a list of strings")
    return tuple(str(v) for v in value if str(v).strip())


def load_suite(path: Path) -> EvalSuite:
    """Load a JSON eval suite."""
    with open(path) as f:
        raw = json.load(f)

    case = str(raw.get("case", "")).strip()
    if not case:
        raise ValueError("Eval suite must include a case name")

    questions = []
    for i, item in enumerate(raw.get("questions", []), start=1):
        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError(f"Question {i} is missing question text")

        questions.append(
            EvalQuestion(
                id=str(item.get("id") or f"q{i}"),
                question=question,
                expected_pages=_as_int_tuple(item.get("expected_pages")),
                required_terms=_as_str_tuple(item.get("required_terms")),
                top_k=int(item.get("top_k", TOP_K)),
                max_page=int(item["max_page"]) if item.get("max_page") else None,
                semantic_weight=float(item.get("semantic_weight", 0.5)),
                bm25_weight=float(item.get("bm25_weight", 0.5)),
            )
        )

    if not questions:
        raise ValueError("Eval suite must include at least one question")

    return EvalSuite(case=case, questions=tuple(questions))


def page_hit_and_mrr(retrieved_pages: list[int], expected_pages: tuple[int, ...]) -> tuple[bool, float]:
    """Return whether any expected page was retrieved, plus reciprocal rank."""
    if not expected_pages:
        return False, 0.0

    expected = set(expected_pages)
    for idx, page in enumerate(retrieved_pages, start=1):
        if page in expected:
            return True, 1.0 / idx
    return False, 0.0


def required_term_coverage(retrieved_text: str, required_terms: tuple[str, ...]) -> tuple[int, int, float]:
    """Return matched count, total terms, and coverage ratio."""
    if not required_terms:
        return 0, 0, 0.0

    haystack = retrieved_text.lower()
    matched = sum(1 for term in required_terms if term.lower() in haystack)
    return matched, len(required_terms), matched / len(required_terms)


def summarize_chunk(chunk: dict) -> dict:
    """Return a report-safe summary of one retrieved chunk."""
    text = chunk.get("text") or ""
    preview = text.replace("\n", " ").strip()
    if len(preview) > 500:
        preview = preview[:497] + "..."

    return {
        "id": chunk.get("id"),
        "page": chunk.get("page"),
        "score": chunk.get("score"),
        "semantic_score": chunk.get("semantic_score"),
        "bm25_score": chunk.get("bm25_score"),
        "rerank_score": chunk.get("rerank_score"),
        "dates": chunk.get("dates"),
        "preview": preview,
    }


def evaluate_question(store: HybridVectorStore, question: EvalQuestion) -> dict:
    """Evaluate one question against an already-built vector store."""
    chunks = store.query(
        question.question,
        top_k=question.top_k,
        max_page=question.max_page,
        semantic_weight=question.semantic_weight,
        bm25_weight=question.bm25_weight,
    )

    pages = [int(c["page"]) for c in chunks if c.get("page") and int(c["page"]) > 0]
    hit, mrr = page_hit_and_mrr(pages, question.expected_pages)
    combined_text = "\n\n".join(c.get("text") or "" for c in chunks)
    matched_terms, total_terms, term_coverage = required_term_coverage(
        combined_text,
        question.required_terms,
    )

    return {
        "id": question.id,
        "question": question.question,
        "expected_pages": list(question.expected_pages),
        "retrieved_pages": pages,
        "page_hit": hit,
        "page_mrr": mrr,
        "required_terms": list(question.required_terms),
        "matched_required_terms": matched_terms,
        "total_required_terms": total_terms,
        "required_term_coverage": term_coverage,
        "top_k": question.top_k,
        "chunks": [summarize_chunk(c) for c in chunks],
    }


def run_suite(suite_path: Path, cases_dir: Path = CASES_DIR) -> dict:
    """Run a retrieval eval suite and return a JSON-serialisable report."""
    suite = load_suite(suite_path)
    case_dir = cases_dir / suite.case
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    store = HybridVectorStore(case_dir)
    if store.count() == 0:
        raise ValueError(f"Vectorstore is empty for case: {suite.case}")

    question_reports = [evaluate_question(store, q) for q in suite.questions]
    page_questions = [r for r in question_reports if r["expected_pages"]]
    term_questions = [r for r in question_reports if r["total_required_terms"]]

    page_hit_rate = (
        sum(1 for r in page_questions if r["page_hit"]) / len(page_questions)
        if page_questions
        else 0.0
    )
    mean_page_mrr = (
        sum(r["page_mrr"] for r in page_questions) / len(page_questions)
        if page_questions
        else 0.0
    )
    mean_term_coverage = (
        sum(r["required_term_coverage"] for r in term_questions) / len(term_questions)
        if term_questions
        else 0.0
    )

    return {
        "suite_path": str(suite_path),
        "case": suite.case,
        "generated_at": datetime.now().isoformat(),
        "question_count": len(question_reports),
        "metrics": {
            "page_hit_rate": page_hit_rate,
            "mean_page_mrr": mean_page_mrr,
            "mean_required_term_coverage": mean_term_coverage,
        },
        "questions": question_reports,
    }


def write_report(report: dict, output_path: Path) -> Path:
    """Write an eval report as pretty JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    return output_path
