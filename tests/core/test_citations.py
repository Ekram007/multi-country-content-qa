"""Tests for citation parsing and assembly from search_content tool output."""

import pytest

from src.schema.models import Citation
from src.service.api import (
    build_citations_from_retrieval,
    parse_retrieved_documents_from_search_content_output,
    select_excerpt_and_answer_overlap,
)


def _sample_tool_output() -> str:
    """Minimal formatted block matching tools.search_content output."""
    return (
        "Content found in es for country B:\n"
        "Search query: returns\n"
        "\n"
        "[1] FAQ - Returns\n"
        "Content ID: b_faq_returns_es\n"
        "Language: es\n"
        "Score: 0.9123\n"
        "Content: Puede devolver cualquier artículo dentro de los 7 días hábiles.\n"
    )


class TestCitationParsing:
    """Parse tool output into indexed documents."""

    def test_parse_retrieved_documents_extracts_score_and_content_id(self):
        docs = parse_retrieved_documents_from_search_content_output(_sample_tool_output())
        assert 1 in docs
        assert docs[1]["content_id"] == "b_faq_returns_es"
        assert docs[1]["type"] == "FAQ"
        assert pytest.approx(docs[1]["match_score"], rel=1e-4) == 0.9123
        assert "7 días" in docs[1]["body"]
        assert "7 días" in docs[1]["excerpt"]

    def test_excerpt_prefers_sentences_overlapping_answer(self):
        body = (
            "These terms govern your use of the platform in Country C. "
            "You must be a legally registered business entity to open an account. "
            "In Country C you can close your account directly from account settings. "
            "Closure takes effect immediately. Any unpaid invoices must be settled within 30 days of closure."
        )
        answer = (
            "Yes. In Country C, you can close your account from account settings "
            "and closure is immediate. Unpaid invoices must be settled within 30 days."
        )
        excerpt, overlap = select_excerpt_and_answer_overlap(body, answer, 320)
        assert "close your account" in excerpt.lower()
        assert "30 days" in excerpt
        assert overlap > 0.25

    def test_match_score_blends_retrieval_and_answer_overlap(self):
        tool = (
            "Content found in en for country C:\nSearch query: close account\n\n"
            "[1] TERMS_AND_CONDITIONS - Terms\n"
            "Content ID: c_tc_en_v1\n"
            "Language: en\n"
            "Score: 0.40\n"
            "Content: These terms govern your use of the platform in Country C. "
            "In Country C you can close your account directly from account settings. "
            "Closure takes effect immediately.\n"
        )
        docs = parse_retrieved_documents_from_search_content_output(tool)
        answer = "You may close your account from settings; closure is immediate [1]."
        citations, _ = build_citations_from_retrieval(docs, answer)
        assert len(citations) == 1
        assert "close your account" in citations[0].excerpt.lower()
        # Combined score should reflect both retrieval (0.40) and strong overlap (>0.4)
        assert citations[0].match_score > 0.45

    def test_build_citations_aligns_with_citation_marker(self):
        docs = parse_retrieved_documents_from_search_content_output(_sample_tool_output())
        answer = "Política: ver [1]."
        citations, count = build_citations_from_retrieval(docs, answer)
        assert count == 1
        assert len(citations) == 1
        assert isinstance(citations[0], Citation)
        assert citations[0].content_id == "b_faq_returns_es"

    def test_build_citations_empty_when_no_docs(self):
        citations, count = build_citations_from_retrieval({}, "No sources [1].")
        assert count == 0
        assert citations == []
