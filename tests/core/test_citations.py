"""Tests for citation extraction and matching functionality."""
import pytest
from src.agents.content_qa.tools import _extract_relevant_excerpt, _compute_match_score


class TestCitationExtraction:
    """Test citation extraction and matching logic."""
    
    def test_excerpt_extraction_finds_relevant_sentence(self):
        """Test that relevant excerpts are correctly extracted."""
        body = (
            "Returns are accepted within 48 hours. "
            "Perishable goods cannot be returned. "
            "Contact support with your order number."
        )
        answer = "You can return items within 48 hours [1]."
        excerpt = _extract_relevant_excerpt(answer, body, "[1]")
        assert "48 hours" in excerpt

    def test_match_score_identical(self):
        """Test match score for identical text."""
        text = "Returns are accepted within 48 hours."
        score = _compute_match_score(text, text)
        assert score == 1.0

    def test_match_score_empty(self):
        """Test match score handling for empty strings."""
        assert _compute_match_score("", "some text") == 0.0
        assert _compute_match_score("text", "") == 0.0

    def test_match_score_partial(self):
        """Test match score for partial text overlap."""
        excerpt = "returns within 48 hours"
        body = "Returns are accepted within 48 hours of delivery for defective items."
        score = _compute_match_score(excerpt, body)
        assert 0.0 < score < 1.0