"""Unit tests for core functions: metadata filtering, citation extraction, ranking."""
import pytest
from difflib import SequenceMatcher

from app.ingestion.retriever import build_metadata_filter
from app.agent.nodes import (
    _extract_relevant_excerpt,
    _compute_match_score,
    validate_input,
    VALID_COUNTRIES,
    COUNTRY_LANGUAGES,
)


class TestMetadataFiltering:
    def test_filter_builds_correct_country_and_language(self):
        f = build_metadata_filter("A", "en")
        conditions = f.must
        assert len(conditions) == 2
        assert conditions[0].key == "country"
        assert conditions[0].match.value == "A"
        assert conditions[1].key == "language"
        assert conditions[1].match.value == "en"

    def test_filter_with_locale_code(self):
        f = build_metadata_filter("C", "fr_CA")
        conditions = f.must
        assert conditions[1].match.value == "fr_CA"

    def test_all_countries_have_language_mappings(self):
        for country in VALID_COUNTRIES:
            assert country in COUNTRY_LANGUAGES
            assert len(COUNTRY_LANGUAGES[country]) >= 1
            assert "en" in COUNTRY_LANGUAGES[country]


class TestCitationExtraction:
    def test_excerpt_extraction_finds_relevant_sentence(self):
        body = (
            "Returns are accepted within 48 hours. "
            "Perishable goods cannot be returned. "
            "Contact support with your order number."
        )
        answer = "You can return items within 48 hours [1]."
        excerpt = _extract_relevant_excerpt(answer, body, "[1]")
        assert "48 hours" in excerpt

    def test_match_score_identical(self):
        text = "Returns are accepted within 48 hours."
        score = _compute_match_score(text, text)
        assert score == 1.0

    def test_match_score_empty(self):
        assert _compute_match_score("", "some text") == 0.0
        assert _compute_match_score("text", "") == 0.0

    def test_match_score_partial(self):
        excerpt = "returns within 48 hours"
        body = "Returns are accepted within 48 hours of delivery for defective items."
        score = _compute_match_score(excerpt, body)
        assert 0.0 < score < 1.0


class TestInputValidation:
    def test_valid_country(self):
        state = {"question": "test", "country": "A", "language": "en"}
        result = validate_input(state)
        assert result["route"] == "retrieve"
        assert result["error"] is None

    def test_invalid_country(self):
        state = {"question": "test", "country": "X", "language": "en"}
        result = validate_input(state)
        assert result["route"] == "error"
        assert "Invalid country" in result["error"]
