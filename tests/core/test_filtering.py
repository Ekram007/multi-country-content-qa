"""Tests for metadata filtering functionality."""
import pytest
from src.ingestion.retriever import build_metadata_filter
from src.agents.content_qa.tools import VALID_COUNTRIES, COUNTRY_LANGUAGES


class TestMetadataFiltering:
    """Test metadata filtering for vector search."""
    
    def test_filter_builds_correct_country_and_language(self):
        """Test that filter correctly builds country and language conditions."""
        f = build_metadata_filter("A", "en")
        conditions = f.must
        assert len(conditions) == 2
        assert conditions[0].key == "country"
        assert conditions[0].match.value == "A"
        assert conditions[1].key == "language"
        assert conditions[1].match.value == "en"

    def test_filter_with_locale_code(self):
        """Test that filter preserves locale codes like fr_CA."""
        f = build_metadata_filter("C", "fr_CA")
        conditions = f.must
        assert conditions[1].match.value == "fr_CA"

    def test_all_countries_have_language_mappings(self):
        """Test that all valid countries have language mappings with English fallback."""
        for country in VALID_COUNTRIES:
            assert country in COUNTRY_LANGUAGES
            assert len(COUNTRY_LANGUAGES[country]) >= 1
            assert "en" in COUNTRY_LANGUAGES[country]