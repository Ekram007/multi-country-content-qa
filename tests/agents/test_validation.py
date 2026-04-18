"""Tests for API request validation (country / language)."""

import pytest
from pydantic import ValidationError

from src.schema.models import AskRequest


class TestInputValidation:
    """AskRequest must reject invalid countries before the agent runs."""

    def test_valid_country(self):
        """Valid country and language parse and normalize."""
        r = AskRequest(question="test", country="A", language="EN")
        assert r.country == "A"
        assert r.language == "en"

    def test_invalid_country(self):
        """Invalid country is rejected by Pydantic (HTTP 422 at API layer)."""
        with pytest.raises(ValidationError):
            AskRequest(question="test", country="X", language="en")
