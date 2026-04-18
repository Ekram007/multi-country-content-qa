"""Tests for agent input validation."""
import pytest
from src.agents.content_qa.tools import validate_input


class TestInputValidation:
    """Test agent input validation logic."""
    
    def test_valid_country(self):
        """Test validation with valid country and language."""
        state = {"question": "test", "country": "A", "language": "en"}
        result = validate_input(state)
        assert result["route"] == "retrieve"
        assert result["error"] is None

    def test_invalid_country(self):
        """Test validation with invalid country."""
        state = {"question": "test", "country": "X", "language": "en"}
        result = validate_input(state)
        assert result["route"] == "error"
        assert "Invalid country" in result["error"]