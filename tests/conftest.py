"""Shared test configuration and fixtures."""
import pytest
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_content():
    """Sample content for testing."""
    return {
        "content_id": "test_faq_001",
        "type": "FAQ",
        "title": "Test FAQ",
        "body": "This is a sample FAQ content for testing purposes.",
        "country": "A",
        "language": "en"
    }


@pytest.fixture
def sample_chunks():
    """Sample chunks for retrieval testing."""
    return [
        {
            "content_id": "test_faq_001",
            "type": "FAQ", 
            "title": "Test FAQ",
            "body": "Returns are accepted within 48 hours of purchase.",
            "country": "A",
            "language": "en",
            "score": 0.95
        },
        {
            "content_id": "test_terms_001",
            "type": "TERMS_AND_CONDITIONS",
            "title": "Test Terms",
            "body": "All sales are final unless defective.",
            "country": "A", 
            "language": "en",
            "score": 0.82
        }
    ]