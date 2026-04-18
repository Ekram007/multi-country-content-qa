"""Service utilities for logging and error handling."""

import logging
from typing import Tuple

from src.core.settings import settings


def setup_logging() -> None:
    """Setup application logging configuration."""
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Reduce verbosity from external libraries
    external_loggers = [
        "httpx",
        "sentence_transformers",
        "transformers", 
        "google_genai.models",
        "urllib3.connectionpool",
        "httpcore.connection",
        "httpcore.http11",
    ]
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Set specific log levels for our modules
    logging.getLogger("src.core").setLevel(logging.INFO)
    logging.getLogger("src.agents").setLevel(logging.INFO)
    logging.getLogger("src.service").setLevel(logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {settings.log_level}")


def handle_agent_error(error: Exception) -> Tuple[str, str]:
    """Handle and categorize agent errors.
    
    Args:
        error: The exception that occurred
        
    Returns:
        Tuple of (user_message, error_code)
    """
    error_str = str(error).lower()
    
    # Categorize common errors
    if "rate limit" in error_str or "429" in error_str:
        return (
            "Service is temporarily busy due to high demand. Please wait a moment and try again.",
            "RATE_LIMIT_EXCEEDED"
        )
    
    if "api key" in error_str or "authentication" in error_str:
        return (
            "Service configuration error. Please contact support.",
            "AUTHENTICATION_ERROR"
        )
    
    if "timeout" in error_str:
        return (
            "Request timed out. Please try again with a simpler question.",
            "TIMEOUT_ERROR"
        )
    
    if "connection" in error_str or "network" in error_str:
        return (
            "Service temporarily unavailable due to network issues. Please try again.",
            "NETWORK_ERROR"
        )
    
    if "qdrant" in error_str or "vector" in error_str:
        return (
            "Content database temporarily unavailable. Please try again.",
            "DATABASE_ERROR"
        )
    
    # Default error message
    return (
        "An unexpected error occurred while processing your request. Please try again.",
        "UNKNOWN_ERROR"
    )


def validate_country_language_combo(country: str, language: str) -> bool:
    """Validate if a country-language combination is supported.
    
    Args:
        country: Country code
        language: Language code
        
    Returns:
        True if combination is potentially valid
    """
    country = country.upper()
    language = language.lower()
    
    # Known country-language mappings from the corpus
    known_mappings = {
        "A": ["en", "hi"],
        "B": ["en", "es"],
        "C": ["en", "fr_ca"],
        "D": ["en"],
    }
    
    if country not in known_mappings:
        return False
        
    # Allow any reasonable language code, not just known ones
    # This provides flexibility for future content additions
    return len(language) >= 2


def sanitize_log_data(data: dict) -> dict:
    """Sanitize sensitive data for logging.
    
    Args:
        data: Dictionary that might contain sensitive information
        
    Returns:
        Sanitized dictionary safe for logging
    """
    sensitive_keys = {"api_key", "token", "password", "secret", "auth"}
    
    sanitized = {}
    for key, value in data.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value)
        elif isinstance(value, str) and len(value) > 100:
            # Truncate very long strings
            sanitized[key] = value[:100] + "..."
        else:
            sanitized[key] = value
            
    return sanitized


def format_processing_time(start_time: float, end_time: float) -> str:
    """Format processing time in a human-readable way.
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp
        
    Returns:
        Formatted time string
    """
    duration_ms = int((end_time - start_time) * 1000)
    
    if duration_ms < 1000:
        return f"{duration_ms}ms"
    elif duration_ms < 60000:
        return f"{duration_ms / 1000:.1f}s"
    else:
        return f"{duration_ms / 60000:.1f}m"