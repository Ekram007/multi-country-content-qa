"""Application settings and configuration management."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""
    
    # LLM Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    google_api_key: str = Field(default="", description="Google Gemini API key")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")

    llm_provider: Literal["openai", "google", "openrouter"] = Field(
        default="openai", description="LLM provider to use"
    )
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model name")

    # Embedding Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", 
        description="Sentence transformer embedding model"
    )

    # Vector Database Configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant server port")
    qdrant_collection: str = Field(default="content_qa", description="Qdrant collection name")

    # API Server Configuration
    app_host: str = Field(default="0.0.0.0", description="API server host")
    app_port: int = Field(default=8000, description="API server port")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )

    # Data Configuration
    corpus_path: str = Field(
        default="data/corpus.jsonl", 
        description="Path to corpus JSONL file"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()