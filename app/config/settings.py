from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    openai_api_key: str = ""
    google_api_key: str = ""
    openrouter_api_key: str = ""

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    embedding_model: str = "all-MiniLM-L6-v2"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "content_qa"

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    corpus_path: str = "data/corpus.jsonl"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
