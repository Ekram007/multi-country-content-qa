from typing import TypedDict


class AgentState(TypedDict, total=False):
    question: str
    country: str
    language: str

    retrieved_chunks: list[dict]
    fallback_used: bool
    fallback_language: str | None

    answer: str
    citations: list[dict]
    llm_failed: bool

    error: str | None
    route: str
