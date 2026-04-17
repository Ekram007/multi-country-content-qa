from pydantic import BaseModel, Field


class ContentItem(BaseModel):
    content_id: str
    country: str
    language: str
    type: str
    version: float
    title: str
    body: str
    updated_at: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    country: str = Field(..., pattern="^[A-D]$", description="Country code (A, B, C, D)")
    language: str = Field(..., min_length=2, description="Language code (en, es, hi, fr_CA)")


class Citation(BaseModel):
    content_id: str
    type: str
    excerpt: str
    match_score: float


class Trace(BaseModel):
    retrieval_count: int
    latency_ms: int
    model: str


class AskResponse(BaseModel):
    answer: str
    language_used: str
    citations: list[Citation]
    trace: Trace
