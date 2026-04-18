"""Data models and schemas for the content Q&A system."""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ContentItem(BaseModel):
    """Content item schema for corpus data."""
    
    content_id: str = Field(..., description="Unique content identifier")
    country: Literal["A", "B", "C", "D"] = Field(..., description="Country code")
    language: str = Field(..., min_length=2, description="Language code (en, es, hi, fr_CA)")
    type: str = Field(..., description="Content type (FAQ, TERMS_AND_CONDITIONS, etc.)")
    version: float = Field(..., gt=0, description="Content version number")
    title: str = Field(..., min_length=1, description="Content title")
    body: str = Field(..., min_length=1, description="Content body text")
    updated_at: str = Field(..., description="Last update timestamp (ISO format)")
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language code format."""
        if not v or len(v) < 2:
            raise ValueError("Language code must be at least 2 characters")
        return v.lower()
    
    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        """Validate and normalize country code."""
        return v.upper()


class AskRequest(BaseModel):
    """Request schema for Q&A endpoint."""
    
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Natural language question to answer"
    )
    country: Literal["A", "B", "C", "D"] = Field(
        ..., 
        description="Country code for content filtering"
    )
    language: str = Field(
        ..., 
        min_length=2, 
        max_length=10,
        description="Language code for content filtering (en, es, hi, fr_CA)"
    )
    
    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate and clean question text."""
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v
    
    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        """Validate and normalize country code."""
        return v.upper()
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate and normalize language code."""
        return v.lower().strip()


class Citation(BaseModel):
    """Citation schema for source attribution."""
    
    content_id: str = Field(..., description="Source content identifier")
    type: str = Field(..., description="Content type")
    excerpt: str = Field(
        ..., 
        max_length=500, 
        description="Relevant excerpt from source content"
    )
    match_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Relevance score between 0.0 and 1.0"
    )
    
    @field_validator("match_score")
    @classmethod
    def validate_match_score(cls, v: float) -> float:
        """Validate and round match score."""
        return round(v, 2)


class Trace(BaseModel):
    """Trace information for request processing."""
    
    retrieval_count: int = Field(
        ..., 
        ge=0, 
        description="Number of content chunks retrieved"
    )
    latency_ms: int = Field(
        ..., 
        gt=0, 
        description="Total request processing time in milliseconds"
    )
    model: str = Field(..., description="LLM model used for answer generation")


class AskResponse(BaseModel):
    """Response schema for Q&A endpoint."""
    
    answer: str = Field(..., description="Generated answer to the question")
    language_used: str = Field(
        ..., 
        description="Language code of content used for answer generation"
    )
    citations: List[Citation] = Field(
        default_factory=list, 
        description="Source citations for answer verification"
    )
    trace: Trace = Field(..., description="Request processing trace information")
    
    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        """Validate answer content."""
        if not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: Literal["healthy", "unhealthy"] = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field("0.1.0", description="API version")
    dependencies: Optional[dict] = Field(
        default=None, 
        description="Status of external dependencies"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Machine-readable error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier for tracking")


# Type aliases for convenience
ContentItemType = ContentItem
AskRequestType = AskRequest
AskResponseType = AskResponse
CitationType = Citation
TraceType = Trace