"""State management for the content Q&A agent."""

from typing import Any, Dict, List, Optional, TypedDict


class ContentChunk(TypedDict):
    """Type definition for content chunk from retrieval."""
    content_id: str
    country: str
    language: str
    type: str
    version: float
    title: str
    body: str
    updated_at: str


class Citation(TypedDict):
    """Type definition for citation in response."""
    content_id: str
    type: str
    excerpt: str
    match_score: float


class AgentState(TypedDict, total=False):
    """State container for the content Q&A agent workflow.
    
    This TypedDict defines the shared state that flows through
    the LangGraph agent nodes during question answering.
    """
    
    # Input parameters
    question: str
    country: str  
    language: str
    
    # Retrieval results
    retrieved_chunks: List[ContentChunk]
    fallback_used: bool
    fallback_language: Optional[str]
    
    # LLM synthesis results
    answer: str
    citations: List[Citation]
    llm_failed: bool
    
    # Error handling
    error: Optional[str]
    
    # Routing control
    route: str  # Controls next node in the workflow


# Type aliases for better code readability
AgentStateType = AgentState
ContentChunkType = ContentChunk
CitationType = Citation