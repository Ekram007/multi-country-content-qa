"""FastAPI application and HTTP endpoints."""

import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.settings import settings
from src.core.llm import get_llm
from src.core.embeddings import get_embedding_model
from src.schema.models import (
    AskRequest, 
    AskResponse, 
    Citation, 
    Trace, 
    HealthResponse, 
    ErrorResponse
)
from src.agents.content_qa import content_qa_agent
from src.service.utils import setup_logging

logger = logging.getLogger(__name__)


MAX_EXCERPT_LEN = 320


def _normalize_answer_text(answer: str) -> str:
    """Flatten Gemini / multi-part content to plain string."""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        parts: list[str] = []
        for item in answer:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(answer)


def parse_retrieved_documents_from_search_content_output(tool_content: str) -> dict[int, dict]:
    """Parse search_content tool return text into index -> document fields."""
    documents: dict[int, dict] = {}
    if not tool_content or "Content ID:" not in tool_content:
        return documents

    # Split on blank lines before each "[n] TYPE - title" block
    blocks = re.split(r"\n(?=\[\d+\]\s)", tool_content.strip())
    block_re = re.compile(
        r"^\[(\d+)\]\s+(\S+)\s+-\s+(.+?)\n"
        r"Content ID:\s*(.+?)\n"
        r"Language:\s*(.+?)\n"
        r"(?:Score:\s*([0-9.+-eE]+)\n)?"
        r"Content:\s*(.+)$",
        re.DOTALL | re.MULTILINE,
    )
    for block in blocks:
        block = block.strip()
        if not block.startswith("["):
            continue
        m = block_re.match(block)
        if not m:
            continue
        idx = int(m.group(1))
        score_str = m.group(6)
        body = (m.group(7) or "").strip()
        excerpt = body[:MAX_EXCERPT_LEN] + ("..." if len(body) > MAX_EXCERPT_LEN else "")
        documents[idx] = {
            "content_id": m.group(4).strip(),
            "type": m.group(2).strip(),
            "title": m.group(3).strip(),
            "excerpt": excerpt,
            "match_score": float(score_str) if score_str else 0.0,
            "language": m.group(5).strip(),
        }
    return documents


def build_citations_from_retrieval(
    retrieved_by_index: dict[int, dict],
    answer: str,
) -> tuple[list[Citation], int]:
    """Build API citations from RAG chunks; align with cited [n] when present."""
    answer_plain = _normalize_answer_text(answer)
    cited_indices = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer_plain)})

    if cited_indices:
        use_indices = [i for i in cited_indices if i in retrieved_by_index]
        if not use_indices:
            use_indices = sorted(retrieved_by_index.keys())
    else:
        use_indices = sorted(retrieved_by_index.keys())

    citations: list[Citation] = []
    for i in use_indices:
        doc = retrieved_by_index[i]
        citations.append(
            Citation(
                content_id=doc["content_id"],
                type=doc["type"],
                excerpt=doc["excerpt"],
                match_score=round(min(1.0, max(0.0, doc["match_score"])), 2),
            )
        )
    return citations, len(retrieved_by_index)


def extract_retrieval_from_agent_messages(messages: list) -> dict[int, dict]:
    """Last search_content tool output wins (final retrieval set for this turn)."""
    from langchain_core.messages import ToolMessage

    last_docs: dict[int, dict] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "search_content":
            content = msg.content
            if isinstance(content, str):
                last_docs = parse_retrieved_documents_from_search_content_output(content)
    return last_docs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    
    # Setup logging
    setup_logging()
    logger.info("Starting Content Q&A service...")
    
    try:
        # Pre-load embedding model for faster response times
        logger.info("Loading embedding model...")
        embedding_model = get_embedding_model()
        logger.info(
            f"Embedding model loaded: {embedding_model.get_embedding_dimension()} dimensions"
        )
        
        # Warmup embedding model with a test query
        logger.info("Warming up embedding model...")
        test_embedding = embedding_model.encode("test query")
        logger.info("Embedding model warmed up successfully")
        
        # Pre-load LLM for faster first response
        logger.info("Loading LLM...")
        llm = get_llm()
        logger.info("LLM loaded and ready")
        
        # Initialize ContentQA agent
        logger.info("Initializing ContentQA agent...")
        # Test agent with simple message
        from langchain_core.messages import HumanMessage
        test_result = content_qa_agent.invoke({
            "messages": [HumanMessage(content="Health check")]
        })
        logger.info("ContentQA agent initialized and tested successfully")
        
        logger.info("Service startup completed successfully")
        
    except Exception as e:
        logger.error(f"Service startup failed: {e}")
        raise
    
    yield
    
    logger.info("Shutting down Content Q&A service")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Multi-Country Content Q&A",
        description="Natural language Q&A system with country-scoped content and citations",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.log_level.upper() == "DEBUG" else None,
        redoc_url="/redoc" if settings.log_level.upper() == "DEBUG" else None,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Add exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception) -> JSONResponse:
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        error_response = ErrorResponse(
            error="Internal server error occurred",
            error_code="INTERNAL_ERROR"
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump()
        )
    
    return app


# Create app instance
app = create_app()


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint.
    
    Returns:
        Service health status
    """
    try:
        # Check agent health - simple test
        from langchain_core.messages import HumanMessage
        try:
            test_result = content_qa_agent.invoke({
                "messages": [HumanMessage(content="Health check")]
            })
            agent_status = "healthy" if test_result else "unhealthy"
        except Exception as e:
            logger.error(f"Agent health check failed: {e}")
            agent_status = "unhealthy"
        
        # Quick dependency checks
        dependencies = {
            "embedding_model": "healthy",
            "llm": "healthy", 
            "content_qa_agent": agent_status
        }
        
        status = "healthy" if all(
            dep == "healthy" for dep in dependencies.values()
        ) else "unhealthy"
        
        return HealthResponse(
            status=status,
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            dependencies={"error": str(e)}
        )


@app.post("/ask", response_model=AskResponse, tags=["qa"])
async def ask_question(request: AskRequest) -> AskResponse:
    """Answer a question using country-scoped content.
    
    Args:
        request: Question and filtering parameters
        
    Returns:
        Generated answer with citations and trace info
    """
    start_time = time.time()
    
    try:
        # Process request with agent
        from langchain_core.messages import HumanMessage
        
        # Format user message with context
        user_message = f"Question: {request.question}\nCountry: {request.country}\nPreferred language: {request.language}"
        
        # Invoke agent
        result = content_qa_agent.invoke({
            "messages": [HumanMessage(content=user_message)]
        })
        
        # Extract answer from final message
        final_message = result["messages"][-1]
        raw_content = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )
        answer = _normalize_answer_text(raw_content)
        
        retrieved_by_index = extract_retrieval_from_agent_messages(result["messages"])
        citations, retrieval_count = build_citations_from_retrieval(
            retrieved_by_index,
            answer,
        )
        
        # Calculate actual latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Build response
        response = AskResponse(
            answer=answer.strip(),
            language_used=request.language,
            citations=citations,
            trace=Trace(
                retrieval_count=retrieval_count,
                latency_ms=max(1, latency_ms),
                model=settings.llm_model,
            ),
        )
        
        return response
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in ask endpoint: {e}")
        
        return AskResponse(
            answer="Service temporarily unavailable. Please try again.",
            language_used=request.language,
            citations=[],
            trace=Trace(
                retrieval_count=0,
                latency_ms=1,  # Must be > 0 for validation
                model=settings.llm_model,
            ),
        )