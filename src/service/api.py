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


def extract_citations_from_answer(answer: str) -> list[Citation]:
    """Extract citations from answer text (basic implementation)."""
    citations = []
    
    # Find citation references like [1], [2], etc.
    citation_refs = re.findall(r'\[(\d+)\]', answer)
    
    # For now, create placeholder citations
    # In real implementation, this would match against retrieved documents
    for i, ref_num in enumerate(set(citation_refs)):
        citations.append(Citation(
            content_id=f"placeholder_content_{ref_num}",
            type="FAQ",  # Placeholder
            excerpt=f"Excerpt from source {ref_num}...",  # Placeholder
            match_score=0.8  # Placeholder
        ))
    
    return citations


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
        if hasattr(final_message, 'content'):
            if isinstance(final_message.content, str):
                answer = final_message.content
            elif isinstance(final_message.content, list):
                # Extract text from structured content
                answer = ""
                for item in final_message.content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        answer += item.get('text', '')
            else:
                answer = str(final_message.content)
        else:
            answer = str(final_message)
        
        # Count tool calls for retrieval_count
        tool_call_count = 0
        for msg in result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_call_count += len(msg.tool_calls)
        
        # Extract citations from answer (basic implementation)
        citations = extract_citations_from_answer(answer)
        
        # Calculate actual latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Build response
        response = AskResponse(
            answer=answer.strip(),
            language_used=request.language,  # Will be enhanced based on actual content used
            citations=citations,
            trace=Trace(
                retrieval_count=tool_call_count,
                latency_ms=latency_ms,
                model=settings.llm_model
            )
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