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


# Hard cap for API excerpts; spans are shrunk to matching words when possible.
MAX_EXCERPT_LEN = 200
# How much Qdrant retrieval vs answer–source overlap contributes to citation match_score.
_RETRIEVAL_SCORE_WEIGHT = 0.35
_ANSWER_OVERLAP_WEIGHT = 0.65

_CITATION_MARKERS_RE = re.compile(r"\[\d+\]")


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


def _answer_content_tokens(answer: str, min_len: int = 3) -> set[str]:
    """Word-like tokens from the final answer for overlap scoring (Unicode-aware)."""
    plain = _CITATION_MARKERS_RE.sub(" ", answer.lower())
    return {w for w in re.findall(r"\w+", plain, flags=re.UNICODE) if len(w) >= min_len}


def _tokens_from_text(text: str, min_len: int = 3) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower(), flags=re.UNICODE) if len(w) >= min_len}


def _split_into_sentences(body: str) -> list[str]:
    """Split on common sentence boundaries (incl. Devanagari danda)."""
    body = body.strip()
    if not body:
        return []
    parts = re.split(r"(?<=[.!?।])\s+", body)
    return [p.strip() for p in parts if p.strip()]


def _compress_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _compress_to_matching_word_span(text: str, answer_tokens: set[str], pad_words: int = 1) -> str:
    """Keep only a tight run of words around tokens that appear in the answer (plus small padding)."""
    text = text.strip()
    if not text or not answer_tokens:
        return text
    words = text.split()
    if not words:
        return text
    hit: list[int] = []
    for i, raw in enumerate(words):
        for tok in re.findall(r"\w+", raw.lower(), flags=re.UNICODE):
            if len(tok) >= 3 and tok in answer_tokens:
                hit.append(i)
                break
    if not hit:
        return text
    lo = max(0, min(hit) - pad_words)
    hi = min(len(words) - 1, max(hit) + pad_words)
    return _compress_whitespace(" ".join(words[lo : hi + 1]))


def select_excerpt_and_answer_overlap(
    body: str,
    answer: str,
    max_len: int,
) -> tuple[str, float]:
    """Pick a short excerpt whose wording overlaps the answer; return (excerpt, overlap score).

    Overlap is |answer_tokens ∩ window_tokens| / |answer_tokens| in [0, 1].
    Falls back to a length-truncated prefix when the answer gives no signal or overlap is negligible.
    """
    body = (body or "").strip()
    if not body:
        return "", 0.0

    answer_plain = _normalize_answer_text(answer)
    answer_tokens = _answer_content_tokens(answer_plain)
    if not answer_tokens:
        excerpt = body[:max_len] + ("..." if len(body) > max_len else "")
        return excerpt.rstrip(), 0.0

    sentences = _split_into_sentences(body)
    if not sentences:
        sentences = [body]

    def overlap_score(window_text: str) -> float:
        w_tokens = _tokens_from_text(window_text)
        if not w_tokens:
            return 0.0
        inter = len(answer_tokens & w_tokens)
        return inter / max(1, len(answer_tokens))

    best_overlap = 0.0
    best_window = sentences[0]

    max_k = min(3, len(sentences))
    for k in range(1, max_k + 1):
        for i in range(0, len(sentences) - k + 1):
            window_text = " ".join(sentences[i : i + k])
            score = overlap_score(window_text)
            if score > best_overlap:
                best_overlap = score
                best_window = window_text
            elif score == best_overlap and score > 0.0 and len(window_text) < len(best_window):
                best_window = window_text

    # Among windows within a hair of the best score, keep the shortest (jototuk match tototuk).
    _tie_eps = 0.02
    shortest = best_window
    shortest_len = len(best_window)
    for k in range(1, max_k + 1):
        for i in range(0, len(sentences) - k + 1):
            window_text = " ".join(sentences[i : i + k])
            score = overlap_score(window_text)
            if score >= best_overlap - _tie_eps and len(window_text) < shortest_len:
                shortest = window_text
                shortest_len = len(window_text)
    best_window = shortest

    # Long unpunctuated blobs: coarse sliding windows so we are not stuck on the prefix only.
    if best_overlap < 0.06 and len(body) > max_len * 2:
        step = max(32, max_len // 2)
        upper = min(len(body), 6000)
        for start in range(0, upper, step):
            end = min(len(body), start + max_len * 4)
            chunk = body[start:end]
            score = overlap_score(chunk)
            if score > best_overlap:
                best_overlap = score
                inner_sents = _split_into_sentences(chunk) or [chunk]
                inner_best = inner_sents[0]
                inner_score = overlap_score(inner_best)
                for s in inner_sents:
                    sc = overlap_score(s)
                    if sc > inner_score or (
                        sc == inner_score and len(s) < len(inner_best)
                    ):
                        inner_score = sc
                        inner_best = s
                best_window = _compress_to_matching_word_span(
                    inner_best, answer_tokens, pad_words=1
                )

    excerpt = _compress_to_matching_word_span(best_window, answer_tokens, pad_words=1)
    if len(excerpt) > max_len:
        cut = excerpt[: max_len - 3].rsplit(" ", 1)[0].strip()
        excerpt = (cut + "...") if cut else excerpt[: max_len].rstrip() + "..."

    if best_overlap < 0.03:
        excerpt = body[:max_len] + ("..." if len(body) > max_len else "")
        excerpt = excerpt.rstrip()

    return excerpt, min(1.0, best_overlap)


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
        retrieval_score = float(score_str) if score_str else 0.0
        prefix_excerpt = body[:MAX_EXCERPT_LEN] + ("..." if len(body) > MAX_EXCERPT_LEN else "")
        documents[idx] = {
            "content_id": m.group(4).strip(),
            "type": m.group(2).strip(),
            "title": m.group(3).strip(),
            "body": body,
            "excerpt": prefix_excerpt,
            "match_score": retrieval_score,
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
        body = doc.get("body") or doc.get("excerpt") or ""
        excerpt, overlap = select_excerpt_and_answer_overlap(body, answer_plain, MAX_EXCERPT_LEN)
        retrieval = float(doc.get("match_score", 0.0) or 0.0)
        retrieval = min(1.0, max(0.0, retrieval))
        if overlap > 0.0:
            combined = (
                _RETRIEVAL_SCORE_WEIGHT * retrieval + _ANSWER_OVERLAP_WEIGHT * overlap
            )
        else:
            combined = retrieval
        citations.append(
            Citation(
                content_id=doc["content_id"],
                type=doc["type"],
                excerpt=excerpt,
                match_score=round(min(1.0, max(0.0, combined)), 2),
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