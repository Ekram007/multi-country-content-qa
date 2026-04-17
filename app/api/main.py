import time
import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.config.settings import get_settings
from app.models.schemas import AskRequest, AskResponse, Citation, Trace
from app.agent.graph import get_compiled_graph
from app.ingestion.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logging.basicConfig(
        level=getattr(logging, get_settings().log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    # Reduce verbosity from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)  # Suppress AFC messages
    logger.info("Starting Content Q&A service...")
    
    # Pre-load embedding model for faster response times  
    logger.info("Loading embedding model...")
    embedding_model = get_embedding_model()
    logger.info(f"Embedding model loaded: {embedding_model.get_embedding_dimension()} dimensions")
    
    # Warmup embedding model with a test query
    logger.info("Warming up embedding model...")
    test_embedding = embedding_model.encode("test query", normalize_embeddings=True)
    logger.info(f"Embedding model warmed up successfully")
    
    # Pre-load LLM for faster first response
    logger.info("Loading LLM...")
    from app.agent.llm import get_llm
    llm = get_llm()
    logger.info("LLM loaded and ready")
    
    # Compile LangGraph agent
    _graph = get_compiled_graph()
    logger.info("LangGraph agent compiled and ready.")
    
    yield
    logger.info("Shutting down Content Q&A service.")


app = FastAPI(
    title="Multi-Country Content Q&A",
    description="Natural language Q&A over multi-country content with citations",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    start_time = time.time()
    settings = get_settings()

    initial_state = {
        "question": request.question,
        "country": request.country.upper(),
        "language": request.language,
    }

    result = _graph.invoke(initial_state)

    latency_ms = int((time.time() - start_time) * 1000)

    citations = [
        Citation(
            content_id=c["content_id"],
            type=c["type"],
            excerpt=c["excerpt"],
            match_score=c["match_score"],
        )
        for c in result.get("citations", [])
    ]

    response = AskResponse(
        answer=result.get("answer", "Unable to generate an answer."),
        language_used=request.language,
        citations=citations,
        trace=Trace(
            retrieval_count=len(result.get("retrieved_chunks", [])),
            latency_ms=latency_ms,
            model=settings.llm_model,
        ),
    )

    logger.info(
        f"Query: country={request.country}, lang={request.language}, "
        f"citations={len(citations)}, latency={latency_ms}ms"
    )
    return response
