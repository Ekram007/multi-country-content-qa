# Multi-Country Content Q&A with Citations

A natural language Q&A system that retrieves country-scoped content and provides grounded answers with verifiable citations. Built with LangGraph, FastAPI, Qdrant, and sentence transformers.

## Problem Statement

This system solves the challenge of serving contextualized customer support answers across multiple countries and languages while maintaining strict data isolation. Given a natural language question, country, and language, it retrieves relevant content from that specific country's knowledge base and generates a grounded answer with citations pointing to the exact source content, ensuring no cross-country information leakage.

## Quick Start

### Prerequisites
- Python 3.11+
- Qdrant running on localhost:6333 (or Docker)
- LLM API key (Google Gemini, OpenAI, or OpenRouter)

### Single Setup Command

```bash
# 1. Clone and configure
git clone <repository-url>
cd multi-country-content-qa
cp .env.example .env
# Edit .env and add your LLM API key

# 2. Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 3. Run everything (ingest + serve)
./setup.sh
```

### Manual Setup (Alternative)

```bash
# Install dependencies
uv sync

# Ingest content corpus
uv run python -m app.ingestion.ingest

# Start API server
uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

### Docker Setup (Alternative)

```bash
# Start everything with Docker Compose
docker-compose up -d
```

## Example Usage

### Basic Request
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is your return policy?",
    "country": "B", 
    "language": "es"
  }'
```

### Expected Response
```json
{
  "answer": "Puede devolver cualquier artículo dentro de los 7 días posteriores a la entrega para un reembolso completo [1]. Los artículos defectuosos pueden devolverse dentro de los 30 días [1].",
  "language_used": "es",
  "citations": [
    {
      "content_id": "b_faq_returns_es",
      "type": "FAQ",
      "title": "¿Cuál es su política de devoluciones?",
      "excerpt": "Puede devolver cualquier artículo dentro de los 7 días posteriores a la entrega para un reembolso completo, sin preguntas.",
      "match_score": 0.87
    }
  ],
  "trace": {
    "retrieval_count": 2,
    "latency_ms": 2334,
    "model": "gemini-2.5-flash",
    "fallback_used": false,
    "fallback_language": null
  }
}
```

## Multi-Tenant Isolation

The system enforces strict country-level data isolation:

- **Country A** (English + Hindi): 48-hour return window
- **Country B** (English + Spanish): 7-day return window  
- **Country C** (English + French): 14-day return window
- **Country D** (English only): 30-day return window

Queries for return policies from different countries will return different, country-specific answers with zero cross-contamination.

## Language Fallback

When content doesn't exist in the requested language for a country, the system falls back to available languages in the same country and translates the answer, maintaining country isolation.

## Testing

```bash
# Run unit tests
uv run pytest tests/ -v

# Run evaluation harness
uv run python evaluate.py

# Run with custom spacing (for rate-limited LLM providers)
EVAL_SLEEP_SECONDS=20 uv run python evaluate.py
```

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   FastAPI   │───▶│  LangGraph  │───▶│   Qdrant    │
│   Server    │    │   Agent     │    │  VectorDB   │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       │                   ▼                   │
       │            ┌─────────────┐           │
       │            │ Retrieval + │           │
       │            │ Synthesis   │           │
       │            │ + Citations │           │
       │            └─────────────┘           │
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    LLM      │    │ Embeddings  │    │  Metadata   │
│ (Gemini/GPT)│    │(Sentence-T) │    │  Filtering  │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Key Components

1. **LangGraph Agent**: 6-node state machine with conditional routing:
   - validate_input → retrieve → [fallback_retrieve] → synthesize → extract_citations
   
2. **Qdrant Integration**: Vector similarity search with metadata pre-filtering on country/language
   
3. **Multi-tenant Isolation**: Enforced at query time via Qdrant filters, not post-processing

4. **Citation Verification**: Fuzzy matching between generated answer excerpts and source content

## Performance Characteristics

- **Startup Time**: ~10-15 seconds (pre-loads embedding model for fast responses)
- **First Request**: Fast (~2-3 seconds after startup)
- **Subsequent Requests**: Very fast (~1-2 seconds)
- **Throughput**: Limited by LLM provider rate limits, not system architecture

## Known Limitations

- **LLM Rate Limits**: Free-tier APIs (Gemini 5 req/min) may cause 429 errors during high-volume evaluation
- **Embedding Quality**: Uses sentence-transformers locally; commercial embeddings would improve cross-lingual retrieval  
- **No Conversation Memory**: Each request is stateless
- **Basic Citation Matching**: Uses SequenceMatcher; production would benefit from semantic similarity
- **No Authentication**: API is open; production needs authN/authZ

## What I Would Do Next With More Time

1. **Production Hardening**:
   - Add request authentication and rate limiting
   - Implement request/response validation middleware
   - Add comprehensive error handling and retry logic
   - Set up proper logging, metrics, and health checks

2. **Improved Retrieval**:
   - Hybrid search (keyword + semantic)
   - Query expansion and rewriting
   - Better cross-lingual embeddings (Cohere Multilingual, OpenAI)
   - Chunk-level metadata for finer filtering

3. **Enhanced Citations**:
   - Semantic similarity for citation verification  
   - Source document highlighting and deep-linking
   - Multi-source answer synthesis with source attribution

4. **Performance Optimization**:
   - Vector index optimization and caching
   - LLM response caching for common queries
   - Async processing for batch requests
   - Connection pooling and request queuing

5. **Advanced Features**:
   - Multi-turn conversation with context
   - Query intent classification  
   - Confidence scoring and uncertainty handling
   - A/B testing framework for different retrieval strategies

## Project Structure

```
├── app/
│   ├── agent/           # LangGraph agent implementation
│   │   ├── graph.py     # State graph definition
│   │   ├── nodes.py     # Graph node functions  
│   │   ├── state.py     # State schema
│   │   └── llm.py       # LLM provider abstraction
│   ├── api/             # FastAPI application
│   │   └── main.py      # HTTP server and /ask endpoint
│   ├── config/          # Configuration management
│   │   └── settings.py  # Pydantic settings
│   ├── ingestion/       # Data ingestion pipeline
│   │   ├── ingest.py    # Corpus loading and embedding
│   │   ├── embeddings.py # Sentence transformer wrapper
│   │   └── retriever.py # Qdrant query interface
│   └── models/          # Data schemas
│       └── schemas.py   # Pydantic models
├── tests/               # Unit tests
├── data/               # Corpus data
├── screenshots/        # Visual evidence
├── evaluate.py         # Evaluation harness  
└── docker-compose.yml  # Docker orchestration
```

## Dependencies

- **LangGraph**: Agent orchestration and state management
- **FastAPI**: HTTP API framework  
- **Qdrant**: Vector database with metadata filtering
- **sentence-transformers**: Local embeddings (all-MiniLM-L6-v2)
- **LangChain**: LLM provider integrations
- **Pydantic**: Data validation and settings management