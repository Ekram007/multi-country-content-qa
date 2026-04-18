# Multi-Country Content Q&A with Citations

A natural language Q&A system that retrieves country-scoped content and provides grounded answers with verifiable citations. Built with LangGraph, FastAPI, Qdrant, and sentence transformers.

## Problem Statement

This system solves the challenge of serving contextualized customer support answers across multiple countries and languages while maintaining strict data isolation. Given a natural language question, country, and language, it retrieves relevant content from that specific country's knowledge base and generates a grounded answer with citations pointing to the exact source content, ensuring no cross-country information leakage.

**Built according to the technical specification in [`AI-Interview.txt`](./AI-Interview.txt).**

## Quick Start

### Prerequisites
- Python 3.10+
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

### Individual Commands

```bash
# Install dependencies
uv sync

# Run corpus ingestion
uv run python -m src.run_ingest

# Run API service  
uv run python -m src.run_service

# Interactive agent CLI
uv run python -m src.run_agent

# Direct agent query (CLI)
uv run python -m src.run_agent ask "What is your return policy?" A en

# Health check (CLI)
uv run python -m src.run_agent health
```

### Docker Setup (Alternative)

```bash
# Start everything with Docker Compose
docker-compose up -d

# Or build individual containers
docker build -f docker/Dockerfile.app -t content-qa:app .
docker build -f docker/Dockerfile.service -t content-qa:service .
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

Shape matches [`AI-Interview.txt`](./AI-Interview.txt) (`content_id`, `type`, `excerpt`, `match_score` — no `title` on citations; `trace` has only `retrieval_count`, `latency_ms`, `model`).

```json
{
  "answer": "You may return any item within 7 days of delivery for a full refund [1]. ...",
  "language_used": "es",
  "citations": [
    {
      "content_id": "b_faq_returns_es",
      "type": "FAQ",
      "excerpt": "Puede devolver cualquier artículo dentro de los 7 días...",
      "match_score": 0.87
    }
  ],
  "trace": {
    "retrieval_count": 5,
    "latency_ms": 2334,
    "model": "gemini-2.5-flash"
  }
}
```

Values vary by run (`retrieval_count` follows retrieved chunks, typically `top_k=5`; `match_score` comes from Qdrant similarity).

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
uv run python scripts/evaluate.py

# Run with custom spacing (for rate-limited LLM providers)
EVAL_SLEEP_SECONDS=20 uv run python scripts/evaluate.py
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

1. **LangGraph Agent**: Tool-calling graph (`model` ↔ `tools`) — LLM invokes `search_content` (RAG) then answers; explicit nodes and edges in `src/agents/content_qa/content_qa_agent.py`.

2. **Qdrant Integration**: Vector similarity search with metadata pre-filtering on country/language **before** ranking (no post-filter leak).

3. **Multi-tenant Isolation**: Enforced at query time via Qdrant filters, not post-processing.

4. **Citations API**: Built from retrieved chunks (parsed from tool output); `match_score` reflects retrieval similarity; excerpts are sourced from stored content bodies.

## Performance Characteristics

- **Startup Time**: ~7-10 seconds (pre-loads embedding model with reduced logging)
- **Device Detection**: Auto-detects best available (MPS/Apple GPU → CUDA/NVIDIA → CPU)
- **First Request**: Fast (~2-3 seconds after startup)
- **Subsequent Requests**: Very fast (~1-2 seconds)
- **Throughput**: Limited by LLM provider rate limits, not system architecture

## Known Limitations

- **LLM Rate Limits**: Free-tier APIs (Gemini 5 req/min) may cause 429 errors during high-volume evaluation
- **Embedding Quality**: Uses sentence-transformers locally; commercial embeddings would improve cross-lingual retrieval  
- **No Conversation Memory**: Each request is stateless
- **Fixed top-K**: Default `top_k=5` retrieval (not adaptive); tune via tool/retriever if needed
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
├── scripts/             # 📜 Utility scripts
│   └── evaluate.py     # Evaluation harness (10 test cases)
├── src/
│   ├── run_service.py   # 🚀 Service entry
│   ├── run_ingest.py    # 📁 Ingestion runner 
│   ├── run_agent.py     # 🤖 Interactive agent CLI
│   ├── core/            # Core utilities and configuration
│   │   ├── settings.py  # Application settings
│   │   ├── llm.py       # LLM provider abstraction
│   │   └── embeddings.py# Embedding model wrapper
│   ├── agents/          # Agent implementations
│   │   └── content_qa/  # Multi-country Q&A agent
│   │       ├── content_qa_agent.py  # LangGraph compile (model + tools)
│   │       └── tools.py             # @tool search_content, get_supported_countries
│   ├── schema/          # Data models and schemas
│   │   └── models.py    # Pydantic models
│   ├── service/         # HTTP API service layer
│   │   ├── api.py       # FastAPI application
│   │   └── utils.py     # Service utilities
│   ├── ingestion/       # Data ingestion pipeline
│   │   ├── ingest.py    # Corpus ingestion
│   │   └── retriever.py # Vector search interface
│   ├── prompts/         # LLM prompt templates (text files only)
│   │   └── content_qa_agent_system_prompt.txt
│   └── knowledge_base/  # Knowledge base management
│       └── corpus.py    # Corpus loading utilities
├── tests/               # 🧪 Test suite
│   ├── core/           # Core functionality tests
│   │   ├── test_filtering.py   # Metadata filtering tests
│   │   └── test_citations.py   # Citation extraction tests
│   ├── agents/         # Agent-specific tests
│   │   └── test_validation.py  # Input validation tests
│   ├── integration/    # Integration tests (future use)
│   └── conftest.py     # Shared test fixtures
├── docker/              # Docker configuration
│   ├── Dockerfile.app      # Application container
│   └── Dockerfile.service  # Service container
├── data/                  # Corpus data (`corpus.jsonl`)
├── screenshots/           # Visual evidence (see `AI-Interview.txt` for expected PNGs)
├── AI-Interview.txt       # Original technical specification
├── setup.sh               # Ingest + serve (expects Qdrant already reachable)
└── docker-compose.yml     # Docker orchestration
```

## Dependencies

- **LangGraph**: Agent orchestration and state management
- **FastAPI**: HTTP API framework  
- **Qdrant**: Vector database with metadata filtering
- **sentence-transformers**: Local embeddings (all-MiniLM-L6-v2)
- **LangChain**: LLM provider integrations
- **Pydantic**: Data validation and settings management