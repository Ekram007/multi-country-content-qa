# Multi-Country Content Q&A with Citations

A natural language Q&A system that retrieves country-scoped content and provides grounded answers with verifiable citations. Built with LangGraph, FastAPI, Qdrant, and sentence transformers.

## Problem Statement

This system solves the challenge of serving contextualized customer support answers across multiple countries and languages while maintaining strict data isolation. Given a natural language question, country, and language, it retrieves relevant content from that specific country's knowledge base and generates a grounded answer with citations pointing to the exact source content, ensuring no cross-country information leakage.

**Built according to the technical specification in `[AI-Interview.txt](./AI-Interview.txt)`.**

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

### Docker Setup

```bash
# Start everything with Docker Compose
cp .env.example .env

docker compose up -d
```

The `app` container runs **corpus ingestion** (`src.run_ingest`) once on startup, then starts the API, so Qdrant is filled automatically. Optional environment variables (set in `docker-compose.yml` or override as needed):

- `RUN_INGEST_ON_START` — set to `false` to skip ingestion and only run the API.
- `INGEST_RECREATE` — set to `true` to drop and recreate the Qdrant collection on each start; `false` (default in Compose) only upserts, which is faster on restarts.

### Individual Commands

```bash
# Install dependencies
uv sync

# Run corpus ingestion (not needed before first Docker run if you use Compose; still used for local dev)
uv run python -m src.run_ingest

# Run API service  
uv run python -m src.run_service

```

## Example Usage

### Basic Request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is your return policy?",
    "country": "B", 
    "language": "en"
  }'
```

### Expected Response

```json
{
  "answer": "You may return any item within 7 days of delivery for a full refund [1]. ...",
  "language_used": "en",
  "citations": [
    {
      "content_id": "b_faq_returns_en",
      "type": "FAQ",
      "excerpt": "You may return any item within 7 days of delivery for a full refund...",
      "match_score": 0.87
    }
  ],
  "trace": {
    "retrieval_count": 1,
    "latency_ms": 2334,
    "model": "gemini-2.5-flash"
  }
}
```

Values vary by run (`retrieval_count` follows retrieved chunks, typically `top_k=5`). `match_score` blends Qdrant similarity with answer–source overlap when building citations (see `ARCHITECTURE.md`).

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


## Known Limitations

- **LLM Rate Limits**: Free-tier APIs (Gemini 5 req/min) may cause 429 errors during high-volume evaluation
- **Embedding Quality**: Uses sentence-transformers locally; commercial embeddings would improve cross-lingual retrieval  
- **No Conversation Memory**: Each request is stateless
- **Fixed top-K**: Default `top_k=5` retrieval (not adaptive); tune via tool/retriever if needed
- **No Authentication**: API is open; production needs authN/authZ

## What I would do next with more time

This is the **project backlog** for continued work. Things I’d tackle first:

### Multi-tenant safety (country / language)

- Remove `country` from LLM tool-call parameters where the model could pass the wrong value; inject `country` (and `language` where appropriate) from the HTTP request / graph state inside the tool so retrieval always uses the caller’s scope.
- Apply the same pattern to **language** and fallback: filtering and same-country locale fallback should follow request + explicit policy, not model-supplied tool arguments.

### Retrieval (beyond `top_k=5`)

- **Dynamic k**: replace top-k with bounds and heuristics (score gaps, query type, min/max caps).
- **Thresholds**: drop weak similarity matches; optionally fetch more candidates when all scores are low (with a safety ceiling).
- **Reranking**: cross-encoder or lightweight reranker on the vector shortlist.
- **Hybrid search**: combine dense vectors with lexical / **fuzzy** signals on chunk excerpts and/or titles (RRF or weighted fusion) so exact keywords and typo-tolerant matches help when embedding similarity alone is ambiguous.
- **Embeddings**: optional upgrade to stronger cross-lingual models (e.g. Cohere Multilingual, OpenAI); query expansion or rewriting where it helps.

### Ingestion, citations, quality

- Revisit **chunking** (size, overlap, headings) and per-chunk metadata (`content_id`, `type`, `version`) for sharper citations.
- **Grounding**: verify cited spans appear in retrieved bodies; trim or reject hallucinated excerpts; optional “unverified” flags in trace for debugging.
- Expand **evaluation** with adversarial cases (wrong country/language, contradictory cross-country FAQs); track retrieval hit rate and citation fidelity over time.

### Observability, performance, product

- **Observability**: structured logs/traces for retrieval k, threshold cuts, rerank deltas, fusion weights, per-stage latency.
- **Performance / cost**: cache embeddings for hot queries; batch reranking where possible; cap LLM context to the smallest sufficient retrieved set; vector index tuning.
- **Production hardening**: authentication, rate limiting, validation middleware, retries, metrics, health checks (see interview “out of scope” for the exercise itself).
- **Advanced (later)**: multi-turn memory, intent classification, confidence scores, A/B tests for retrieval strategies, richer citation UX (highlighting, deep links).

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

