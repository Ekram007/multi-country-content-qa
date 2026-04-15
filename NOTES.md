# Submission Notes

## What I Built

A complete multi-country content Q&A system that meets all core requirements:

### ✅ Core Functionality
- **End-to-end pipeline**: Corpus ingestion → vector storage → retrieval → LLM synthesis → citation extraction
- **Multi-tenant isolation**: Zero cross-country data leakage (verified by evaluation tests)
- **LangGraph agent**: 6-node state machine with conditional routing and fallback logic
- **FastAPI endpoint**: `/ask` endpoint matching exact specification from interview brief
- **Citation fidelity**: Generated excerpts are verified against source content with match scores

### ✅ Technical Implementation
- **44 content items** across 4 countries and 4 languages successfully ingested
- **Qdrant vector database** with metadata indexing on country, language, type, content_id
- **Sentence transformers** for local embeddings (384-dim, no API dependency)
- **Multi-provider LLM support**: Google Gemini, OpenAI, OpenRouter via LangChain
- **Docker containerization** with docker-compose orchestration

### ✅ Quality Assurance  
- **9 unit tests** covering metadata filtering, citation extraction, and input validation (all passing)
- **10-question evaluation harness** testing multi-tenant isolation, language fallback, and citation accuracy
- **Language fallback system**: Spanish queries on Country A fall back to English content with translation

## What I Chose to Skip and Why

1. **Caching layers**: Focused on core functionality over optimization
2. **Authentication**: Out of scope per interview brief section 11
3. **Advanced embeddings**: Used reliable sentence-transformers instead of experimental multilingual models  
4. **UI development**: API-first approach as specified
5. **Extensive error recovery**: Basic graceful degradation for LLM rate limits

## What I'm Most Proud Of

1. **Multi-tenant correctness**: The metadata filtering architecture ensures perfect isolation. Country A return policy queries never see Country B's 7-day window, even when semantically similar.

2. **LangGraph design**: Clean separation of concerns with explicit state transitions:
   ```
   validate → retrieve → [fallback if needed] → synthesize → extract_citations
   ```

3. **Citation verification**: The system doesn't just generate citations as decoration—it extracts relevant excerpts and computes match scores for verification.

4. **Production thinking**: Error handling for rate limits, Docker setup, comprehensive evaluation, and clear documentation.

## What I Would Do Differently Given More Time

### Immediate Improvements (1 day)
- **Hybrid retrieval**: Combine semantic search with keyword matching for better recall
- **Better citation extraction**: Use semantic similarity instead of string matching for excerpt verification  
- **Request batching**: Queue requests to handle LLM rate limits gracefully
- **Health monitoring**: Add /metrics endpoint for retrieval performance and LLM latency

### Longer-term (1 week)
- **Advanced embeddings**: Migrate to multilingual models (Cohere, OpenAI) for better cross-language performance
- **Query optimization**: Add query expansion, intent classification, and confidence scoring
- **Conversation memory**: Extend to multi-turn conversations with context preservation
- **Advanced testing**: Property-based testing, load testing, and A/B testing framework

## Technical Gotchas for Reviewers

### Prerequisites
- **Python 3.11+** required (uses `str | None` union syntax)
- **Qdrant must be running** on localhost:6333 before ingestion/serving
- **LLM API key required** - add to `.env` file (Google Gemini configured by default)

### Setup Sequence
```bash
# 1. Install dependencies
uv sync

# 2. Configure environment  
cp .env.example .env
# Add your GOOGLE_API_KEY to .env

# 3. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 4. Ingest corpus (one-time)
uv run python -m app.ingestion.ingest

# 5. Start server
uv run uvicorn app.api.main:app --port 8000
```

### Rate Limit Handling
- **Gemini Free Tier**: 5 requests/minute per model
- **Evaluation spacing**: Set `EVAL_SLEEP_SECONDS=20` for conservative testing
- **Error graceful**: System returns informative error messages instead of 500s when LLM fails

### Multi-tenant Verification
```bash
# Test isolation - these should return different content IDs:
curl -X POST localhost:8000/ask -d '{"question":"return policy","country":"A","language":"en"}'
curl -X POST localhost:8000/ask -d '{"question":"return policy","country":"B","language":"en"}'
```

## Architecture Decisions

### Vector Database Choice: Qdrant
- **Pro**: Excellent metadata filtering at query time (not post-processing)
- **Pro**: Easy local deployment, good Python client
- **Con**: Requires separate service (vs. embedded FAISS)

### LangGraph vs Simple Pipeline
- **Pro**: Explicit state machine makes fallback logic testable and debuggable
- **Pro**: Easy to add new nodes (e.g., query classification, confidence scoring)  
- **Con**: More complexity than linear pipeline for simple use case

### Local Embeddings vs API
- **Pro**: No external dependency, faster iteration during development
- **Pro**: Cost predictable, no rate limits
- **Con**: Lower quality for cross-lingual queries vs commercial multilingual models

## Evaluation Results Summary

- **Multi-tenant isolation**: ✅ Perfect (0 cross-country leakage detected)
- **Language support**: ✅ English, Spanish, Hindi, French all working
- **Citation accuracy**: ✅ Generated excerpts match source content with high fidelity
- **Fallback behavior**: ✅ Spanish queries on Country A fallback to English content + translation
- **Error handling**: ✅ Invalid countries rejected with 422 status
- **Rate limit resilience**: ✅ Graceful degradation when LLM quota exceeded

The system successfully demonstrates all core requirements from the interview brief.