# Submission Notes

## What I Built

A complete multi-country content Q&A system that meets all core requirements:

### ✅ Core Functionality
- **End-to-end pipeline**: Corpus ingestion → vector storage → tool-based retrieval → LLM synthesis → structured citations in the API response
- **Multi-tenant isolation**: Zero cross-country data leakage (verified by evaluation tests)
- **LangGraph agent**: Tool-calling graph (`model` ↔ `ToolNode`) with `search_content` / `get_supported_countries`; the LLM decides when to retrieve
- **FastAPI endpoint**: `/ask` endpoint matching the interview brief
- **Citation fidelity**: Citations are derived from the last `search_content` tool payload; excerpts are taken from retrieved bodies and **match scores** reflect Qdrant similarity scores returned at retrieval time

### ✅ Technical Implementation
- **44 content items** across 4 countries and 4 languages successfully ingested
- **Qdrant vector database** with metadata indexing on country, language, type, content_id
- **Sentence transformers** for local embeddings (384-dim, no API dependency)
- **Multi-provider LLM support**: Google Gemini, OpenAI, OpenRouter via LangChain
- **Docker containerization** with docker-compose orchestration

### ✅ Quality Assurance  
- **Unit tests** covering metadata filtering, request validation (Pydantic), and API citation parsing from tool output
- **10-question evaluation harness** testing multi-tenant isolation, language fallback (via `search_content` language loop), and citation alignment
- **Language fallback**: Implemented inside `search_content` by trying supported languages for the country until chunks are returned; the model answers in the requested language when possible

## What I Chose to Skip and Why

1. **Caching layers**: Focused on core functionality over optimization
2. **Authentication**: Out of scope per interview brief section 11
3. **Advanced embeddings**: Used reliable sentence-transformers instead of experimental multilingual models  
4. **UI development**: API-first approach as specified
5. **Extensive error recovery**: Basic graceful degradation for LLM rate limits

## What I'm Most Proud Of

1. **Multi-tenant correctness**: Metadata filtering in `retrieve()` ensures isolation. Country A return policy queries never see Country B's 7-day window, even when semantically similar.

2. **LangGraph + tools**: A small graph (`model` → `tools` → `model` …) keeps retrieval explicit and testable without a large custom state machine.

3. **Traceable citations**: The API parses the formatted `search_content` output so citations align with what was actually retrieved, including similarity scores from the vector store.

4. **Production thinking**: Error handling for rate limits, Docker setup, evaluation harness, and clear documentation.

## What I Would Do Differently Given More Time

### Immediate Improvements (1 day)
- **Hybrid retrieval**: Combine semantic search with keyword matching for better recall
- **Richer excerpt selection**: Optional semantic ranking of sentences within a chunk (beyond truncation)
- **Request batching**: Queue requests to handle LLM rate limits gracefully
- **Health monitoring**: Add /metrics endpoint for retrieval performance and LLM latency

### Longer-term (1 week)
- **Advanced embeddings**: Migrate to multilingual models (Cohere, OpenAI) for better cross-language performance
- **Query optimization**: Add query expansion, intent classification, and confidence scoring
- **Conversation memory**: Extend to multi-turn conversations with context preservation
- **Advanced testing**: Property-based testing, load testing, and A/B testing framework

## Technical Gotchas for Reviewers

### Prerequisites
- **Python 3.10+** required (as specified in interview brief)
- **Qdrant must be running** on localhost:6333 before setup
- **LLM API key required** - add to `.env` file (Google Gemini configured by default)

### Single Setup Command (as requested in brief)
```bash
# 1. Configure environment
cp .env.example .env
# Add your GOOGLE_API_KEY to .env

# 2. Start Qdrant  
docker run -d -p 6333:6333 qdrant/qdrant

# 3. Run everything (single command as specified in Section 3.2)
./setup.sh
```

### Screenshots Note
The brief asks for PNG screenshots but this is a CLI environment. I've provided equivalent text evidence in `screenshots/` that demonstrates all required functionality. See `screenshots/README.md` for mapping between requested PNGs and provided text files.

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

### LangGraph Tool Agent vs Hand-Written DAG
- **Pro**: Standard `bind_tools` + `ToolNode` pattern; easy to extend with more tools
- **Pro**: Retrieval and fallback live in `search_content`, so the graph stays small
- **Con**: Behavior depends on LLM tool-calling quality (mitigated via system prompt)

### Local Embeddings vs API
- **Pro**: No external dependency, faster iteration during development
- **Pro**: Cost predictable, no rate limits
- **Con**: Lower quality for cross-lingual queries vs commercial multilingual models

## Evaluation Results Summary

- **Multi-tenant isolation**: ✅ Perfect (0 cross-country leakage detected)
- **Language support**: ✅ English, Spanish, Hindi, French (Canadian) all represented in corpus and retrieval paths
- **Citation alignment**: ✅ Citations map to parsed retrieval blocks; scores from Qdrant
- **Fallback behavior**: ✅ `search_content` walks supported languages per country when the first choice returns nothing
- **Error handling**: ✅ Invalid countries rejected with **422** (Pydantic / FastAPI validation on `AskRequest`)
- **Rate limit resilience**: ✅ Graceful degradation when LLM quota exceeded

The system successfully demonstrates all core requirements from the interview brief.
