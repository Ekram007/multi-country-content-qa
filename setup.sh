#!/bin/bash
set -e

echo "🚀 Multi-Country Content Q&A Setup"
echo "=================================="

# Check if Qdrant is running
if ! curl -s http://localhost:6333/health > /dev/null; then
    echo "❌ Qdrant not running on localhost:6333"
    echo "Please start Qdrant first:"
    echo "  docker run -d -p 6333:6333 qdrant/qdrant"
    exit 1
fi

echo "✅ Qdrant is running"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
fi

echo "📦 Installing dependencies..."
uv sync

echo "📁 Ingesting corpus (44 items)..."
uv run python -m src.run_ingest

echo "🔥 Starting API server..."
echo "API will be available at: http://localhost:8000"
echo "Test endpoint: curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' -d '{\"question\":\"What is your return policy?\",\"country\":\"A\",\"language\":\"en\"}'"
echo ""

uv run python -m src.run_service