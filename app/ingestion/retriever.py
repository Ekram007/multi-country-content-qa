import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config.settings import get_settings
from app.ingestion.embeddings import embed_query

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def build_metadata_filter(country: str, language: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="country", match=MatchValue(value=country)),
            FieldCondition(key="language", match=MatchValue(value=language)),
        ]
    )


def retrieve(
    query: str,
    country: str,
    language: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve content chunks filtered by country and language DURING similarity search."""
    settings = get_settings()
    client = get_qdrant_client()
    query_vector = embed_query(query)

    metadata_filter = build_metadata_filter(country, language)

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=metadata_filter,
        limit=top_k,
        with_payload=True,
    )

    chunks = []
    for point in results.points:
        chunks.append({
            "content_id": point.payload["content_id"],
            "country": point.payload["country"],
            "language": point.payload["language"],
            "type": point.payload["type"],
            "version": point.payload["version"],
            "title": point.payload["title"],
            "body": point.payload["body"],
            "updated_at": point.payload["updated_at"],
            "score": point.score,
        })

    logger.info(
        f"Retrieved {len(chunks)} chunks for query in country={country}, "
        f"language={language} (top_k={top_k})"
    )
    return chunks
