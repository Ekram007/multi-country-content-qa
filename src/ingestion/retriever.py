"""Content retrieval from Qdrant vector database."""

import logging
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.settings import settings
from src.core.embeddings import embed_query

logger = logging.getLogger(__name__)

# Global client instance for connection reuse
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client instance (singleton pattern).
    
    Returns:
        QdrantClient: Configured Qdrant client
        
    Raises:
        ConnectionError: If unable to connect to Qdrant
    """
    global _qdrant_client
    
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(
                host=settings.qdrant_host, 
                port=settings.qdrant_port,
                timeout=30.0
            )
            
            # Test connection
            _qdrant_client.get_collections()
            logger.info(f"Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Cannot connect to Qdrant database: {e}")
    
    return _qdrant_client


def build_metadata_filter(country: str, language: str, content_type: Optional[str] = None) -> Filter:
    """Build Qdrant filter for metadata matching.
    
    Args:
        country: Country code filter
        language: Language code filter
        content_type: Optional content type filter
        
    Returns:
        Qdrant Filter object
    """
    conditions = [
        FieldCondition(key="country", match=MatchValue(value=country.upper())),
        FieldCondition(key="language", match=MatchValue(value=language)),  # Preserve original case
    ]
    
    if content_type:
        conditions.append(
            FieldCondition(key="type", match=MatchValue(value=content_type))
        )
    
    return Filter(must=conditions)


def retrieve(
    query: str,
    country: str,
    language: str,
    top_k: int = 5,
    content_type: Optional[str] = None,
    min_score: float = 0.0
) -> List[Dict]:
    """Retrieve content chunks using vector similarity with metadata filtering.
    
    Args:
        query: Search query text
        country: Country code for filtering
        language: Language code for filtering
        top_k: Maximum number of results to return
        content_type: Optional content type filter
        min_score: Minimum similarity score threshold
        
    Returns:
        List of content chunks with metadata and scores
        
    Raises:
        ConnectionError: If Qdrant is unreachable
        ValueError: If query parameters are invalid
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    
    try:
        client = get_qdrant_client()
        
        # Embed the query
        logger.debug(f"Embedding query: '{query[:50]}...'")
        query_vector = embed_query(query)
        
        # Build metadata filter
        metadata_filter = build_metadata_filter(country.upper(), language, content_type)
        
        # Search vector database
        logger.debug(
            f"Searching Qdrant: country={country}, language={language}, "
            f"type={content_type}, top_k={top_k}"
        )
        
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            query_filter=metadata_filter,
            limit=top_k,
            with_payload=True,
            score_threshold=min_score
        )
        
        # Process results
        chunks = []
        for point in results.points:
            chunk = {
                "content_id": point.payload["content_id"],
                "country": point.payload["country"],
                "language": point.payload["language"],
                "type": point.payload["type"],
                "version": point.payload["version"],
                "title": point.payload["title"],
                "body": point.payload["body"],
                "updated_at": point.payload["updated_at"],
                "score": float(point.score) if point.score else 0.0,
            }
            chunks.append(chunk)
        
        logger.info(
            f"Retrieved {len(chunks)} chunks for query in country={country}, "
            f"language={language} (requested top_k={top_k})"
        )
        
        return chunks
        
    except UnexpectedResponse as e:
        logger.error(f"Qdrant query failed: {e}")
        raise ConnectionError(f"Vector database query failed: {e}")
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise


def get_collection_info() -> Dict:
    """Get information about the Qdrant collection.
    
    Returns:
        Dictionary with collection information
        
    Raises:
        ConnectionError: If Qdrant is unreachable
    """
    try:
        client = get_qdrant_client()
        
        collection_info = client.get_collection(settings.qdrant_collection)
        
        return {
            "name": settings.qdrant_collection,
            "status": collection_info.status,
            "vectors_count": collection_info.vectors_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "points_count": collection_info.points_count,
            "config": {
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.name,
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise ConnectionError(f"Cannot access collection information: {e}")


def test_connection() -> bool:
    """Test Qdrant connection and collection availability.
    
    Returns:
        True if connection and collection are healthy
    """
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        
        collection_names = [col.name for col in collections.collections]
        
        if settings.qdrant_collection not in collection_names:
            logger.error(f"Collection '{settings.qdrant_collection}' not found")
            return False
            
        logger.info("Qdrant connection test passed")
        return True
        
    except Exception as e:
        logger.error(f"Qdrant connection test failed: {e}")
        return False


def reset_client() -> None:
    """Reset the Qdrant client (useful for testing or reconnection)."""
    global _qdrant_client
    if _qdrant_client:
        try:
            _qdrant_client.close()
        except:
            pass
    _qdrant_client = None