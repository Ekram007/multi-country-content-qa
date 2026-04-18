"""Corpus ingestion into Qdrant vector database."""

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
)

from src.core.settings import settings
from src.core.embeddings import embed_texts, get_embedding_dimension
from src.schema.models import ContentItem
from src.knowledge_base.corpus import load_corpus

logger = logging.getLogger(__name__)


class CorpusIngester:
    """Handles corpus ingestion into Qdrant vector database."""
    
    def __init__(self, corpus_path: Optional[str] = None):
        """Initialize ingester.
        
        Args:
            corpus_path: Path to corpus file (uses settings default if None)
        """
        self.corpus_path = corpus_path or settings.corpus_path
        self.batch_size = 50  # Configurable batch size for upserts
    
    def create_collection(
        self, 
        client: QdrantClient, 
        collection_name: str, 
        vector_size: int,
        recreate: bool = True
    ) -> None:
        """Create Qdrant collection with proper configuration.
        
        Args:
            client: Qdrant client instance
            collection_name: Name of collection to create
            vector_size: Embedding vector dimension
            recreate: Whether to drop existing collection
        """
        try:
            # Check if collection exists
            collections = [c.name for c in client.get_collections().collections]
            
            if collection_name in collections:
                if recreate:
                    logger.info(f"Dropping existing collection: {collection_name}")
                    client.delete_collection(collection_name)
                else:
                    logger.info(f"Collection '{collection_name}' already exists")
                    return
            
            # Create collection with cosine similarity
            logger.info(f"Creating collection '{collection_name}' with vector_size={vector_size}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            
            # Create payload indexes for efficient filtering
            indexes = [
                ("country", PayloadSchemaType.KEYWORD),
                ("language", PayloadSchemaType.KEYWORD),
                ("type", PayloadSchemaType.KEYWORD),
                ("content_id", PayloadSchemaType.KEYWORD),
            ]
            
            for field_name, schema_type in indexes:
                logger.debug(f"Creating payload index: {field_name}")
                client.create_payload_index(collection_name, field_name, schema_type)
            
            logger.info(
                f"Collection '{collection_name}' created successfully with "
                f"{len(indexes)} payload indexes"
            )
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def prepare_embeddings(self, items: List[ContentItem]) -> List[List[float]]:
        """Prepare embeddings for content items.
        
        Args:
            items: List of content items to embed
            
        Returns:
            List of embedding vectors
        """
        if not items:
            raise ValueError("No items to embed")
        
        # Combine title and body for richer embeddings
        texts = [f"{item.title}\n{item.body}" for item in items]
        
        logger.info(f"Embedding {len(texts)} content items...")
        vectors = embed_texts(texts, show_progress=True)
        
        logger.info(f"Generated {len(vectors)} embedding vectors")
        return vectors
    
    def create_points(
        self, 
        items: List[ContentItem], 
        vectors: List[List[float]]
    ) -> List[PointStruct]:
        """Create Qdrant points from content items and vectors.
        
        Args:
            items: List of content items
            vectors: List of embedding vectors
            
        Returns:
            List of PointStruct objects ready for upsert
        """
        if len(items) != len(vectors):
            raise ValueError("Number of items and vectors must match")
        
        points = []
        for item, vector in zip(items, vectors):
            # Generate deterministic UUID from content_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item.content_id))
            
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "content_id": item.content_id,
                    "country": item.country,
                    "language": item.language,
                    "type": item.type,
                    "version": item.version,
                    "title": item.title,
                    "body": item.body,
                    "updated_at": item.updated_at,
                },
            )
            points.append(point)
        
        logger.info(f"Created {len(points)} Qdrant points")
        return points
    
    def upsert_points(
        self, 
        client: QdrantClient, 
        collection_name: str, 
        points: List[PointStruct]
    ) -> int:
        """Upsert points to Qdrant collection in batches.
        
        Args:
            client: Qdrant client instance
            collection_name: Target collection name
            points: List of points to upsert
            
        Returns:
            Total number of points upserted
        """
        if not points:
            raise ValueError("No points to upsert")
        
        total_points = len(points)
        logger.info(f"Upserting {total_points} points in batches of {self.batch_size}")
        
        upserted_count = 0
        for i in range(0, total_points, self.batch_size):
            batch = points[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            try:
                client.upsert(collection_name=collection_name, points=batch)
                upserted_count += len(batch)
                
                logger.info(
                    f"Batch {batch_num}/{(total_points + self.batch_size - 1) // self.batch_size}: "
                    f"Upserted {len(batch)} points ({upserted_count}/{total_points})"
                )
                
            except Exception as e:
                logger.error(f"Failed to upsert batch {batch_num}: {e}")
                raise
        
        return upserted_count
    
    def verify_ingestion(self, client: QdrantClient, collection_name: str) -> dict:
        """Verify ingestion by checking collection status.
        
        Args:
            client: Qdrant client instance
            collection_name: Collection to verify
            
        Returns:
            Dictionary with verification results
        """
        try:
            collection_info = client.get_collection(collection_name)
            count_result = client.count(collection_name)
            
            verification = {
                "collection_exists": True,
                "points_count": count_result.count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status.name,
            }
            
            logger.info(f"Verification completed: {verification}")
            return verification
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"collection_exists": False, "error": str(e)}
    
    def run_ingestion(self, recreate_collection: bool = True) -> dict:
        """Run the complete ingestion process.
        
        Args:
            recreate_collection: Whether to recreate the collection
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Load corpus
            logger.info(f"Loading corpus from {self.corpus_path}")
            corpus_items = load_corpus()
            
            if not corpus_items:
                raise ValueError(f"No items loaded from {self.corpus_path}")
            
            # Convert to ContentItem objects with validation
            validated_items = [ContentItem(**item) for item in corpus_items]
            logger.info(f"Validated {len(validated_items)} content items")
            
            # Prepare embeddings
            vectors = self.prepare_embeddings(validated_items)
            vector_size = get_embedding_dimension()
            
            # Initialize Qdrant client
            client = QdrantClient(
                host=settings.qdrant_host, 
                port=settings.qdrant_port,
                timeout=60.0
            )
            
            # Create collection
            self.create_collection(
                client, 
                settings.qdrant_collection, 
                vector_size,
                recreate=recreate_collection
            )
            
            # Create and upsert points
            points = self.create_points(validated_items, vectors)
            upserted_count = self.upsert_points(client, settings.qdrant_collection, points)
            
            # Verify ingestion
            verification = self.verify_ingestion(client, settings.qdrant_collection)
            
            results = {
                "success": True,
                "corpus_path": str(self.corpus_path),
                "items_loaded": len(validated_items),
                "points_upserted": upserted_count,
                "collection_name": settings.qdrant_collection,
                "vector_size": vector_size,
                "verification": verification,
            }
            
            logger.info("Ingestion completed successfully")
            logger.info(f"Results: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "corpus_path": str(self.corpus_path),
            }


def ingest_corpus(corpus_path: Optional[str] = None, recreate: bool = True) -> dict:
    """Main function to ingest corpus into Qdrant.
    
    Args:
        corpus_path: Path to corpus JSONL file
        recreate: Whether to recreate the collection
        
    Returns:
        Dictionary with ingestion results
    """
    ingester = CorpusIngester(corpus_path)
    return ingester.run_ingestion(recreate_collection=recreate)


if __name__ == "__main__":
    # Setup logging for CLI usage
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    
    try:
        results = ingest_corpus()
        
        if results["success"]:
            print(f"\n✅ Ingestion completed successfully!")
            print(f"   Items loaded: {results['items_loaded']}")
            print(f"   Points upserted: {results['points_upserted']}")
            print(f"   Collection: {results['collection_name']}")
            print(f"   Final count: {results['verification'].get('points_count', 'unknown')}")
        else:
            print(f"\n❌ Ingestion failed: {results['error']}")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Ingestion interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        exit(1)