import json
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
)

from app.config.settings import get_settings
from app.ingestion.embeddings import embed_texts, get_embedding_dimension
from app.models.schemas import ContentItem

logger = logging.getLogger(__name__)


def load_corpus(corpus_path: str) -> list[ContentItem]:
    items = []
    with open(corpus_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                items.append(ContentItem(**data))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Skipping line {line_num}: {e}")
    logger.info(f"Loaded {len(items)} content items from {corpus_path}")
    return items


def create_collection(client: QdrantClient, collection_name: str, vector_size: int):
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        logger.info(f"Dropping existing collection: {collection_name}")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    client.create_payload_index(collection_name, "country", PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, "language", PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, "type", PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, "content_id", PayloadSchemaType.KEYWORD)

    logger.info(
        f"Created collection '{collection_name}' with vector_size={vector_size}, "
        f"indexed on country, language, type, content_id"
    )


def ingest_corpus(corpus_path: str | None = None) -> int:
    settings = get_settings()
    path = corpus_path or settings.corpus_path

    items = load_corpus(path)
    if not items:
        raise ValueError(f"No items loaded from {path}")

    texts = [f"{item.title}\n{item.body}" for item in items]
    logger.info(f"Embedding {len(texts)} texts...")
    vectors = embed_texts(texts)
    vector_size = get_embedding_dimension()

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    create_collection(client, settings.qdrant_collection, vector_size)

    points = []
    for item, vector in zip(items, vectors):
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, item.content_id)),
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

    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=settings.qdrant_collection, points=batch)
        logger.info(f"Upserted batch {i // batch_size + 1}: {len(batch)} points")

    total = client.count(settings.qdrant_collection).count
    logger.info(f"Ingestion complete. Total points in collection: {total}")
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    count = ingest_corpus()
    print(f"\nDone. Ingested {count} items into Qdrant.")
