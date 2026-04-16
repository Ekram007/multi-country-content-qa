import logging
from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        
        # Load model with reduced verbosity
        _model = SentenceTransformer(settings.embedding_model)
        
        # Log device info (mps=Apple GPU, cuda=NVIDIA, cpu=fallback)
        device = str(_model.device).upper()
        logger.info(f"Model loaded on device: {device} ({_model.get_embedding_dimension()} dimensions)")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def get_embedding_dimension() -> int:
    model = get_embedding_model()
    return model.get_embedding_dimension()
