"""Embedding model abstraction and utilities."""

import logging
from functools import cache
from typing import Any, List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.settings import settings

logger = logging.getLogger(__name__)

# Global embedding model instance
_embedding_model: SentenceTransformer | None = None


class EmbeddingModel:
    """Wrapper class for sentence transformer embedding model."""
    
    def __init__(self, model_name: str):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        
    @property
    def model(self) -> SentenceTransformer:
        """Get or load the embedding model (lazy loading)."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            
            # Log device and dimension info
            device = str(self._model.device).upper()
            dimensions = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded on device: {device} ({dimensions} dimensions)")
            
        return self._model
    
    def encode(self, texts: str | List[str], normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        """Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            normalize_embeddings: Whether to normalize embeddings to unit vectors
            **kwargs: Additional arguments passed to model.encode()
            
        Returns:
            numpy array of embeddings
        """
        show_progress_bar = kwargs.pop("show_progress_bar", False)
        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between embeddings."""
        from sentence_transformers.util import cos_sim
        return cos_sim(embeddings1, embeddings2)


def get_embedding_model() -> EmbeddingModel:
    """Get or create the global embedding model instance.
    
    Returns:
        EmbeddingModel: Configured embedding model instance
    """
    global _embedding_model
    
    if _embedding_model is None:
        _embedding_model = EmbeddingModel(settings.embedding_model)
    
    return _embedding_model


def embed_texts(texts: List[str], show_progress: bool = True) -> List[List[float]]:
    """Embed a list of texts.
    
    Args:
        texts: List of texts to embed
        show_progress: Whether to show progress bar
        
    Returns:
        List of embedding vectors
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=show_progress)
    return embeddings.tolist()


def embed_query(text: str) -> List[float]:
    """Embed a single query text.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector as list
    """
    model = get_embedding_model()
    embedding = model.encode(text)
    return embedding.tolist()


@cache
def get_embedding_dimension() -> int:
    """Get the embedding dimension (cached).
    
    Returns:
        Embedding dimension
    """
    model = get_embedding_model()
    return model.get_embedding_dimension()


def reset_embedding_model() -> None:
    """Reset the embedding model instance (useful for testing)."""
    global _embedding_model
    _embedding_model = None