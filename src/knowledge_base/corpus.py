"""Knowledge base corpus management utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.core.settings import settings

logger = logging.getLogger(__name__)


class CorpusManager:
    """Manager for corpus data loading and validation."""
    
    def __init__(self, corpus_path: Optional[str] = None):
        """Initialize corpus manager.
        
        Args:
            corpus_path: Path to corpus JSONL file (uses settings default if None)
        """
        self.corpus_path = Path(corpus_path or settings.corpus_path)
        self._corpus_cache: Optional[List[Dict]] = None
    
    def load_corpus(self, use_cache: bool = True) -> List[Dict]:
        """Load corpus data from JSONL file.
        
        Args:
            use_cache: Whether to use cached corpus data
            
        Returns:
            List of content items as dictionaries
            
        Raises:
            FileNotFoundError: If corpus file doesn't exist
            json.JSONDecodeError: If corpus file is malformed
        """
        if use_cache and self._corpus_cache is not None:
            return self._corpus_cache
            
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")
        
        logger.info(f"Loading corpus from {self.corpus_path}")
        
        corpus_items = []
        line_count = 0
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    self._validate_corpus_item(item, line_num)
                    corpus_items.append(item)
                    line_count += 1
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON at line {line_num}: {e}")
                    raise
                except ValueError as e:
                    logger.error(f"Invalid corpus item at line {line_num}: {e}")
                    raise
        
        logger.info(f"Loaded {line_count} corpus items from {self.corpus_path}")
        
        if use_cache:
            self._corpus_cache = corpus_items
            
        return corpus_items
    
    def get_corpus_stats(self) -> Dict:
        """Get statistics about the corpus.
        
        Returns:
            Dictionary with corpus statistics
        """
        corpus = self.load_corpus()
        
        stats = {
            "total_items": len(corpus),
            "countries": {},
            "languages": {},
            "types": {},
        }
        
        for item in corpus:
            # Count by country
            country = item["country"]
            stats["countries"][country] = stats["countries"].get(country, 0) + 1
            
            # Count by language
            language = item["language"]
            stats["languages"][language] = stats["languages"].get(language, 0) + 1
            
            # Count by type
            item_type = item["type"]
            stats["types"][item_type] = stats["types"].get(item_type, 0) + 1
        
        return stats
    
    def get_items_by_filters(
        self, 
        country: Optional[str] = None,
        language: Optional[str] = None,
        item_type: Optional[str] = None
    ) -> List[Dict]:
        """Get corpus items matching the given filters.
        
        Args:
            country: Filter by country code
            language: Filter by language code
            item_type: Filter by content type
            
        Returns:
            List of matching corpus items
        """
        corpus = self.load_corpus()
        filtered_items = []
        
        for item in corpus:
            if country and item["country"] != country:
                continue
            if language and item["language"] != language:
                continue
            if item_type and item["type"] != item_type:
                continue
                
            filtered_items.append(item)
        
        return filtered_items
    
    def _validate_corpus_item(self, item: Dict, line_num: int) -> None:
        """Validate a single corpus item.
        
        Args:
            item: Corpus item dictionary
            line_num: Line number for error reporting
            
        Raises:
            ValueError: If item is invalid
        """
        required_fields = [
            "content_id", "country", "language", "type", 
            "version", "title", "body", "updated_at"
        ]
        
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Missing required field '{field}' at line {line_num}")
            
        # Validate field types
        if not isinstance(item["content_id"], str):
            raise ValueError(f"content_id must be string at line {line_num}")
        if not isinstance(item["country"], str):
            raise ValueError(f"country must be string at line {line_num}")
        if not isinstance(item["language"], str):
            raise ValueError(f"language must be string at line {line_num}")
        if not isinstance(item["version"], (int, float)):
            raise ValueError(f"version must be number at line {line_num}")
        if not isinstance(item["title"], str):
            raise ValueError(f"title must be string at line {line_num}")
        if not isinstance(item["body"], str):
            raise ValueError(f"body must be string at line {line_num}")
    
    def clear_cache(self) -> None:
        """Clear the corpus cache."""
        self._corpus_cache = None


# Global corpus manager instance
_corpus_manager: Optional[CorpusManager] = None


def get_corpus_manager() -> CorpusManager:
    """Get the global corpus manager instance.
    
    Returns:
        CorpusManager: Global corpus manager instance
    """
    global _corpus_manager
    if _corpus_manager is None:
        _corpus_manager = CorpusManager()
    return _corpus_manager


def load_corpus() -> List[Dict]:
    """Load corpus using the global manager.
    
    Returns:
        List of corpus items
    """
    return get_corpus_manager().load_corpus()


def get_corpus_stats() -> Dict:
    """Get corpus statistics using the global manager.
    
    Returns:
        Corpus statistics dictionary
    """
    return get_corpus_manager().get_corpus_stats()