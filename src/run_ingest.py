"""Run corpus ingestion into Qdrant vector database.

This script handles the data ingestion pipeline.
Agent-Service-Toolkit style runner.
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.core.settings import settings
from src.ingestion.ingest import ingest_corpus

# Load environment variables
load_dotenv()


def main():
    """Main entry point for corpus ingestion."""
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    
    print("📁 Multi-Country Content Q&A - Corpus Ingestion")
    print(f"📊 Log Level: {settings.log_level}")
    print(f"🗃️  Corpus Path: {settings.corpus_path}")
    print(f"🔍 Embedding Model: {settings.embedding_model}")
    print(f"🏠 Qdrant Host: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"📚 Collection: {settings.qdrant_collection}")
    print("=" * 50)
    
    try:
        # Run corpus ingestion
        results = ingest_corpus()
        
        if results["success"]:
            print(f"\n✅ Ingestion completed successfully!")
            print(f"   📁 Corpus: {results['corpus_path']}")
            print(f"   📊 Items loaded: {results['items_loaded']}")
            print(f"   📤 Points upserted: {results['points_upserted']}")
            print(f"   🗄️  Collection: {results['collection_name']}")
            print(f"   📈 Vector size: {results['vector_size']}")
            
            verification = results.get("verification", {})
            if verification:
                print(f"   ✅ Final count: {verification.get('points_count', 'unknown')}")
                print(f"   📊 Status: {verification.get('status', 'unknown')}")
                
        else:
            print(f"\n❌ Ingestion failed!")
            print(f"   Error: {results['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Ingestion interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n💥 Unexpected error during ingestion: {e}")
        logging.exception("Ingestion failed with unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()