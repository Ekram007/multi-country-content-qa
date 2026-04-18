"""Run the Multi-Country Content Q&A FastAPI service (CLI entry point)."""

import asyncio
import logging
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from src.core.settings import settings

# Load environment variables
load_dotenv()

def main():
    """Main entry point for the service."""
    
    # Configure root logger
    root_logger = logging.getLogger()
    if root_logger.handlers:
        print(
            f"Warning: Root logger already has {len(root_logger.handlers)} handler(s) configured. "
            f"basicConfig() will be ignored. Current level: {logging.getLevelName(root_logger.level)}"
        )
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    
    # Set compatible event loop policy on Windows systems
    # This prevents issues with async database drivers and provides better compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    print("🚀 Starting Multi-Country Content Q&A Service")
    print(f"📍 Host: {settings.app_host}")
    print(f"🔌 Port: {settings.app_port}")
    print(f"📊 Log Level: {settings.log_level}")
    print(f"🤖 LLM Provider: {settings.llm_provider}")
    print(f"🔍 Embedding Model: {settings.embedding_model}")
    print("=" * 50)
    
    # Check if we're in development mode
    is_dev = settings.log_level.upper() == "DEBUG"
    
    # Run the uvicorn server
    uvicorn.run(
        "src.service.api:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=is_dev,  # Auto-reload in debug mode
        log_level=settings.log_level.lower(),
        timeout_graceful_shutdown=30,  # 30 seconds for graceful shutdown
        # Enable access logs in debug mode
        access_log=is_dev,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Service interrupted by user")
    except Exception as e:
        print(f"💥 Failed to start service: {e}")
        sys.exit(1)