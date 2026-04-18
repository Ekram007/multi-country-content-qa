#!/bin/sh
set -e

# Run corpus load into Qdrant before serving (set RUN_INGEST_ON_START=false to skip)
case "${RUN_INGEST_ON_START:-true}" in
  0|false|FALSE|no|NO|False) ;;
  *)
    uv run python -m src.run_ingest
    ;;
esac

exec uv run python -m src.run_service
