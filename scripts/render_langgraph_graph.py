#!/usr/bin/env python3
"""Render the Content Q&A LangGraph to a PNG (Mermaid via Graphviz)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("screenshots/03_langgraph_graph.png"),
        help="Output PNG path (default: screenshots/03_langgraph_graph.png)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.agents.content_qa.content_qa_agent import content_qa_agent

    args.output.parent.mkdir(parents=True, exist_ok=True)
    content_qa_agent.get_graph().draw_mermaid_png(output_file_path=str(args.output))
    print(f"Wrote {args.output.resolve()} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
