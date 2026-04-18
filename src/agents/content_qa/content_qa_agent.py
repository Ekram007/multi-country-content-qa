"""Content Q&A Agent - Agent-Service-Toolkit Style."""

import logging
import time
from pathlib import Path
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from src.agents.content_qa.schema import AgentState
from src.agents.content_qa.tools import (
    validate_input,
    retrieve_chunks,
    fallback_retrieve, 
    synthesize_answer,
    extract_citations,
    no_answer_response
)
from src.schema.models import AskRequest, AskResponse, Citation, Trace
from src.core.settings import settings

logger = logging.getLogger(__name__)


class ContentQAAgent:
    """Multi-country content Q&A agent with citation support."""
    
    def __init__(self):
        """Initialize the content Q&A agent."""
        self.name = "content_qa"
        self.description = "Multi-country content Q&A with citations"
        self.version = "1.0.0"
        self._graph = None
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("validate_input", validate_input)
        graph.add_node("retrieve", retrieve_chunks)
        graph.add_node("fallback_retrieve", fallback_retrieve)
        graph.add_node("synthesize", synthesize_answer)
        graph.add_node("extract_citations", extract_citations)
        graph.add_node("no_answer", no_answer_response)

        # Set entry point
        graph.set_entry_point("validate_input")

        # Add conditional edges with routing logic
        graph.add_conditional_edges(
            "validate_input",
            lambda state: state.get("route", "error"),
            {
                "retrieve": "retrieve",
                "error": "no_answer",
            },
        )

        graph.add_conditional_edges(
            "retrieve",
            lambda state: state.get("route", "no_answer"),
            {
                "synthesize": "synthesize",
                "fallback": "fallback_retrieve",
            },
        )

        graph.add_conditional_edges(
            "fallback_retrieve",
            lambda state: state.get("route", "no_answer"),
            {
                "synthesize": "synthesize",
                "no_answer": "no_answer",
            },
        )

        # Add deterministic edges
        graph.add_edge("synthesize", "extract_citations")
        graph.add_edge("extract_citations", END)
        graph.add_edge("no_answer", END)

        return graph
        
    @property
    def graph(self):
        """Get or initialize the compiled LangGraph workflow."""
        if self._graph is None:
            workflow = self._build_graph()
            self._graph = workflow.compile()
        return self._graph
    
    def ask(self, request: AskRequest) -> AskResponse:
        """Process a Q&A request.
        
        Args:
            request: The question and filtering parameters
            
        Returns:
            Response with answer, citations, and trace info
        """
        start_time = time.time()
        
        # Prepare agent state
        initial_state: AgentState = {
            "question": request.question,
            "country": request.country.upper(),
            "language": request.language,
        }
        
        logger.info(
            f"Processing request: country={request.country}, "
            f"language={request.language}, question='{request.question[:50]}...'"
        )
        
        try:
            # Execute agent workflow
            result = self.graph.invoke(initial_state)
            
            # Calculate processing time
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Build citations
            citations = [
                Citation(
                    content_id=c["content_id"],
                    type=c["type"],
                    excerpt=c["excerpt"],
                    match_score=c["match_score"],
                )
                for c in result.get("citations", [])
            ]
            
            # Build response
            response = AskResponse(
                answer=result.get("answer", "Unable to generate an answer."),
                language_used=request.language,
                citations=citations,
                trace=Trace(
                    retrieval_count=len(result.get("retrieved_chunks", [])),
                    latency_ms=latency_ms,
                    model=settings.llm_model,
                ),
            )
            
            logger.info(
                f"Request completed: citations={len(citations)}, "
                f"latency={latency_ms}ms"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            
            # Return error response in expected format
            latency_ms = int((time.time() - start_time) * 1000)
            
            return AskResponse(
                answer=f"Unable to process request: {str(e)}",
                language_used=request.language,
                citations=[],
                trace=Trace(
                    retrieval_count=0,
                    latency_ms=latency_ms,
                    model=settings.llm_model,
                ),
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health status.
        
        Returns:
            Health status information
        """
        try:
            # Test graph compilation
            graph = self.graph
            
            return {
                "status": "healthy",
                "agent": self.name,
                "version": self.version,
                "graph_compiled": graph is not None,
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "agent": self.name,
                "version": self.version,
                "error": str(e),
            }


# Global agent instance (singleton)
_content_qa_agent: ContentQAAgent | None = None


def get_content_qa_agent() -> ContentQAAgent:
    """Get the global ContentQA agent instance.
    
    Returns:
        ContentQAAgent: Global agent instance
    """
    global _content_qa_agent
    if _content_qa_agent is None:
        _content_qa_agent = ContentQAAgent()
    return _content_qa_agent