from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import (
    validate_input,
    retrieve_chunks,
    fallback_retrieve,
    synthesize_answer,
    extract_citations,
    no_answer_response,
)


def route_after_validation(state: AgentState) -> str:
    return state.get("route", "error")


def route_after_retrieval(state: AgentState) -> str:
    return state.get("route", "no_answer")


def route_after_fallback(state: AgentState) -> str:
    return state.get("route", "no_answer")


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("validate_input", validate_input)
    graph.add_node("retrieve", retrieve_chunks)
    graph.add_node("fallback_retrieve", fallback_retrieve)
    graph.add_node("synthesize", synthesize_answer)
    graph.add_node("extract_citations", extract_citations)
    graph.add_node("no_answer", no_answer_response)

    graph.set_entry_point("validate_input")

    graph.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {
            "retrieve": "retrieve",
            "error": "no_answer",
        },
    )

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "synthesize": "synthesize",
            "fallback": "fallback_retrieve",
        },
    )

    graph.add_conditional_edges(
        "fallback_retrieve",
        route_after_fallback,
        {
            "synthesize": "synthesize",
            "no_answer": "no_answer",
        },
    )

    graph.add_edge("synthesize", "extract_citations")
    graph.add_edge("extract_citations", END)
    graph.add_edge("no_answer", END)

    return graph


def get_compiled_graph():
    graph = build_graph()
    return graph.compile()
