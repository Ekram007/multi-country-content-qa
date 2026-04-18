"""
Content Q&A Agent.

Simple 2-node agent (model + tools) that lets the LLM decide when to call tools.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda, 
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from src.agents.content_qa.tools import content_qa_tools
from src.core.llm import get_llm

logger = logging.getLogger(__name__)


class ContentQAState(MessagesState, total=False):
    """State for the Content Q&A Agent - Simple MessagesState."""
    
    remaining_steps: RemainingSteps


def load_system_prompt() -> str:
    """Load system prompt from text file."""
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Load prompt from file
    prompt_file = Path(__file__).parent.parent.parent / "prompts" / "content_qa_agent_system_prompt.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_file}")
    
    prompt = prompt_file.read_text(encoding='utf-8').strip()
    
    return prompt


def wrap_model(model: BaseChatModel) -> RunnableSerializable[ContentQAState, AIMessage]:
    """Wrap the model with tools and system instructions."""
    bound_model = model.bind_tools(content_qa_tools)
    
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=load_system_prompt())] + state["messages"],
        name="StateModifier",
    )
    
    return preprocessor | bound_model


def call_model(state: ContentQAState, config: RunnableConfig) -> ContentQAState:
    """Call the model with tools and system prompt."""
    model = get_llm()  # Get configured LLM
    model_runnable = wrap_model(model)
    
    try:
        # Use synchronous invoke
        response = model_runnable.invoke(state, config)
        
        # Check if we're running out of steps and still have tool calls
        if state.get("remaining_steps", 10) < 2 and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="I need more steps to process this request fully. Please try asking a more specific question.",
                    )
                ]
            }
        
        return {"messages": [response]}
        
    except Exception as e:
        logger.error(f"Model call failed: {e}")
        return {
            "messages": [
                AIMessage(
                    content="I apologize, but I'm experiencing a technical issue. Please try again in a moment.",
                )
            ]
        }


def should_continue(state: ContentQAState) -> Literal["tools", "done"]:
    """Check if there are pending tool calls."""
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage):
        return "done"
        
    if last_message.tool_calls:
        logger.debug(f"Tool calls pending: {len(last_message.tool_calls)}")
        return "tools"
        
    return "done"


# Build the agent graph - Simple 2-node design
agent_graph = StateGraph(ContentQAState)

# Add nodes
agent_graph.add_node("model", call_model)
agent_graph.add_node("tools", ToolNode(content_qa_tools))

# Set entry point
agent_graph.set_entry_point("model")

# Add edges
agent_graph.add_edge("tools", "model")  # Always go back to model after tools
agent_graph.add_conditional_edges(
    "model", 
    should_continue, 
    {"tools": "tools", "done": END}
)

# Compile the agent
content_qa_agent = agent_graph.compile()


# Export the compiled agent directly
# This is what gets imported and used
__all__ = ["content_qa_agent"]