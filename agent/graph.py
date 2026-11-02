from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    intent_node,
    rag_node,
    lead_collection_node,
    tool_execution_node,
)


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> str:
    """Decide next node after intent classification."""
    intent = state.get("intent")

    # Greeting response is already set in intent_node — terminate
    if intent == "greeting":
        return END

    if intent == "high_intent":
        return "lead_collection"

    # product_query, or anything unexpected: answering is harmless, while the
    # lead flow would ask a stranger for personal details.
    return "rag"


def route_after_lead_collection(state: AgentState) -> str:
    """Decide next node after a lead collection step."""
    if state.get("lead_collection_step") == "complete":
        return "tool_execution"
    return END


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("intent", intent_node)
    builder.add_node("rag", rag_node)
    builder.add_node("lead_collection", lead_collection_node)
    builder.add_node("tool_execution", tool_execution_node)

    # Entry point
    builder.set_entry_point("intent")

    # Edges
    builder.add_conditional_edges("intent", route_after_intent)
    builder.add_edge("rag", END)
    builder.add_conditional_edges("lead_collection", route_after_lead_collection)
    builder.add_edge("tool_execution", END)

    return builder.compile()


graph = build_graph()
