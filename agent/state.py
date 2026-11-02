import operator
from typing import Annotated, Literal, Optional, TypedDict


class AgentState(TypedDict):
    """Shared state passed between all LangGraph nodes."""

    # Conversation
    user_message: str
    response: str
    # Greeting and product exchanges only. Lead answers stay out, so personal
    # details are never sent to the LLM in a later turn's history.
    chat_history: Annotated[list[dict], operator.add]

    # Intent classification
    intent: Optional[Literal["greeting", "product_query", "high_intent"]]

    # Lead collection fields
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_plan: Optional[str]
    lead_collection_step: Optional[Literal["ask_name", "ask_email", "ask_platform", "complete"]]

    # Flow control
    lead_captured: bool
