from typing import Optional, Literal
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state passed between all LangGraph nodes."""

    # Conversation
    user_message: str
    response: str

    # Intent classification
    intent: Optional[Literal["greeting", "product_query", "high_intent"]]

    # Lead collection fields
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_collection_step: Optional[Literal["ask_name", "ask_email", "ask_platform", "complete"]]

    # Flow control
    lead_captured: bool
