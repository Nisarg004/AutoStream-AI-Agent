"""The state schema shared by every node."""
from agent.state import AgentState

EXPECTED_FIELDS = {
    "user_message",
    "response",
    "chat_history",
    "intent",
    "lead_name",
    "lead_email",
    "lead_platform",
    "lead_plan",
    "lead_collection_step",
    "lead_captured",
}


def test_agent_state_has_exactly_the_expected_fields():
    assert set(AgentState.__annotations__) == EXPECTED_FIELDS
