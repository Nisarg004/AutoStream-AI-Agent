"""The state schema shared by every node.

chat_history used to live here: the CLI loop appended every turn to it and
no node ever read it. It was removed rather than wired into a prompt, so the
schema now describes only state the graph actually uses. These tests are the
canary against a write-only field creeping back in.
"""
from agent.state import AgentState

EXPECTED_FIELDS = {
    "user_message",
    "response",
    "intent",
    "lead_name",
    "lead_email",
    "lead_platform",
    "lead_collection_step",
    "lead_captured",
}


def test_agent_state_has_exactly_the_expected_fields():
    assert set(AgentState.__annotations__) == EXPECTED_FIELDS


def test_chat_history_is_not_in_the_schema():
    assert "chat_history" not in AgentState.__annotations__
