"""The two conditional-edge functions that steer the graph."""
import pytest
from langgraph.graph import END

from agent.graph import route_after_intent, route_after_lead_collection


@pytest.mark.parametrize(
    "intent,expected",
    [
        ("greeting", END),  # intent_node already wrote the reply
        ("product_query", "rag"),
        ("high_intent", "lead_collection"),
    ],
)
def test_route_after_intent(intent, expected):
    assert route_after_intent({"intent": intent}) == expected


@pytest.mark.parametrize("intent", [None, "nonsense_label"])
def test_route_after_intent_sends_unknown_labels_to_rag(intent):
    """An unrecognised label must never start asking for personal details."""
    assert route_after_intent({"intent": intent}) == "rag"


def test_route_after_lead_collection_fires_the_tool_when_complete():
    assert route_after_lead_collection({"lead_collection_step": "complete"}) == "tool_execution"


@pytest.mark.parametrize("step", [None, "ask_name", "ask_email", "ask_platform"])
def test_route_after_lead_collection_ends_the_turn_mid_flow(step):
    """Mid-flow the turn ends so the CLI can collect the next answer."""
    assert route_after_lead_collection({"lead_collection_step": step}) == END
