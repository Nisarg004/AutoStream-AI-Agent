"""Regression tests for the multi-turn lead capture flow.

These drive the real compiled LangGraph the same way main.run_chat does
(persistent state dict, merged after every turn). The LLM intent classifier
and the capture tool are stubbed, so no network access or API key is needed.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")  # client is built at import time

import pytest

import agent.nodes as nodes
from agent.graph import graph


def _new_state() -> dict:
    return {
        "user_message": "",
        "chat_history": [],
        "response": "",
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_collection_step": None,
        "lead_captured": False,
    }


@pytest.fixture
def captured(monkeypatch):
    """Stub the classifier and the tool; return the list of captured leads."""
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(nodes, "classify_intent", lambda msg: "high_intent")
    monkeypatch.setattr(
        nodes, "mock_lead_capture", lambda name, email, platform: calls.append((name, email, platform))
    )
    return calls


def _turn(state: dict, message: str) -> None:
    """One CLI turn: inject message, invoke graph, merge result back."""
    state["user_message"] = message
    state.update(graph.invoke(state))


def _run_lead_flow(state: dict, name: str, email: str, platform: str) -> None:
    _turn(state, "I want to sign up for the Pro plan")
    _turn(state, name)
    _turn(state, email)
    _turn(state, platform)


def test_lead_is_captured_once_after_full_flow(captured):
    state = _new_state()
    _run_lead_flow(state, "Alice", "alice@example.com", "YouTube")

    assert captured == [("Alice", "alice@example.com", "YouTube")]
    assert state["lead_captured"] is True


def test_state_is_reset_after_successful_capture(captured):
    state = _new_state()
    _run_lead_flow(state, "Alice", "alice@example.com", "YouTube")

    assert state["lead_collection_step"] is None
    assert state["lead_name"] is None
    assert state["lead_email"] is None
    assert state["lead_platform"] is None


def test_later_high_intent_message_does_not_recapture_same_lead(captured):
    state = _new_state()
    _run_lead_flow(state, "Alice", "alice@example.com", "YouTube")
    assert len(captured) == 1

    # User says something high-intent again after being captured.
    _turn(state, "Actually I want to buy another plan")

    assert len(captured) == 1, "same lead was captured a second time"
    # A fresh flow starts by asking for the name.
    assert state["lead_collection_step"] == "ask_name"


def test_second_flow_captures_new_lead_separately(captured):
    state = _new_state()
    _run_lead_flow(state, "Alice", "alice@example.com", "YouTube")
    _run_lead_flow(state, "Bob", "bob@example.com", "TikTok")

    assert captured == [
        ("Alice", "alice@example.com", "YouTube"),
        ("Bob", "bob@example.com", "TikTok"),
    ]


# ---------------------------------------------------------------------------
# Input validation
#
# These call lead_collection_node directly: validation is pure state handling,
# so no graph invocation or LLM stubbing is needed.
# ---------------------------------------------------------------------------


def _at_step(step: str, message: str) -> dict:
    return _new_state() | {"lead_collection_step": step, "user_message": message}


@pytest.mark.parametrize(
    "email",
    [
        "alice@example.com",
        "a.b+tag@sub.example.co.uk",
        "  alice@example.com  ",
    ],
)
def test_valid_email_advances_to_platform(email):
    update = nodes.lead_collection_node(_at_step("ask_email", email))

    assert update["lead_email"] == email.strip()
    assert update["lead_collection_step"] == "ask_platform"


@pytest.mark.parametrize(
    "email",
    [
        "",
        "   ",
        "not-an-email",
        "alice@",
        "@example.com",
        "alice@example",
        "alice example.com",
        "alice@exa mple.com",
    ],
)
def test_invalid_email_reasks_without_advancing(email):
    update = nodes.lead_collection_node(_at_step("ask_email", email))

    assert "lead_email" not in update, "invalid email must not be stored"
    assert "lead_collection_step" not in update, "step must not advance"
    assert "email" in update["response"].lower()


def test_valid_platform_completes_the_flow():
    update = nodes.lead_collection_node(_at_step("ask_platform", " YouTube "))

    assert update["lead_platform"] == "YouTube"
    assert update["lead_collection_step"] == "complete"


@pytest.mark.parametrize("platform", ["", "   "])
def test_empty_platform_reasks_without_advancing(platform):
    update = nodes.lead_collection_node(_at_step("ask_platform", platform))

    assert "lead_platform" not in update, "empty platform must not be stored"
    assert "lead_collection_step" not in update, "step must not advance"
    assert "platform" in update["response"].lower()


def test_flow_recovers_after_an_invalid_email(captured):
    """A rejected email re-asks, and the corrected one carries the flow through."""
    state = _new_state()
    _turn(state, "I want to sign up for the Pro plan")
    _turn(state, "Alice")
    _turn(state, "alice-at-example.com")

    assert state["lead_collection_step"] == "ask_email", "should still be asking for email"
    assert captured == []

    _turn(state, "alice@example.com")
    _turn(state, "YouTube")

    assert captured == [("Alice", "alice@example.com", "YouTube")]


def test_valid_name_advances_to_email():
    update = nodes.lead_collection_node(_at_step("ask_name", " Alice "))

    assert update["lead_name"] == "Alice"
    assert update["lead_collection_step"] == "ask_email"


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_name_reasks_without_advancing(name):
    update = nodes.lead_collection_node(_at_step("ask_name", name))

    assert "lead_name" not in update, "empty name must not be stored"
    assert "lead_collection_step" not in update, "step must not advance"
    assert "name" in update["response"].lower()


@pytest.mark.parametrize("missing", ["lead_name", "lead_email", "lead_platform"])
def test_tool_execution_guard_survives_a_none_field(missing, captured):
    """A None field must take the reset path, not raise AttributeError."""
    state = _new_state() | {
        "lead_name": "Alice",
        "lead_email": "alice@example.com",
        "lead_platform": "YouTube",
        "lead_collection_step": "complete",
    }
    state[missing] = None

    update = nodes.tool_execution_node(state)

    assert captured == [], "the tool must not fire with a missing field"
    assert update["lead_captured"] is False
    assert update["lead_collection_step"] is None
