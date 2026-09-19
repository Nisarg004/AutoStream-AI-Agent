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
