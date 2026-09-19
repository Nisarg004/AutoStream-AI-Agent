"""Behaviour when the OpenAI API is unreachable.

classify_intent already degrades to keyword matching, but that only helps if
the node it routes to can still produce a reply. Both LLM-calling nodes are
exercised here with a client that raises, and the working path is pinned too
so the guards cannot quietly change a successful answer.
"""
from unittest.mock import MagicMock

import pytest

import agent.nodes as nodes
import utils.intent_classifier as ic
from agent.graph import graph


def _raising_client() -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("503 service unavailable")
    return client


def _client_returning(text: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = text
    return client


def _state(message: str) -> dict:
    return {
        "user_message": message,
        "response": "",
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_collection_step": None,
        "lead_captured": False,
    }


# --- intent_node -----------------------------------------------------------


def test_greeting_uses_the_llm_reply_when_the_api_works(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda m: "greeting")
    monkeypatch.setattr(nodes, "client", _client_returning("  Hey there!  "))

    assert nodes.intent_node(_state("hi"))["response"] == "Hey there!"


def test_greeting_falls_back_to_a_canned_reply_when_the_api_fails(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda m: "greeting")
    monkeypatch.setattr(nodes, "client", _raising_client())

    update = nodes.intent_node(_state("hi"))

    assert update["intent"] == "greeting"
    assert update["response"].strip(), "a greeting must still get a reply"


# --- rag_node --------------------------------------------------------------


def test_rag_uses_the_llm_answer_when_the_api_works(monkeypatch):
    monkeypatch.setattr(nodes, "retrieve", lambda q: "Pro Plan: $79/month.")
    monkeypatch.setattr(nodes, "client", _client_returning("The Pro plan is $79/month."))

    assert nodes.rag_node(_state("price?"))["response"] == "The Pro plan is $79/month."


def test_rag_hands_over_the_retrieved_text_when_the_api_fails(monkeypatch):
    monkeypatch.setattr(nodes, "retrieve", lambda q: "Pro Plan: $79/month, 4K resolution.")
    monkeypatch.setattr(nodes, "client", _raising_client())

    response = nodes.rag_node(_state("price?"))["response"]

    assert "Pro Plan: $79/month, 4K resolution." in response, "facts must not be dropped"
    assert response.strip() != "Pro Plan: $79/month, 4K resolution.", "and must be explained"


def test_rag_still_replies_when_nothing_was_retrieved(monkeypatch):
    monkeypatch.setattr(nodes, "retrieve", lambda q: "")
    monkeypatch.setattr(nodes, "client", _raising_client())

    assert nodes.rag_node(_state("?"))["response"].strip()


# --- whole turn, nothing reachable -----------------------------------------


@pytest.fixture
def total_outage(monkeypatch):
    """Every OpenAI call fails: the classifier's and both nodes'."""
    monkeypatch.setattr(ic, "client", _raising_client())
    monkeypatch.setattr(nodes, "client", _raising_client())


def test_greeting_turn_survives_a_total_outage(total_outage):
    result = graph.invoke(_state("hi"))

    assert result["intent"] == "greeting", "keyword fallback should still label it"
    assert result["response"].strip()


def test_product_query_turn_survives_a_total_outage(total_outage, monkeypatch):
    monkeypatch.setattr(nodes, "retrieve", lambda q: "Pro Plan: $79/month.")

    result = graph.invoke(_state("how much is the pro plan?"))

    assert result["intent"] == "product_query"
    assert "$79/month" in result["response"], "the retrieved facts should survive"


def test_lead_flow_turn_survives_a_total_outage(total_outage):
    """The lead path never calls the LLM, so it should be untouched."""
    result = graph.invoke(_state("I want to sign up"))

    assert result["intent"] == "high_intent"
    assert result["lead_collection_step"] == "ask_name"
    assert result["response"].strip()
