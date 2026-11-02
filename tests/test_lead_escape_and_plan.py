"""Leaving lead collection, questions during it, and naming the right plan."""
import uuid

import pytest

import agent.nodes as nodes
from agent.graph import graph
from utils.intent_classifier import extract_plan, is_product_question


@pytest.fixture
def chat(monkeypatch, captured):
    """Send messages on one thread. The keyword rules stand in for the LLM."""
    from utils.intent_classifier import _keyword_fallback

    monkeypatch.setattr(nodes, "classify_intent", _keyword_fallback)
    monkeypatch.setattr(nodes, "retrieve", lambda q: f"retrieved for: {q}")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    def send(*messages):
        for message in messages:
            result = graph.invoke({"user_message": message}, config)
        return result

    send.values = lambda: graph.get_state(config).values
    return send


# --- cancelling ------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    ["cancel", "Never mind", "nevermind", "ok, never mind.", "forget it", "no thanks", "cancel please"],
)
def test_escape_phrases_clear_the_flow(chat, captured, phrase):
    chat("sign me up for pro", "Alice", "alice@example.com")

    result = chat(phrase)

    assert "cancelled" in result["response"]
    values = chat.values()
    assert values["lead_collection_step"] is None
    assert values["lead_name"] is None
    assert values["lead_email"] is None
    assert values["lead_plan"] is None
    assert captured == []


def test_can_cancel_at_each_step(chat):
    for messages in (["sign me up"], ["sign me up", "Alice"], ["sign me up", "Alice", "a@b.co"]):
        chat(*messages)
        chat("cancel")
        assert chat.values()["lead_collection_step"] is None


def test_a_new_flow_after_cancelling_starts_empty(chat, captured):
    chat("sign me up", "Alice", "cancel")

    chat("sign me up", "Bob", "bob@example.com", "TikTok")

    assert captured == [("Bob", "bob@example.com", "TikTok")]


def test_cancel_wording_outside_a_flow_is_a_product_question(chat):
    result = chat("cancel")

    assert result["intent"] == "product_query"
    assert "cancelled" not in result["response"]


# --- questions during the flow ---------------------------------------------


def test_question_mid_flow_keeps_fields_and_repeats_the_pending_question(chat):
    chat("sign me up", "Alice")

    result = chat("what's your refund policy?")

    assert "retrieved for: what's your refund policy?" in result["response"]
    assert "What's your email address?" in result["response"]
    values = chat.values()
    assert values["lead_name"] == "Alice"
    assert values["lead_collection_step"] == "ask_email"
    assert values.get("lead_email") is None


def test_flow_finishes_normally_after_a_question(chat, captured):
    chat("sign me up", "Alice", "how much is pro?", "alice@example.com", "does pro have captions?", "YouTube")

    assert captured == [("Alice", "alice@example.com", "YouTube")]


def test_question_that_mentions_cancelling_does_not_cancel(chat):
    chat("sign me up", "Alice")

    result = chat("what happens if I cancel my plan?")

    assert result["intent"] == "product_query"
    assert chat.values()["lead_name"] == "Alice"


@pytest.mark.parametrize("name", ["Will Smith", "Mark Pro", "Grace Hopper"])
def test_names_that_resemble_questions_are_stored_as_names(chat, name):
    chat("sign me up")

    assert chat(name)["lead_name"] == name


@pytest.mark.parametrize("platform", ["YouTube", "Pro Tools", "YouTube?"])
def test_platform_answers_complete_the_flow(chat, captured, platform):
    chat("sign me up", "Alice", "alice@example.com", platform)

    assert captured == [("Alice", "alice@example.com", platform)]


@pytest.mark.parametrize(
    "message,expected",
    [
        ("pricing?", True),
        ("how much is pro", True),
        ("do you work with snapchat?", True),
        ("Alice", False),
        ("alice@example.com", False),
        ("Will Smith", False),
        ("Support", False),
    ],
)
def test_is_product_question(message, expected):
    assert is_product_question(message) is expected


# --- the plan in the closing message ---------------------------------------


@pytest.mark.parametrize(
    "opening,expected",
    [
        ("sign me up for the Basic plan", "the Basic plan"),
        ("sign me up for pro", "the Pro plan"),
        ("sign me up", "AutoStream"),
    ],
)
def test_closing_message_names_the_plan_requested(chat, opening, expected):
    result = chat(opening, "Alice", "alice@example.com", "YouTube")

    assert f"Your interest in {expected} has been recorded" in result["response"]


def test_closing_message_uses_a_plan_asked_about_before_signing_up(chat):
    result = chat("what does basic cost?", "sign me up", "Alice", "alice@example.com", "YouTube")

    assert "the Basic plan" in result["response"]


def test_question_about_another_plan_does_not_change_the_chosen_plan(chat):
    result = chat("sign me up for pro", "Alice", "does basic have captions?", "alice@example.com", "YouTube")

    assert "the Pro plan" in result["response"]


def test_a_plan_does_not_leak_into_the_next_lead(chat):
    chat("sign me up for basic", "Alice", "alice@example.com", "YouTube")

    result = chat("sign me up", "Bob", "bob@example.com", "TikTok")

    assert "the Basic plan" not in result["response"]


@pytest.mark.parametrize(
    "message,expected",
    [
        ("the Pro plan please", "Pro"),
        ("BASIC", "Basic"),
        ("basic vs pro?", None),
        ("I have a basic question", None),
        ("any pro tips?", None),
        ("improve my process", None),
    ],
)
def test_extract_plan(message, expected):
    assert extract_plan(message) == expected
