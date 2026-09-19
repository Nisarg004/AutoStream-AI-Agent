"""Intent classification: the LLM path, and the keyword fallback behind it."""
from unittest.mock import MagicMock

import pytest

import utils.intent_classifier as ic


def _stub_llm(monkeypatch, *, content=None, error=None) -> MagicMock:
    client = MagicMock()
    if error is not None:
        client.chat.completions.create.side_effect = error
    else:
        client.chat.completions.create.return_value.choices[0].message.content = content
    monkeypatch.setattr(ic, "client", client)
    return client


def test_llm_label_is_normalised(monkeypatch):
    _stub_llm(monkeypatch, content="  High_Intent\n")

    assert ic.classify_intent("I want the Pro plan") == "high_intent"


def test_llm_failure_falls_back_to_keywords(monkeypatch):
    """An API error (quota, network, outage) must not break the turn."""
    client = _stub_llm(monkeypatch, error=RuntimeError("429 rate limit exceeded"))

    assert ic.classify_intent("I want to buy the Pro plan") == "high_intent"
    assert client.chat.completions.create.called, "the LLM should be tried first"


@pytest.mark.parametrize(
    "message,expected",
    [
        ("I want to sign up", "high_intent"),
        ("can I purchase the pro plan", "high_intent"),
        ("how much does it cost", "product_query"),
        ("what is your refund policy", "product_query"),
        ("hey there", "greeting"),
    ],
)
def test_keyword_fallback_labels(message, expected):
    assert ic._keyword_fallback(message) == expected


def test_keyword_fallback_prefers_buying_intent():
    """Both keyword sets match here; buying intent must win."""
    assert ic._keyword_fallback("I want to buy it, what's the price?") == "high_intent"


@pytest.mark.parametrize(
    "message,expected",
    [
        # A plan name is a topic, not an intention.
        ("how much is the pro plan?", "product_query"),
        ("what does the basic plan include?", "product_query"),
        # Buying language still wins, even phrased as a question.
        ("can I purchase the basic plan?", "high_intent"),
        # Commitment, payment and sign-up wording.
        ("let's do it", "high_intent"),
        ("take my money", "high_intent"),
        ("where do I pay?", "high_intent"),
        ("set me up", "high_intent"),
        # Openers are matched explicitly now, not left to the default.
        ("good morning", "greeting"),
        ("thanks!", "greeting"),
    ],
)
def test_keyword_fallback_rules(message, expected):
    assert ic._keyword_fallback(message) == expected


def test_unrecognised_message_defaults_to_product_query():
    """The default used to be greeting, which silently swallowed real questions."""
    assert ic._keyword_fallback("mmm, not sure what I need yet") == "product_query"


def test_matching_respects_word_boundaries():
    """'hi' must not fire inside 'this'; substring matching used to allow that."""
    assert ic._keyword_fallback("what is this?") == "product_query"
