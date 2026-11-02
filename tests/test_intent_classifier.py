"""Intent classification: the LLM path, and the keyword fallback behind it."""
from unittest.mock import MagicMock

import sys
from pathlib import Path

import pytest

import utils.intent_classifier as ic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from baseline_keyword import keyword_fallback_before as rules_v0


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


# --- bugs reported in the original rules -----------------------------------


@pytest.mark.parametrize(
    "message,buggy,fixed",
    [
        # High-intent keywords ran before product questions.
        ("What does the Pro plan cost?", "high_intent", "product_query"),
        # Substring matches.
        ("country", "high_intent", "product_query"),
        ("restart", "high_intent", "product_query"),
        ("I want to know the refund policy", "high_intent", "product_query"),
        # Unmatched messages became greetings.
        ("asdf qwer", "greeting", "product_query"),
    ],
)
def test_reported_bug_exists_in_v0_and_is_fixed_now(message, buggy, fixed):
    assert rules_v0(message) == buggy, "the frozen v0 rules should still show the bug"
    assert ic._keyword_fallback(message) == fixed


# --- current precedence ----------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        # Stating a plan choice is not the same as asking about one.
        ("i need the basic plan", "high_intent"),
        ("I want to know about the basic plan", "product_query"),
        # A question that only mentions buying is still a question.
        ("do I need to pay extra for captions?", "product_query"),
        ("what can I buy here?", "product_query"),
        # A question about how to act is intent.
        ("how can I register?", "high_intent"),
        # Commitment beats a trailing question.
        ("sign me up, what's next?", "high_intent"),
        # Product vocabulary beats a polite opener.
        ("hello, how much is pro?", "product_query"),
    ],
)
def test_precedence_between_rules(message, expected):
    assert ic._keyword_fallback(message) == expected


@pytest.mark.parametrize(
    "message,expected",
    [
        ("i wnat to subscirbe today", "high_intent"),
        ("refnd rules?", "product_query"),
        ("heyyy", "greeting"),
        ("Let\u2019s do it", "high_intent"),
        ("THANK YOU", "greeting"),
    ],
)
def test_normalisation_handles_typos_case_and_curly_apostrophes(message, expected):
    assert ic._keyword_fallback(message) == expected


@pytest.mark.parametrize("word", ["history", "thistle", "shipping", "paying attention"])
def test_keywords_do_not_fire_inside_other_words(word):
    assert ic._keyword_fallback(word) == "product_query"


# --- LLM label validation --------------------------------------------------


@pytest.mark.parametrize("label", ["greeting", "product_query", "high_intent"])
def test_each_allowed_llm_label_is_trusted_over_keywords(monkeypatch, label):
    _stub_llm(monkeypatch, content=label)

    assert ic.classify_intent("zzz") == label


@pytest.mark.parametrize(
    "bad_label",
    ["sales", "high intent", "Label: greeting", "high_intent.", "", "I cannot classify that"],
)
def test_unexpected_llm_label_falls_back_to_keywords(monkeypatch, bad_label):
    _stub_llm(monkeypatch, content=bad_label)

    assert ic.classify_intent("what does it cost?") == "product_query"
    assert ic.classify_intent("sign me up") == "high_intent"


def test_null_llm_content_falls_back_to_keywords(monkeypatch):
    _stub_llm(monkeypatch, content=None)

    assert ic.classify_intent("thanks") == "greeting"
