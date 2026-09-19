"""Frozen copy of the keyword fallback as it stood before the rewrite.

Extracted verbatim from `git show c5ff08d:utils/intent_classifier.py`, the
commit that recorded the baseline, so eval/RESULTS.md can report before and
after from one run instead of quoting numbers from memory. Nothing imports
this except run_eval.py, and it must never be "fixed" -- its whole value is
being the old behaviour.

To check it is still faithful:

    git show c5ff08d:utils/intent_classifier.py | sed -n '36,51p'
"""


def keyword_fallback_before(message: str) -> str:
    message = message.lower()
    high_intent = [
        "sign up", "buy", "try", "subscribe", "pro plan",
        "basic plan", "want to", "start", "get started", "purchase",
    ]
    product_query = [
        "price", "plan", "feature", "cost", "support",
        "refund", "resolution", "caption", "video", "how much", "what is",
    ]

    if any(k in message for k in high_intent):
        return "high_intent"
    if any(k in message for k in product_query):
        return "product_query"
    return "greeting"
