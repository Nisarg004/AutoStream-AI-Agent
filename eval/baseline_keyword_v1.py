"""Frozen copy of the keyword fallback as it stood at commit 309d6c5 ("v1").

This is the BASELINE for eval/dev.jsonl and eval/heldout.jsonl: the rules that
were live when those sets were written. It must never be "fixed". The older,
pre-rewrite rules live in baseline_keyword.py ("v0").

To check it is still faithful:

    git show 309d6c5:utils/intent_classifier.py | sed -n '38,120p'
"""
import re

# Buying language, grouped the way people actually express it. Plan names are
# deliberately absent: "the pro plan" names a topic, not an intention, so
# "how much is the pro plan?" stays a product_query.
_SIGNUP = (
    "sign up", "sign me up", "signup", "set me up", "register",
    "create an account", "make an account", "open an account",
    "get started", "getting started",
)
_PURCHASE = (
    "buy", "purchase", "subscribe", "want to try", "like to try",
    "upgrade to", "upgrade me",
)
_PAYMENT = (
    "pay", "payment", "checkout", "card details", "credit card",
)
_COMMITMENT = (
    "i'm in", "im in", "i am in", "count me in", "let's do it", "lets do it",
    "go ahead", "i'll take", "ill take", "i will take", "take my money",
    "put me down",
)

# Product vocabulary: what the knowledge base is about. Plan names live here.
_PRODUCT = (
    "price", "prices", "pricing", "cost", "costs", "how much", "how many",
    "plan", "plans", "tier", "tiers", "package", "packages",
    "feature", "features", "include", "included", "includes", "difference",
    "resolution", "quality", "720p", "1080p", "4k", "8k",
    "caption", "captions", "subtitle", "subtitles",
    "video", "videos", "export", "exports", "upload", "uploads",
    "refund", "refunds", "money back", "policy", "policies", "cancel",
    "trial", "limit", "limits", "unlimited", "support",
    "pro", "basic",
)

# Openers and pleasantries. Previously there was no such list at all --
# "greeting" was whatever fell through, which is why it never appeared to miss.
_GREETING = (
    "hi", "hello", "hey", "heya", "hiya", "yo", "sup", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "morning", "afternoon", "evening",
    "thanks", "thank you", "thx", "cheers", "appreciate it",
    "bye", "goodbye", "see you", "talk later", "take care", "have a good",
    "how are you", "how's it going", "hows it going", "what's up", "whats up",
    "nice to meet", "no worries", "lol", "haha", "hehe", "who are you",
    # "sure" is deliberately absent: "not sure what I need" is a question in
    # disguise, and matching it as a pleasantry sends it to the greeting node.
    "ok", "okay", "cool", "nice", "great", "awesome", "got it",
)


def _compile(patterns: tuple[str, ...]) -> re.Pattern:
    """Match any pattern on word boundaries, so 'hi' cannot fire inside 'this'."""
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in patterns) + r")\b")


_BUYING_RE = _compile(_SIGNUP + _PURCHASE + _PAYMENT + _COMMITMENT)
_PRODUCT_RE = _compile(_PRODUCT)
_GREETING_RE = _compile(_GREETING)


def _keyword_fallback(message: str) -> str:
    """Rule-based classifier used when the LLM call fails.

    Checked in order: an explicit move to buy, then product vocabulary, then
    conversational openers. Buying language is checked first so that "can I
    purchase the basic plan?" reads as intent despite being a question, while
    a plan name on its own never does.

    Anything unrecognised is treated as a product_query rather than a
    greeting. An unclassified message in a support chat is far more likely to
    be a question than a hello; answering from the knowledge base is the most
    useful way to be wrong, and unlike a high_intent guess it never starts
    demanding a stranger's name and email.
    """
    message = message.lower()

    if _BUYING_RE.search(message):
        return "high_intent"
    if _PRODUCT_RE.search(message):
        return "product_query"
    if _GREETING_RE.search(message):
        return "greeting"
    return "product_query"
