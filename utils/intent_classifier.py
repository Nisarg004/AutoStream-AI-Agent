import difflib
import os
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# The placeholder lets this module import without a key. A call made with it
# fails authentication, and the callers already handle that failure.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "missing-api-key")

ALLOWED_INTENTS = frozenset({"greeting", "product_query", "high_intent"})


def classify_intent(message: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user message into exactly one of: "
                        "greeting, product_query, high_intent.\n"
                        "- greeting: hellos, hi, hey, thanks, casual conversation\n"
                        "- product_query: questions about pricing, features, plans, policies\n"
                        "- high_intent: user wants to sign up, buy, try, start, or subscribe\n"
                        "Return only the label, nothing else. No explanation."
                    ),
                },
                {"role": "user", "content": message},
            ],
            max_tokens=10,
            temperature=0,
        )
        label = response.choices[0].message.content.strip().lower()
    except Exception:
        return _keyword_fallback(message)
    # An unexpected label must not reach the router.
    return label if label in ALLOWED_INTENTS else _keyword_fallback(message)


# All patterns are written without apostrophes, because _normalise removes them.

# Commitment: unambiguous even inside a question.
_STRONG_BUYING = (
    "sign me up", "set me up", "get me on", "get me started", "put me down",
    "count me in", "im in", "i am in", "take my money", "take my card",
    "take my payment",
    "lets do it", "lets do this", "lets start", "lets begin",
    "go ahead", "ill take", "i will take",
    "want to buy", "want to purchase", "want to subscribe", "want to sign up",
    "want to try", "want to get started", "want to start",
    "like to buy", "like to purchase", "like to subscribe", "like to sign up",
    "like to try", "like to get started", "like to start",
    "upgrade me", "upgrade to",
)
# "I want the Pro plan" states a choice; "I want to know about Pro" does not.
_CHOOSING_PLAN_RE = re.compile(
    r"\b(?:i|we)(?: would| d)? (?:want|need|like)(?: to (?:get|have))? (?:the |a )?(?:pro|basic)\b"
)
# Buying words that a question can merely mention: "before I buy, what's ...?"
_WEAK_BUYING = (
    "sign up", "signup", "register", "create an account", "make an account",
    "open an account", "get started", "getting started",
    "buy", "purchase", "subscribe",
    "pay", "payment", "checkout", "card details", "credit card",
)
# Questions that ask how to act rather than what the product does.
_ACTION_QUESTION_RE = re.compile(
    r"^(?:(?:ok|okay|so|hi|hey|and) )*"
    r"(?:how (?:do|can|should) (?:i|we)|where (?:do|can|should) (?:i|we)|"
    r"(?:can|could|may) (?:i|we)|who do (?:i|we))\b"
)

_PRODUCT = (
    "price", "prices", "pricing", "cost", "costs", "how much", "how many",
    "plan", "plans", "tier", "tiers", "package", "packages",
    "feature", "features", "include", "included", "includes", "difference",
    "resolution", "quality", "720p", "1080p", "4k", "8k",
    "caption", "captions", "subtitle", "subtitles",
    "video", "videos", "export", "exports", "upload", "uploads",
    "refund", "refunds", "money back", "policy", "policies", "cancel",
    "trial", "limit", "limits", "unlimited", "support", "billing",
    "pro", "basic",
)

_GREETING = (
    "hi", "hello", "hey", "heya", "hiya", "yo", "sup", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "morning", "afternoon", "evening",
    "thanks", "thank you", "thx", "ty", "cheers", "appreciate it",
    "bye", "goodbye", "see you", "see ya", "talk later", "talk soon",
    "take care", "have a good",
    "how are you", "hows it going", "whats up",
    "nice to meet", "no worries", "lol", "haha", "hehe", "who are you",
    "are you a bot", "are you an ai", "are you human", "your name",
    # "sure" is deliberately absent: "not sure what I need" is a question in
    # disguise, and matching it as a pleasantry sends it to the greeting node.
    "ok", "okay", "cool", "nice", "great", "awesome", "got it",
)

_INTERROGATIVE_RE = re.compile(
    r"^(?:what|whats|how|which|why|when|where|who|is|are|does|do|did|can|could|"
    r"will|would|should|any)\b"
)
_PLAN_RE = re.compile(
    r"\b(basic|pro)\b(?! (?:question|questions|info|information|tip|tips|stuff|level)\b)"
)


def _compile(patterns: tuple[str, ...]) -> re.Pattern:
    """Match any pattern on word boundaries, so 'hi' cannot fire inside 'this'."""
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in patterns) + r")\b")


_STRONG_RE = _compile(_STRONG_BUYING)
_WEAK_RE = _compile(_WEAK_BUYING)
_PRODUCT_RE = _compile(_PRODUCT)
_GREETING_RE = _compile(_GREETING)

_VOCABULARY = frozenset(
    word
    for patterns in (_STRONG_BUYING, _WEAK_BUYING, _PRODUCT, _GREETING)
    for pattern in patterns
    for word in pattern.split()
)
_FUZZY_TARGETS = sorted(word for word in _VOCABULARY if len(word) >= 5)


def _correct_token(token: str) -> str:
    """Repair common typos of known words; leave everything else alone."""
    if token in _VOCABULARY:
        return token
    # Elongation: "hii", "heyy".
    squeezed = re.sub(r"(.)\1+$", r"\1", token)
    if squeezed in _VOCABULARY:
        return squeezed
    # One adjacent swap: "wnat", "subscirbe".
    if len(token) >= 4:
        for i in range(len(token) - 1):
            swapped = token[:i] + token[i + 1] + token[i] + token[i + 2 :]
            if swapped in _VOCABULARY:
                return swapped
    # A dropped or changed letter in a longer word: "refnd", "captons".
    if len(token) >= 5:
        close = difflib.get_close_matches(token, _FUZZY_TARGETS, n=1, cutoff=0.85)
        if close:
            return close[0]
    return token


def _normalise(message: str) -> str:
    message = message.lower().replace("’", "").replace("'", "")
    return " ".join(_correct_token(token) for token in re.findall(r"[a-z0-9/]+", message))


def _is_question(message: str, normalised: str) -> bool:
    return "?" in message or bool(_INTERROGATIVE_RE.match(normalised))


def _keyword_fallback(message: str) -> str:
    """Rule-based classifier used when the LLM call fails or returns a bad label.

    Precedence, first match wins:

    1. Commitment, or naming a plan as a choice -> high_intent.
    2. A buying word outside a question, or in a question about how to act
       ("how do I sign up?") -> high_intent.
    3. Product vocabulary, including a buying word that a question only
       mentions -> product_query.
    4. Conversational phrases -> greeting.
    5. Anything else -> product_query. Answering from the knowledge base is the
       most useful way to be wrong, and unlike a high_intent guess it never
       asks a stranger for their name and email.
    """
    normalised = _normalise(message)

    if _STRONG_RE.search(normalised) or _CHOOSING_PLAN_RE.search(normalised):
        return "high_intent"
    if _WEAK_RE.search(normalised):
        if not _is_question(message, normalised) or _ACTION_QUESTION_RE.search(normalised):
            return "high_intent"
        return "product_query"
    if _PRODUCT_RE.search(normalised):
        return "product_query"
    if _GREETING_RE.search(normalised):
        return "greeting"
    return "product_query"


def is_product_question(message: str) -> bool:
    """True if a message sent during lead collection is a question, not an answer.

    A name or platform can contain product words or start like a question ("Will
    Smith", "Pro Tools"), so one signal is not enough: this needs a question mark
    or an interrogative opener together with product vocabulary, or both of those.
    """
    normalised = _normalise(message)
    has_mark = "?" in message
    has_opener = bool(_INTERROGATIVE_RE.match(normalised))
    if has_mark and has_opener:
        return True
    return (has_mark or has_opener) and bool(_PRODUCT_RE.search(normalised))


def extract_plan(message: str) -> str | None:
    """The single plan a message names, or None if it names none or both."""
    plans = {match.capitalize() for match in _PLAN_RE.findall(_normalise(message))}
    return plans.pop() if len(plans) == 1 else None
