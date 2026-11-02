import os
import re

from openai import OpenAI
from dotenv import load_dotenv

from agent.state import AgentState
from rag.retriever import retrieve
from utils.intent_classifier import classify_intent
from tools.lead_capture import mock_lead_capture

load_dotenv()

# The placeholder lets this module import without a key. A call made with it
# fails authentication, and the callers already handle that failure.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "missing-api-key")
_MODEL = "gpt-4o-mini"

# Deliberately permissive: rejects obvious typos (missing @, missing TLD,
# embedded spaces) without turning away unusual but valid addresses.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")

# Used when the API is unreachable. classify_intent already degrades to keyword
# matching, but that only helps if the node it routes to can still answer --
# both of these nodes call the LLM again, and an unguarded call would take the
# whole turn down exactly when the fallback was supposed to save it.
_FALLBACK_GREETING = "Hello! I'm the AutoStream assistant. How can I help you today?"
_ANSWER_UNAVAILABLE = (
    "I can't reach the answer service right now — please try again shortly."
)

# How many past messages each LLM call sees: five exchanges.
_HISTORY_WINDOW = 10


def _recent_history(state: AgentState) -> list[dict]:
    return list(state.get("chat_history") or [])[-_HISTORY_WINDOW:]


def _exchange(user_message: str, reply: str) -> list[dict]:
    return [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]


# ---------------------------------------------------------------------------
# Node 1: Intent Classification
# ---------------------------------------------------------------------------

def intent_node(state: AgentState) -> dict:
    """
    Classifies user intent.
    If we are mid lead-collection, skip reclassification and keep high_intent.
    For greetings, sets the response here so the graph can terminate early.
    """
    # If already collecting lead, don't reclassify – preserve flow
    active_steps = ("ask_name", "ask_email", "ask_platform")
    if state.get("lead_collection_step") in active_steps:
        print(f"[IntentNode] Mid lead-collection, keeping high_intent")
        return {"intent": "high_intent"}

    intent = classify_intent(state["user_message"])

    update: dict = {"intent": intent}

    if intent == "greeting":
        try:
            response = client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly assistant for AutoStream, a video editing SaaS. "
                            "Respond warmly to greetings."
                        ),
                    },
                    *_recent_history(state),
                    {"role": "user", "content": state["user_message"]},
                ],
                max_tokens=100,
                temperature=0.7,
            )
            reply = response.choices[0].message.content.strip()
            print(f"[IntentNode] Greeting detected — responding directly")
        except Exception:
            reply = _FALLBACK_GREETING
            print("[IntentNode] Greeting detected — LLM unavailable, canned reply")
        update["response"] = reply
        update["chat_history"] = _exchange(state["user_message"], reply)

    return update


# ---------------------------------------------------------------------------
# Node 2: RAG (product queries)
# ---------------------------------------------------------------------------

def rag_node(state: AgentState) -> dict:
    """Retrieves relevant knowledge and generates a grounded answer.

    If the answer call fails, the retrieved text is handed over as-is rather
    than dropped: unphrased facts are more use to the reader than an error.
    """
    context = retrieve(state["user_message"])
    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for AutoStream. "
                        "Answer using only the context provided. Be concise and accurate. "
                        "Earlier messages are there only to resolve references such as 'it'."
                    ),
                },
                *_recent_history(state),
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {state['user_message']}",
                },
            ],
            max_tokens=300,
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()
        print(f"[RAGNode] Retrieved context and generated answer")
    except Exception:
        print("[RAGNode] Answer service unavailable — returning retrieved text")
        answer = _ANSWER_UNAVAILABLE
        if context.strip():
            answer = f"{_ANSWER_UNAVAILABLE}\n\nHere's what I have on file:\n{context}"
    return {"response": answer, "chat_history": _exchange(state["user_message"], answer)}


# ---------------------------------------------------------------------------
# Node 3: Lead Collection (multi-turn)
# ---------------------------------------------------------------------------

def lead_collection_node(state: AgentState) -> dict:
    """
    Manages the step-by-step lead capture dialogue.
    Reads the current step from state, stores the user's last answer,
    advances the step, and asks the next question.
    An answer that fails validation re-asks the same question instead of
    advancing, so the step and the fields collected so far are preserved.
    """
    step = state.get("lead_collection_step")
    user_msg = state["user_message"].strip()
    update: dict = {}

    if step is None:
        # First entry: user just expressed high intent
        update["lead_collection_step"] = "ask_name"
        update["response"] = (
            "Great choice! I'd love to get you set up. "
            "Could you please share your full name?"
        )
        print("[LeadCollectionNode] Starting lead flow — asking for name")

    elif step == "ask_name":
        if not user_msg:
            print("[LeadCollectionNode] Empty name — re-asking")
            return {
                "response": "I didn't catch that — could you share your full name?"
            }
        update["lead_name"] = user_msg
        update["lead_collection_step"] = "ask_email"
        update["response"] = f"Thanks, {user_msg}! What's your email address?"
        print(f"[LeadCollectionNode] Name collected: {user_msg}")

    elif step == "ask_email":
        if not _EMAIL_RE.match(user_msg):
            print("[LeadCollectionNode] Invalid email — re-asking")
            return {
                "response": (
                    "That doesn't look like a valid email address. "
                    "Could you double-check and send it again?"
                )
            }
        update["lead_email"] = user_msg
        update["lead_collection_step"] = "ask_platform"
        update["response"] = (
            "Perfect! Last question — which creator platform do you use? "
            "(e.g. YouTube, Instagram, TikTok, etc.)"
        )
        print(f"[LeadCollectionNode] Email collected: {user_msg}")

    elif step == "ask_platform":
        if not user_msg:
            print("[LeadCollectionNode] Empty platform — re-asking")
            return {
                "response": (
                    "I didn't catch that — which creator platform do you use? "
                    "(e.g. YouTube, Instagram, TikTok, etc.)"
                )
            }
        update["lead_platform"] = user_msg
        update["lead_collection_step"] = "complete"
        print(f"[LeadCollectionNode] Platform collected: {user_msg}")

    return update


# ---------------------------------------------------------------------------
# Node 4: Tool Execution (fires only when all fields are ready)
# ---------------------------------------------------------------------------

def tool_execution_node(state: AgentState) -> dict:
    """Validates all lead fields then calls mock_lead_capture."""
    # `or ""` guards against a field that is present but None, which .get's
    # default does not cover.
    name = (state.get("lead_name") or "").strip()
    email = (state.get("lead_email") or "").strip()
    platform = (state.get("lead_platform") or "").strip()

    if not all([name, email, platform]):
        return {
            "response": "Something went wrong — I'm missing some details. Let's start over.",
            "lead_collection_step": None,
            "lead_name": None,
            "lead_email": None,
            "lead_platform": None,
            "lead_captured": False,
        }

    mock_lead_capture(name, email, platform)
    print(f"[ToolExecutionNode] Lead capture executed for {name}")

    # Reset the flow so a later high-intent message starts fresh instead of
    # re-routing here (step == "complete") and capturing the same lead again.
    return {
        "response": (
            f"You're all set, {name}! Your interest in the Pro plan has been recorded. "
            "Our team will reach out to you shortly. Welcome to AutoStream!"
        ),
        "lead_captured": True,
        "lead_collection_step": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
    }
