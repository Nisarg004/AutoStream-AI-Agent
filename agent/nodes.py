import os
import re

from openai import OpenAI
from dotenv import load_dotenv

from agent.state import AgentState
from rag.retriever import retrieve
from utils.intent_classifier import classify_intent
from tools.lead_capture import mock_lead_capture

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = "gpt-4o-mini"

# Deliberately permissive: rejects obvious typos (missing @, missing TLD,
# embedded spaces) without turning away unusual but valid addresses.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


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
                {"role": "user", "content": state["user_message"]},
            ],
            max_tokens=100,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        update["response"] = reply
        print(f"[IntentNode] Greeting detected — responding directly")

    return update


# ---------------------------------------------------------------------------
# Node 2: RAG (product queries)
# ---------------------------------------------------------------------------

def rag_node(state: AgentState) -> dict:
    """Retrieves relevant knowledge and generates a grounded answer."""
    context = retrieve(state["user_message"])
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for AutoStream. "
                    "Answer using only the context provided. Be concise and accurate."
                ),
            },
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
    return {"response": answer}


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
    name = state.get("lead_name", "").strip()
    email = state.get("lead_email", "").strip()
    platform = state.get("lead_platform", "").strip()

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
