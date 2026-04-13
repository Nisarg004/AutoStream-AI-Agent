"""
AutoStream — Social-to-Lead Agentic Workflow
CLI-based conversational agent powered by LangGraph + OpenAI GPT-4o Mini.
"""

import os
import warnings
import logging

# Suppress HuggingFace / transformers noise before any heavy imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from dotenv import load_dotenv

load_dotenv()

from agent.graph import graph
from agent.state import AgentState


def run_chat() -> None:
    print("=" * 60)
    print("  AutoStream AI Assistant")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    # Persistent state across turns
    state: AgentState = {
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

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # Inject new user message; preserve the rest of the state
        state["user_message"] = user_input

        # Run one turn through the graph
        result = graph.invoke(state)

        # Merge result back into persistent state
        state.update(result)

        # Append this turn to history
        state["chat_history"].append({"role": "user", "content": user_input})
        state["chat_history"].append({"role": "assistant", "content": state["response"]})

        print(f"\nAssistant: {state['response']}")


if __name__ == "__main__":
    run_chat()
