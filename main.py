"""
AutoStream — Social-to-Lead Agentic Workflow
CLI-based conversational agent powered by LangGraph + OpenAI GPT-4o Mini.
"""

import os
import warnings
import logging
import uuid

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


def run_chat() -> None:
    print("=" * 60)
    print("  AutoStream AI Assistant")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    # The checkpointer keeps state between turns, keyed by this thread.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

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

        result = graph.invoke({"user_message": user_input}, config)

        print(f"\nAssistant: {result['response']}")


if __name__ == "__main__":
    run_chat()
