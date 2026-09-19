# AutoStream AI Agent — Social Lead Capture

Conversational AI agent that qualifies SaaS leads from social/messaging channels. Engages users, answers product questions via FAISS RAG, and collects leads through multi-turn dialog — built with LangGraph, OpenAI GPT-4o Mini & sentence-transformers.



## Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (StateGraph) |
| LLM | OpenAI GPT-4o Mini (`openai`) |
| Retrieval | FAISS + sentence-transformers |
| Lead storage | CSV (`leads.csv`) |
| Config | python-dotenv |



## Setup and tests

**Tested on Python 3.13.9.** Other versions are untested.

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
cp .env.example .env               # then edit .env and paste your key

# Run the agent
python main.py

# Run the tests
python -m pytest tests
```

The tests need no API key: every OpenAI call is stubbed, and `conftest.py`
forces a dummy key so a real one in `.env` cannot leak into a run.

The first run that touches retrieval downloads the `all-MiniLM-L6-v2`
embedding model (~90 MB) and caches it locally.



## Architecture

### Why LangGraph & How State is Managed

LangGraph was chosen over plain LangChain because the lead capture flow is inherently multi-step and stateful — a simple chain cannot pause mid-conversation, wait for user input, and resume from the exact same point. LangGraph models the agent as a directed graph where each node is a discrete processing step, and conditional edges route execution based on runtime state (e.g., detected intent, which lead field is still missing). This makes the control flow explicit, testable, and easy to extend.

State is managed through a single `AgentState` TypedDict that flows through every node. Each node receives the full state and returns only the fields it modifies — LangGraph merges these partial updates back automatically. The graph itself is stateless per-invocation; the CLI loop owns a persistent `AgentState` dict and merges each graph result back into it, giving the agent memory across turns without requiring an external store. This design also makes WhatsApp integration straightforward: swapping the CLI loop for a webhook handler and keying state by phone number is all that changes.

### File Map

```
main.py                   CLI loop — persistent AgentState across turns
agent/
  state.py                TypedDict schema shared across all graph nodes
  graph.py                LangGraph StateGraph — nodes + conditional routing
  nodes.py                4 nodes: intent, rag, lead_collection, tool_execution
utils/
  intent_classifier.py    OpenAI GPT-4o Mini classifier with keyword-matching fallback
rag/
  retriever.py            FAISS index over knowledge.json (sentence-transformers)
  knowledge.json          Plans + policies knowledge base
tools/
  lead_capture.py         mock_lead_capture() — fires only when all 3 fields collected
tests/                    pytest suite — OpenAI calls stubbed, no API key needed
leads.csv                 Auto-generated on first lead capture
```

### Agent State

```python
class AgentState(TypedDict):
    user_message: str
    chat_history: list[dict]
    response: str
    intent: Optional[Literal["greeting", "product_query", "high_intent"]]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_collection_step: Optional[Literal["ask_name", "ask_email", "ask_platform", "complete"]]
    lead_captured: bool
```

### Graph Flow

```
[intent] ──greeting──────> END
         ──product_query──> [rag] ──> END
         ──high_intent────> [lead_collection] ──incomplete──> END
                                               ──complete───> [tool_execution] ──> END
```

The graph is stateless per-invocation. The CLI loop merges each result back into a persistent `AgentState` dict to maintain multi-turn context across the conversation.

### Node Responsibilities

| Node | Responsibility |
|---|---|
| `intent_node` | Classifies message as `greeting`, `product_query`, or `high_intent`. Generates greeting response inline. Skips reclassification mid lead-collection. |
| `rag_node` | Retrieves relevant chunks from FAISS index, generates a grounded answer using only that context. |
| `lead_collection_node` | Steps through name → email → platform collection. Each turn validates one answer and stores it, or re-asks on invalid input. |
| `tool_execution_node` | Re-checks all three fields, calls `mock_lead_capture()`, appends lead to `leads.csv`, then resets the flow. |

---

## Lead Capture

Each answer is validated before the flow advances:

| Field | Rule |
|---|---|
| Name | Non-empty |
| Email | Must match `^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$` — catches a missing `@`, a missing TLD and embedded spaces |
| Platform | Non-empty |

Invalid input re-asks the same question and does not advance the step, so
nothing already collected is lost. `tool_execution_node` re-checks all three
before firing the tool and resets the flow if any is missing.

Once all three fields (name, email, platform) are collected, the agent fires `mock_lead_capture()` which:

- Prints a confirmation to the terminal
- Appends a row to `leads.csv` with a timestamp

The flow then resets, so a later high-intent message starts a fresh lead
rather than re-capturing the previous one.

```
Name,Email,Platform,Timestamp
Jane Doe,jane@example.com,YouTube,2026-04-13 01:00:00
```



## WhatsApp Integration

The LangGraph pipeline is transport-agnostic — connecting it to WhatsApp requires only a thin wrapper:

**What changes:**
- Replace the `input()` loop in `main.py` with a Flask/FastAPI webhook receiving `POST /webhook` from Twilio
- Use the `from` phone number as a session key to look up and persist `AgentState` between messages
- Send `state["response"]` back via `twilio.rest.Client.messages.create()`
- Move state storage from in-memory dict to Redis or a database (keyed by phone number)

**What stays the same:**
The entire LangGraph graph — intent classification, RAG retrieval, multi-turn lead collection, and `mock_lead_capture` — is reused without modification.

**Minimal additions:** `twilio`, `flask` or `fastapi`, `redis` or `sqlitedict`



## Demo

[Watch screen recording on Google Drive](https://drive.google.com/file/d/1hn51d-naZfjbfTPJjk6LP7OPRV9GiifU/view?usp=sharing)



## License

MIT
