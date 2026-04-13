# AutoStream AI Agent — Social Lead Capture

Conversational AI agent that qualifies SaaS leads from social/messaging channels. Engages users, answers product questions via FAISS RAG, and collects leads through multi-turn dialog — built with LangGraph, OpenAI GPT-4o Mini & sentence-transformers.



## Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (StateGraph) |
| LLM | OpenAI GPT-4o Mini (`openai`, `langchain-openai`) |
| Retrieval | FAISS + sentence-transformers |
| Lead storage | CSV (`leads.csv`) |
| Config | python-dotenv |



## Setup

**Prerequisites:** Python 3.10+

```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Run the agent
python main.py
```



## Architecture

```
main.py                   CLI loop — persistent AgentState across turns
agent/
  state.py                TypedDict schema shared across all graph nodes
  graph.py                LangGraph StateGraph — nodes + conditional routing
  nodes.py                4 nodes: intent, rag, lead_collection, tool_execution
utils/
  intent_classifier.py    OpenAI GPT-4o Mini classifier with keyword regex fallback
rag/
  retriever.py            FAISS index over knowledge.json (sentence-transformers)
  knowledge.json          Plans + policies knowledge base
tools/
  lead_capture.py         mock_lead_capture() — fires only when all 3 fields collected
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
| `lead_collection_node` | Steps through name → email → platform collection. Each turn stores one field and asks for the next. |
| `tool_execution_node` | Validates all three fields, calls `mock_lead_capture()`, appends lead to `leads.csv`. |

---

## Lead Capture

Once all three fields (name, email, platform) are collected, the agent fires `mock_lead_capture()` which:

- Prints a confirmation to the terminal
- Appends a row to `leads.csv` with a timestamp

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
