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
eval/                     Intent classification accuracy check — needs an API key
leads.csv                 Auto-generated on first lead capture
```

### Agent State

```python
class AgentState(TypedDict):
    user_message: str
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



## Evaluation

```bash
python eval/run_eval.py     # writes eval/RESULTS.md
```

Scores the real `gpt-4o-mini` call and the keyword fallback — both the current
rules and the pre-rewrite rules frozen in
[eval/baseline_keyword.py](eval/baseline_keyword.py) — over two datasets.
Prints overall accuracy, per-label accuracy and a confusion table for each,
and writes the full breakdown with every misclassified message to
[eval/RESULTS.md](eval/RESULTS.md).

There are two datasets, and the difference between them matters:

- [eval/intents.json](eval/intents.json) — 36 messages, 12 per label. Its
  misses were read and used to rewrite the fallback, so its score is a
  description of the fix, not evidence for it.
- [eval/intents_heldout.json](eval/intents_heldout.json) — 30 messages, 10 per
  label, written and committed *before* the rules were touched, with phrasings
  that avoid the vocabulary the first set leans on. **This is the honest
  number.**

Latest run (2026-09-19), 0 API errors:

| Dataset | LLM | Keyword before | Keyword after |
|---|---|---|---|
| Original 36 | 36/36 | 25/36 (69.4%) | 36/36 (100%) |
| Held-out 30 | 30/30 | 14/30 (46.7%) | 26/30 (86.7%) |

On the held-out set the rewrite moved `product_query` from 3/10 to 10/10 and
`high_intent` from 1/10 to 9/10. `greeting` went the other way, 10/10 to 7/10:
the old rules scored perfectly there only because "greeting" was the silent
default that everything unrecognised fell into. That trade is deliberate —
misreading a question as a greeting is worse than the reverse. The four
remaining misses are listed in [eval/RESULTS.md](eval/RESULTS.md) and were not
tuned away after being seen.

Three caveats worth stating plainly:

- Both datasets are small and hand-written, **by the same author as the
  rules**. A held-out set written by someone else would be a stronger test.
- The LLM scores 100% on both sets. On 66 easy, hand-written messages that
  means "no obvious failure", not a quality claim.
- **`eval/` needs a real `OPENAI_API_KEY`; the test suite does not.** The eval
  calls the API directly rather than through `classify_intent`, because that
  function catches every exception and falls back to keywords — an outage or a
  dead key would otherwise score as a healthy LLM result instead of an error.

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
