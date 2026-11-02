# AutoStream AI Agent — Social Lead Capture

Conversational AI agent that qualifies SaaS leads from social/messaging channels. Engages users, answers product questions via FAISS RAG, and collects leads through multi-turn dialog — built with LangGraph, OpenAI GPT-4o Mini & sentence-transformers.



## Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (StateGraph, MemorySaver checkpointer) |
| LLM | OpenAI GPT-4o Mini (`openai`), for intent classification, greetings and RAG answers |
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

The tests and `eval/run_eval.py` need no API key. Every OpenAI call in the tests
is mocked, and `conftest.py` forces a dummy key so a real one in `.env` cannot leak
into a run.

The first run that touches retrieval downloads the `all-MiniLM-L6-v2`
embedding model (~90 MB) and caches it locally.



## Architecture

### Why LangGraph & How State is Managed

LangGraph was chosen over plain LangChain because the lead capture flow is inherently multi-step and stateful — a simple chain cannot pause mid-conversation, wait for user input, and resume from the exact same point. LangGraph models the agent as a directed graph where each node is a discrete processing step, and conditional edges route execution based on runtime state (e.g., detected intent, which lead field is still missing). This makes the control flow explicit, testable, and easy to extend.

State is managed through a single `AgentState` TypedDict that flows through every node. Each node receives the full state and returns only the fields it modifies — LangGraph merges these partial updates back automatically. The graph is compiled with LangGraph's `MemorySaver` checkpointer, so each turn sends only the new message with a `thread_id` and state is restored from the checkpoint. `MemorySaver` is in-process: state is lost when the process exits. This design also makes WhatsApp integration straightforward: use the phone number as the `thread_id` and swap in a persistent checkpointer.

### File Map

```
main.py                   CLI loop — one thread_id per session
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
tests/                    pytest suite — OpenAI calls mocked, no API key needed
eval/                     Datasets, offline harness, recorded results and provenance
.github/workflows/ci.yml  Runs pytest and the offline eval
leads.csv                 Auto-generated on first lead capture
```

### Agent State

```python
class AgentState(TypedDict):
    user_message: str
    response: str
    chat_history: Annotated[list[dict], operator.add]  # greeting and RAG exchanges only
    intent: Optional[Literal["greeting", "product_query", "high_intent"]]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_plan: Optional[str]
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

A product question during lead collection goes to `rag`, keeps the collected fields and repeats the pending question. An unrecognised intent also goes to `rag`, never to lead collection.

### Node Responsibilities

| Node | Responsibility |
|---|---|
| `intent_node` | Classifies message as `greeting`, `product_query`, or `high_intent`. Generates greeting response inline. Mid lead-collection it uses deterministic rules instead: a product question goes to RAG and anything else is the pending answer. |
| `rag_node` | Retrieves relevant chunks from FAISS index, generates an answer using that context and recent chat history. |
| `lead_collection_node` | Steps through name → email → platform collection. Each turn validates one answer and stores it, or re-asks on invalid input. Handles cancelling. |
| `tool_execution_node` | Re-checks all three fields, including the email format, calls `mock_lead_capture()`, appends lead to `leads.csv`, then resets the flow. |

### When the API is unreachable

Every LLM call has a fallback, so a turn still produces a reply:

| Step | Degrades to |
|---|---|
| `classify_intent` | Keyword matching (`_keyword_fallback`) |
| `intent_node` greeting | A canned greeting |
| `rag_node` answer | The retrieved knowledge-base text, with a note that the answer service is down |
| Lead collection and tool execution | Unaffected — neither calls the LLM |

The replies are blunter and intent labelling is cruder, but nothing raises. A
lead already in progress completes normally, since `intent_node` skips
classification entirely mid-flow.

### Tool execution is deterministic

This project does **not** use OpenAI function calling. `mock_lead_capture()` runs
only after deterministic routing and validation. Letting the model trigger it would
add no capability here, would send personal details to OpenAI, and would stop lead
capture working when the API is down.

---

## Lead Capture

Each answer is validated before the flow advances:

| Field | Rule |
|---|---|
| Name | Non-empty |
| Email | Must match `^[^@\s]+@[^@\s.]+(?:\.[^@\s.]+)*\.[A-Za-z]{2,}$` — catches a missing `@`, a missing TLD, doubled dots and embedded spaces |
| Platform | Non-empty |

Invalid input re-asks the same question and does not advance the step, so
nothing already collected is lost. `tool_execution_node` re-checks all three
before firing the tool and resets the flow if any is missing.

Once all three fields (name, email, platform) are collected, the agent fires `mock_lead_capture()` which:

- Prints a confirmation to the terminal
- Appends a row to `leads.csv` with a timestamp

The closing message names the plan the user chose or asked about, or names no plan
if none is known. The plan is not written to `leads.csv`.

The flow then resets, so a later high-intent message starts a fresh lead
rather than re-capturing the previous one.

Saying `cancel`, `never mind`, `forget it` or similar during collection discards the
lead. The whole message must be the escape, so "what if I cancel my plan?" stays a
product question.

```
Name,Email,Platform,Timestamp
Jane Doe,jane@example.com,YouTube,2026-04-13 01:00:00
```



## Results

Run on 2026-11-03. The classifier scored here is the rule-based fallback used when
the LLM call fails or returns an invalid label. It is not the LLM itself.

| Rules | Held-out accuracy (n=30) | Wilson 95% CI |
|---|---|---|
| v0: original rules | 17/30 (56.7%) | 39.2%-72.6% |
| v1: first rewrite | 23/30 (76.7%) | 59.1%-88.2% |
| Current | 26/30 (86.7%) | 70.3%-94.7% |

- **Current vs v0:** +30.0 points, 11 fixed, 2 broken, exact McNemar p = 0.022.
- **Current vs v1:** +10.0 points, 3 fixed, 0 broken, exact McNemar p = 0.250. This
  step is **not statistically significant** at n=30.
- **Dev set (n=40, used for tuning):** v0 25/40, v1 29/40, current 39/40. The current
  score describes the tuning and is not evidence of general accuracy.
- **Tests:** 178 collected from 90 test functions; 177 pass and 1 is an expected
  failure that records a retriever weakness.

Reproduce:

```bash
python -m pytest
python eval/run_eval.py                       # held-out set, current rules
python eval/run_eval.py --set dev             # tuning set
python eval/run_eval.py --classifier v1       # or v0
python eval/compare.py                        # latest v1 -> current; --before v0 for v0
python eval/audit_v1_leakage.py               # evidence behind the old figure
```

`eval/run_eval.py` writes a timestamped file to `eval/results/`, recording the commit
and the dataset hash. `eval/PREDICTION.md` holds predictions written before scoring.

### Datasets

| File | Size | Use |
|---|---|---|
| `eval/dev.jsonl` | 40 | Tuning |
| `eval/heldout.jsonl` | 30, 10 per label | Scored once for the current rules, from a clean commit |
| `eval/intents.json` | 36 | Older set, used to write v1 |
| `eval/intents_heldout.json` | 30 | Older set, **no longer held-out** |

### Limitations

- **The older 46.7% → 86.7% is not a held-out result.** Those misses were committed
  minutes before v1 was written, and 24 v1 patterns occur only in that set. Without
  them v1 scores 12/30. See `eval/PROVENANCE.md`.
- **Same author.** I wrote the rules and the sets, knew v1's rules when writing
  the sets, and knew the held-out messages while tuning. A set written by someone
  else would be stronger evidence.
- **Small sample.** With n=30, one message moves accuracy by 3.3 points and the
  intervals are wide.
- **Near context-stuffing.** The knowledge base has 4 chunks and `top_k=3`, so
  retrieval removes only one chunk. "Which plan gives me 4K and unlimited videos?"
  ranks Basic above Pro, and the answer relies on `top_k=3` to include Pro.
- **Retrieval has no memory.** History reaches the LLM but not the retrieval query.
- **Mock LLM in tests.** Tests prove routing, state and validation, not answer
  quality. Retriever tests use the real embedding model.
- **LLM not re-scored.** `eval/run_llm_eval.py` needs a key and has not been run on
  the new sets. `eval/RESULTS.md` holds its earlier results on the older sets only.
- **Memory only.** `MemorySaver` loses state on exit. A captured lead goes to a local
  CSV through a mock function, not to a CRM.

## WhatsApp Integration

The LangGraph pipeline is transport-agnostic — connecting it to WhatsApp requires only a thin wrapper:

**What changes:**
- Replace the `input()` loop in `main.py` with a Flask/FastAPI webhook receiving `POST /webhook` from Twilio
- Use the `from` phone number as the `thread_id`
- Send `state["response"]` back via `twilio.rest.Client.messages.create()`
- Replace `MemorySaver` with a persistent checkpointer

**What stays the same:**
The entire LangGraph graph — intent classification, RAG retrieval, multi-turn lead collection, and `mock_lead_capture` — is reused without modification.

**Minimal additions:** `twilio`, `flask` or `fastapi`, and a persistent checkpointer backend



## Demo

[Watch screen recording on Google Drive](https://drive.google.com/file/d/1hn51d-naZfjbfTPJjk6LP7OPRV9GiifU/view?usp=sharing)



## License

MIT
