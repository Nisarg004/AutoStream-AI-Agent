# Intent classification results

- Run: 2026-09-19 20:47 local time
- Model: `gpt-4o-mini` (temperature 0, max_tokens 10)
- Dataset: `eval/intents.json`, 36 hand-written messages, 12 per label
- Prompt copy in sync with `utils/intent_classifier.py`: yes

36 hand-written messages is a sanity check, not a benchmark. The numbers
below say whether the classifier behaves sensibly on realistic phrasing;
they do not generalise to production traffic.

The LLM path calls the API directly rather than through `classify_intent`,
which catches every exception and falls back to keywords — that would hide
an outage behind a keyword-level score.

### LLM classifier (`gpt-4o-mini`)

- Overall accuracy: **36/36 (100.0%)**
- Errors (API call raised): **0**

| True label | Accuracy |
|---|---|
| greeting | 12/12 (100.0%) |
| product_query | 12/12 (100.0%) |
| high_intent | 12/12 (100.0%) |

Confusion (rows = true, columns = predicted):

| | greeting | product_query | high_intent |
|---|---|---|---|
| **greeting** | 12 | 0 | 0 |
| **product_query** | 0 | 12 | 0 |
| **high_intent** | 0 | 0 | 12 |

No misclassifications.

### Keyword fallback (`_keyword_fallback`)

- Overall accuracy: **25/36 (69.4%)**
- Errors (API call raised): **0**

| True label | Accuracy |
|---|---|
| greeting | 12/12 (100.0%) |
| product_query | 6/12 (50.0%) |
| high_intent | 7/12 (58.3%) |

Confusion (rows = true, columns = predicted):

| | greeting | product_query | high_intent |
|---|---|---|---|
| **greeting** | 12 | 0 | 0 |
| **product_query** | 3 | 6 | 3 |
| **high_intent** | 5 | 0 | 7 |

Misclassified:

| Message | True | Predicted |
|---|---|---|
| `how much is the pro plan?` | product_query | high_intent |
| `what's included in the basic plan?` | product_query | high_intent |
| `what resolution does the basic plan support?` | product_query | high_intent |
| `what's the difference between basic and pro?` | product_query | greeting |
| `sounds good, tell me more` | product_query | greeting |
| `is there a free trial?` | product_query | greeting |
| `sign me up for pro` | high_intent | greeting |
| `take my money` | high_intent | greeting |
| `let's do it` | high_intent | greeting |
| `where do I pay?` | high_intent | greeting |
| `yes, set me up` | high_intent | greeting |

