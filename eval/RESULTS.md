# Intent classification results

- Run: 2026-09-19 21:03 local time
- Model: `gpt-4o-mini` (temperature 0, max_tokens 10)
- Prompt copy in sync with `utils/intent_classifier.py`: yes
- **before** = the rules frozen in `eval/baseline_keyword.py`; **after** = the rules live in `utils/intent_classifier.py`

| Dataset | LLM | Keyword BEFORE | Keyword AFTER | API errors |
|---|---|---|---|---|
| Original set | 36/36 (100.0%) | 25/36 (69.4%) | 36/36 (100.0%) | 0 |
| Held-out set | 30/30 (100.0%) | 14/30 (46.7%) | 26/30 (86.7%) | 0 |

**The original set informed the rewrite, so its 'after' number is not
evidence of anything — the misses there were read and fixed directly.
The held-out number is the fair one.** It was written and committed
before the rules were touched.

Both sets are hand-written and small, so these are sanity checks rather
than benchmarks, and both were written by the same author as the rules.

The LLM path calls the API directly rather than through `classify_intent`,
which catches every exception and falls back to keywords — that would hide
an outage behind a keyword-level score.

## Original set — `eval/intents.json` (36 messages, 12 greeting, 12 product_query, 12 high_intent)

36 hand-written messages for intent classification, 12 per label. Labels are the ones a human would defend for a lead-capture assistant; ambiguous cases carry a note explaining the call. Written to be realistic, not to favour either classifier.

#### LLM classifier (`gpt-4o-mini`)

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

#### Keyword fallback — BEFORE (`eval/baseline_keyword.py`)

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

#### Keyword fallback — AFTER (`utils/intent_classifier.py`)

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

## Held-out set — `eval/intents_heldout.json` (30 messages, 10 greeting, 10 product_query, 10 high_intent)

30 held-out messages, 10 per label, written before any change to the keyword fallback and kept separate from eval/intents.json. No message here is a reword of one there; phrasings deliberately avoid the vocabulary the original set leans on (plan names, the words 'refund', 'price', 'sign up'), so a rule tuned to the original set does not automatically transfer.

#### LLM classifier (`gpt-4o-mini`)

- Overall accuracy: **30/30 (100.0%)**
- Errors (API call raised): **0**

| True label | Accuracy |
|---|---|
| greeting | 10/10 (100.0%) |
| product_query | 10/10 (100.0%) |
| high_intent | 10/10 (100.0%) |

Confusion (rows = true, columns = predicted):

| | greeting | product_query | high_intent |
|---|---|---|---|
| **greeting** | 10 | 0 | 0 |
| **product_query** | 0 | 10 | 0 |
| **high_intent** | 0 | 0 | 10 |

No misclassifications.

#### Keyword fallback — BEFORE (`eval/baseline_keyword.py`)

- Overall accuracy: **14/30 (46.7%)**
- Errors (API call raised): **0**

| True label | Accuracy |
|---|---|
| greeting | 10/10 (100.0%) |
| product_query | 3/10 (30.0%) |
| high_intent | 1/10 (10.0%) |

Confusion (rows = true, columns = predicted):

| | greeting | product_query | high_intent |
|---|---|---|---|
| **greeting** | 10 | 0 | 0 |
| **product_query** | 7 | 3 | 0 |
| **high_intent** | 8 | 1 | 1 |

Misclassified:

| Message | True | Predicted |
|---|---|---|
| `does the cheaper tier do 1080p?` | product_query | greeting |
| `what happens if I cancel halfway through the month?` | product_query | greeting |
| `do I get priority help if something breaks?` | product_query | greeting |
| `can I upgrade later without losing my work?` | product_query | greeting |
| `which one has unlimited exports?` | product_query | greeting |
| `how long do I have to ask for my money back?` | product_query | greeting |
| `do you work with instagram reels?` | product_query | greeting |
| `ok I'm in` | high_intent | greeting |
| `count me in` | high_intent | greeting |
| `send me the checkout link` | high_intent | greeting |
| `I'd like to go ahead with the better plan` | high_intent | product_query |
| `put me down for the unlimited one` | high_intent | greeting |
| `happy to go ahead` | high_intent | greeting |
| `how do I create an account?` | high_intent | greeting |
| `I'd like to upgrade to the 4K one` | high_intent | greeting |
| `take my card details` | high_intent | greeting |

#### Keyword fallback — AFTER (`utils/intent_classifier.py`)

- Overall accuracy: **26/30 (86.7%)**
- Errors (API call raised): **0**

| True label | Accuracy |
|---|---|
| greeting | 7/10 (70.0%) |
| product_query | 10/10 (100.0%) |
| high_intent | 9/10 (90.0%) |

Confusion (rows = true, columns = predicted):

| | greeting | product_query | high_intent |
|---|---|---|---|
| **greeting** | 7 | 3 | 0 |
| **product_query** | 0 | 10 | 0 |
| **high_intent** | 0 | 1 | 9 |

Misclassified:

| Message | True | Predicted |
|---|---|---|
| `are you a real person?` | greeting | product_query |
| `just looking around for now` | greeting | product_query |
| `what can you help me with?` | greeting | product_query |
| `can we get this started today?` | high_intent | product_query |

