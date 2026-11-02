# Prediction

Written before `eval/dev.jsonl` or `eval/heldout.jsonl` were scored by any classifier.

## Baseline failure modes

**v0 (`baseline_keyword.py`, pre-rewrite rules)**

- Plan-named questions go to `high_intent`, because high-intent keywords run first.
- Substrings misfire: `try` in `country`, `start` in `restart`, `want to` in `I want to know`.
- Unmatched messages become `greeting`, so idiomatic buying language is lost.
- Expected held-out accuracy: **40-60%**.

**v1 (`baseline_keyword_v1.py`, the rules at `309d6c5`)**

- `buy` beats question form, so a message like "Before I buy anything, ...?" goes to `high_intent`.
- Plain statements such as "I want the Pro plan" hit product vocabulary and become `product_query`.
- The `product_query` default swallows typo'd greetings and typo'd buying language.
- Expected held-out accuracy: **55-75%**.

## After the dev-tuned fix

- Expected held-out accuracy: **75-90%**. I expect a real gain over v1, with the
  interval overlapping v1's at n=30.
- Remaining misses are likely to be typos and unseen idioms, because the classifier
  is still rule-based.

## Limits of this prediction

I read v1's rules before writing the datasets and deliberately included cases that
attack them. The sets therefore lean harder than a random sample of traffic, and
their scores are not comparable with the old 46.7% and 86.7%.

## What actually happened

Single held-out run from clean commit `5c3a2b9`, with the set unchanged since
`ee9efb0`.

| Rules | Predicted | Measured | 95% CI | Prediction |
|---|---|---|---|---|
| v0 | 40-60% | 17/30 (56.7%) | 39.2%-72.6% | Within range |
| v1 | 55-75% | 23/30 (76.7%) | 59.1%-88.2% | Slightly above range |
| After fix | 75-90% | 26/30 (86.7%) | 70.3%-94.7% | Within range |

- **v1 to after:** +10.0 points, with 3 fixed and 0 broken. Exact McNemar p = 0.250,
  so this improvement is **not statistically significant**. The intervals overlap, as predicted.
- **v0 to after:** +30.0 points, with 11 fixed and 2 broken. Exact McNemar p = 0.022.
- **Failure modes:** v1 lost `h20` to a mentioned buying word and `h27` to a plain
  statement, both as predicted.
- **Remaining misses:** `who am I talking to?`, `helo`, `I'll go with Pro` and
  `Where do I enter my card?`. These are the predicted unseen idioms and typos.
- **Known dev miss, left unfixed:** `d39` (`We are ready to purchase. Who do we talk to?`) scored correctly before the action-question
  pattern was anchored, then failed. I saw the drop to 39/40 only after that commit
  was final and the held-out run had happened, and fixing it now would change the
  rules after the held-out run.

## Coincidence with the old figure

86.7% equals the old 26/30 by coincidence: the set and the rules are different. The
rules were not adjusted after the held-out run. I wrote both the rules and this set,
and I knew the set's messages while tuning, so this is weaker evidence than a set
written by someone else.
