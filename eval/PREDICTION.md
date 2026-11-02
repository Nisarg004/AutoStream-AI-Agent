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

To be filled in after the single held-out run of the fixed classifier.
