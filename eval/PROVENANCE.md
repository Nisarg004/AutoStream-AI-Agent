# Is 46.7% → 86.7% a held-out result?

**Short answer: the numbers are real, but they should not be called held-out.**

`eval/intents_heldout.json` was committed before the rule rewrite, but the same
commit recorded every baseline miss on that set in `eval/RESULTS.md`.

| Order | Commit | What happened |
|---|---|---|
| 1 | `c5ff08d` (20:59:24) | Added the set and listed its misclassified messages in `RESULTS.md` |
| 2 | `212f7a0` (21:01:52) | Rewrote the rules 2.5 minutes later |

The rewrite then contains phrases from those listed misses, such as `count me in`,
`put me down`, `card details`, `checkout` and `money back`.

## Measurement

Reproduce with `python eval/audit_v1_leakage.py`.

- v0 scores 14/30 (46.7%) and v1 scores 26/30 (86.7%) on the old set. Both reproduce.
- 24 v1 patterns occur in the old held-out set but not in the tuning set (`eval/intents.json`).
- With those 24 removed, v1 scores **12/30 (40.0%)** on the old held-out set and
  still scores 36/36 on the tuning set. Every gain on the old held-out set depends
  on them.

## How to read this

The 12/30 is not v1's "true" score. Several of the 24 are generic product words
(`cost`, `video`, `cancel`, `4k`) that a careful author could have added without
seeing the set. The audit shows dependence on the set. It cannot show how many
patterns came from reading the misses.

The clean measure is on `eval/heldout.jsonl`, written after v1 was frozen and
before any further rule change. `eval/intents_heldout.json` and `eval/intents.json`
are kept unmodified for continuity, and neither should be cited as held-out evidence.
