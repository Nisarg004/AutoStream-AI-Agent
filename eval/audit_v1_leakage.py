"""Measure how much the v1 rule rewrite depended on the old held-out set.

Reproduces the evidence in eval/PROVENANCE.md. Needs no API key.
Run from the project root:  python eval/audit_v1_leakage.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baseline_keyword_v1 as v1
from baseline_keyword import keyword_fallback_before as v0


def _load(name: str) -> list[dict]:
    return json.loads((HERE / name).read_text())["messages"]


def _hits(pattern: str, items: list[dict]) -> bool:
    regex = re.compile(r"\b" + re.escape(pattern) + r"\b")
    return any(regex.search(item["message"].lower()) for item in items)


def _without(removed: set[str]):
    """The v1 algorithm with the given patterns deleted from its lists."""

    def compile_(patterns):
        kept = [p for p in patterns if p not in removed]
        return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in kept) + r")\b")

    buying = compile_(v1._SIGNUP + v1._PURCHASE + v1._PAYMENT + v1._COMMITMENT)
    product = compile_(v1._PRODUCT)
    greeting = compile_(v1._GREETING)

    def classify(message: str) -> str:
        message = message.lower()
        if buying.search(message):
            return "high_intent"
        if product.search(message):
            return "product_query"
        if greeting.search(message):
            return "greeting"
        return "product_query"

    return classify


def _score(classify, items: list[dict]) -> str:
    correct = sum(classify(item["message"]) == item["label"] for item in items)
    return f"{correct}/{len(items)} ({correct / len(items):.1%})"


def main() -> None:
    heldout, original = _load("intents_heldout.json"), _load("intents.json")
    patterns = v1._SIGNUP + v1._PURCHASE + v1._PAYMENT + v1._COMMITMENT + v1._PRODUCT + v1._GREETING
    leaked = sorted(p for p in set(patterns) if _hits(p, heldout) and not _hits(p, original))

    print(f"v0 on intents_heldout.json: {_score(v0, heldout)}")
    print(f"v1 on intents_heldout.json: {_score(v1._keyword_fallback, heldout)}")
    print(f"v1 patterns found in the held-out set but not the tuning set: {len(leaked)}")
    print("  " + ", ".join(leaked))
    stripped = _without(set(leaked))
    print(f"v1 without those patterns, held-out: {_score(stripped, heldout)}")
    print(f"v1 without those patterns, tuning set: {_score(stripped, original)}")


if __name__ == "__main__":
    main()
