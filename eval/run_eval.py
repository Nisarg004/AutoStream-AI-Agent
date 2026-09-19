"""Intent classification evaluation.

Scores two classifiers on eval/intents.json:

  llm      the real gpt-4o-mini call, made directly here
  keyword  utils.intent_classifier._keyword_fallback, alone

The LLM path deliberately does NOT call classify_intent. That function
catches every exception and falls back to keywords, so an API outage, an
expired key or a quota block would silently score as keyword accuracy and
look like a healthy LLM result. Calling the API directly means a failure is
counted as an error and reported separately.

Requires OPENAI_API_KEY (unlike the test suite, which stubs every call).
Run from the project root:  python eval/run_eval.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from openai import OpenAI

from utils.intent_classifier import _keyword_fallback

load_dotenv()

HERE = Path(__file__).resolve().parent
DATASETS = [
    ("Original set", HERE / "intents.json"),
    ("Held-out set", HERE / "intents_heldout.json"),
]
RESULTS = HERE / "RESULTS.md"
CLASSIFIER_SRC = ROOT / "utils" / "intent_classifier.py"

MODEL = "gpt-4o-mini"
LABELS = ["greeting", "product_query", "high_intent"]

# Copied verbatim from utils.intent_classifier.classify_intent. Copying means
# the two can drift apart, so _prompt_sync_status() re-reads the source and
# reports whether every line is still present there.
SYSTEM_PROMPT = (
    "Classify the user message into exactly one of: "
    "greeting, product_query, high_intent.\n"
    "- greeting: hellos, hi, hey, thanks, casual conversation\n"
    "- product_query: questions about pricing, features, plans, policies\n"
    "- high_intent: user wants to sign up, buy, try, start, or subscribe\n"
    "Return only the label, nothing else. No explanation."
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _prompt_sync_status() -> tuple[bool, list[str]]:
    """Check every line of the copied prompt still appears in the agent source.

    Adjacent string literals are joined first: the source writes the prompt as
    several literals on consecutive lines, so a prompt line can span two of
    them and would otherwise look missing.
    """
    source = re.sub(r'"\s*\n\s*"', "", CLASSIFIER_SRC.read_text())
    missing = [line for line in SYSTEM_PROMPT.split("\n") if line and line not in source]
    in_sync = not missing and f'"{MODEL}"' in source
    return in_sync, missing


def classify_llm(message: str) -> str:
    """One direct API call. Raises on failure; nothing is caught here."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        max_tokens=10,
        temperature=0,
    )
    return response.choices[0].message.content.strip().lower()


def evaluate(predict, items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        row = {"message": item["message"], "true": item["label"], "pred": None, "error": None}
        try:
            row["pred"] = predict(item["message"])
        except Exception as exc:  # counted, never silently absorbed
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["error"] is None]
    errors = [r for r in rows if r["error"] is not None]
    correct = [r for r in scored if r["pred"] == r["true"]]

    per_label = {}
    for label in LABELS:
        group = [r for r in scored if r["true"] == label]
        hits = [r for r in group if r["pred"] == label]
        per_label[label] = (len(hits), len(group))

    predicted_values = sorted({r["pred"] for r in scored})
    columns = LABELS + [p for p in predicted_values if p not in LABELS]
    confusion = {
        true: Counter(r["pred"] for r in scored if r["true"] == true) for true in LABELS
    }

    return {
        "rows": rows,
        "scored": len(scored),
        "errors": errors,
        "correct": len(correct),
        "accuracy": len(correct) / len(scored) if scored else 0.0,
        "per_label": per_label,
        "columns": columns,
        "confusion": confusion,
        "misclassified": [r for r in scored if r["pred"] != r["true"]],
    }


def _pct(hits: int, total: int) -> str:
    return f"{hits}/{total} ({hits / total:.1%})" if total else "0/0 (n/a)"


def render(name: str, stats: dict) -> str:
    out = [f"#### {name}", ""]
    out.append(f"- Overall accuracy: **{_pct(stats['correct'], stats['scored'])}**")
    out.append(f"- Errors (API call raised): **{len(stats['errors'])}**")
    out.append("")
    out.append("| True label | Accuracy |")
    out.append("|---|---|")
    for label, (hits, total) in stats["per_label"].items():
        out.append(f"| {label} | {_pct(hits, total)} |")
    out.append("")

    out.append("Confusion (rows = true, columns = predicted):")
    out.append("")
    out.append("| | " + " | ".join(stats["columns"]) + " |")
    out.append("|---" * (len(stats["columns"]) + 1) + "|")
    for true in LABELS:
        counts = stats["confusion"][true]
        cells = [str(counts.get(col, 0)) for col in stats["columns"]]
        out.append(f"| **{true}** | " + " | ".join(cells) + " |")
    out.append("")

    if stats["misclassified"]:
        out.append("Misclassified:")
        out.append("")
        out.append("| Message | True | Predicted |")
        out.append("|---|---|---|")
        for r in stats["misclassified"]:
            out.append(f"| `{r['message']}` | {r['true']} | {r['pred']} |")
    else:
        out.append("No misclassifications.")
    out.append("")

    if stats["errors"]:
        out.append("Errors:")
        out.append("")
        for r in stats["errors"]:
            out.append(f"- `{r['message']}` — {r['error']}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. eval/ needs a real key; the test suite does not.")

    in_sync, missing = _prompt_sync_status()
    print(f"Model: {MODEL}")
    if not in_sync:
        print(f"WARNING: the copied prompt has drifted from {CLASSIFIER_SRC.name}: {missing}")

    sections, headline = [], []
    for title, path in DATASETS:
        data = json.loads(path.read_text())
        items = data["messages"]
        spread = ", ".join(f"{n} {l}" for l, n in Counter(i["label"] for i in items).items())
        print(f"\n{title}: {len(items)} messages ({spread})")

        print("  LLM (one API call per message)...")
        llm = summarise(evaluate(classify_llm, items))
        print("  keyword fallback...")
        keyword = summarise(evaluate(_keyword_fallback, items))

        sections.append("\n".join([
            f"## {title} — `eval/{path.name}` ({len(items)} messages, {spread})",
            "",
            data["description"],
            "",
            render(f"LLM classifier (`{MODEL}`)", llm),
            render("Keyword fallback (`_keyword_fallback`)", keyword),
        ]))
        headline.append(
            f"| {title} | {_pct(llm['correct'], llm['scored'])} | "
            f"{_pct(keyword['correct'], keyword['scored'])} | {len(llm['errors'])} |"
        )

    body = "\n".join([
        "# Intent classification results",
        "",
        f"- Run: {datetime.now():%Y-%m-%d %H:%M} local time",
        f"- Model: `{MODEL}` (temperature 0, max_tokens 10)",
        f"- Keyword fallback: **before** — the original rules, not yet changed",
        f"- Prompt copy in sync with `utils/intent_classifier.py`: "
        f"{'yes' if in_sync else 'NO — ' + str(missing)}",
        "",
        "| Dataset | LLM | Keyword fallback (before) | API errors |",
        "|---|---|---|---|",
        *headline,
        "",
        "Both sets are hand-written and small, so these are sanity checks rather",
        "than benchmarks. The original set has already been read and reasoned",
        "about; the held-out set was written before any change to the fallback and",
        "is the fairer of the two.",
        "",
        "The LLM path calls the API directly rather than through `classify_intent`,",
        "which catches every exception and falls back to keywords — that would hide",
        "an outage behind a keyword-level score.",
        "",
        *sections,
    ])
    RESULTS.write_text(body + "\n")
    print(f"\nWritten to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
