"""Score a keyword classifier on a labelled set. Needs no API key.

    python eval/run_eval.py                               # heldout, current rules
    python eval/run_eval.py --set dev --classifier v1
    python eval/run_eval.py --set heldout --summary-only  # hide per-message misses

Classifiers:
  current  utils.intent_classifier._keyword_fallback
  v1       eval/baseline_keyword_v1.py, the rules at commit 309d6c5
  v0       eval/baseline_keyword.py, the pre-rewrite rules

Each run writes eval/results/<timestamp>_<set>_<classifier>.json, including the
misclassified messages. --summary-only only limits what is printed, so held-out
misses need not be seen while the rules are still being changed.

eval/run_llm_eval.py scores the real gpt-4o-mini call and needs a key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

LABELS = ["greeting", "product_query", "high_intent"]
SETS = {"dev": HERE / "dev.jsonl", "heldout": HERE / "heldout.jsonl"}
RESULTS_DIR = HERE / "results"


def get_classifier(name: str):
    if name == "current":
        from utils.intent_classifier import _keyword_fallback

        return _keyword_fallback
    if name == "v1":
        from baseline_keyword_v1 import _keyword_fallback

        return _keyword_fallback
    if name == "v0":
        from baseline_keyword import keyword_fallback_before

        return keyword_fallback_before
    raise ValueError(f"unknown classifier: {name}")


def load_set(path: Path) -> list[dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for item in items:
        if set(item) != {"id", "message", "label"} or item["label"] not in LABELS:
            raise ValueError(f"bad row in {path}: {item}")
    return items


def wilson_interval(correct: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion, 95% by default."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def score(items: list[dict], classify) -> dict:
    rows = [{**item, "pred": classify(item["message"])} for item in items]
    correct = sum(row["pred"] == row["label"] for row in rows)
    confusion = {
        true: {pred: sum(r["label"] == true and r["pred"] == pred for r in rows) for pred in LABELS}
        for true in LABELS
    }
    per_class = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        predicted = sum(confusion[true][label] for true in LABELS)
        actual = sum(confusion[label].values())
        per_class[label] = {
            "precision": true_positive / predicted if predicted else None,
            "recall": true_positive / actual if actual else None,
            "support": actual,
        }
    low, high = wilson_interval(correct, len(rows))
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "wilson_95": [low, high],
        "per_class": per_class,
        "confusion": confusion,
        "predictions": {r["id"]: r["pred"] for r in rows},
        "misclassified": [r for r in rows if r["pred"] != r["label"]],
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def print_summary(result: dict, summary_only: bool) -> None:
    low, high = result["wilson_95"]
    print(
        f"{result['set']} / {result['classifier']}: {result['correct']}/{result['n']} "
        f"= {result['accuracy']:.1%} (Wilson 95% CI {low:.1%}-{high:.1%})"
    )
    if summary_only:
        return
    for label, stats in result["per_class"].items():
        print(
            f"  {label:14} precision {_ratio(stats['precision'])}  "
            f"recall {_ratio(stats['recall'])}  n={stats['support']}"
        )
    print("  confusion (rows = true, columns = predicted):")
    print("  " + " " * 14 + "".join(f"{label:>15}" for label in LABELS))
    for true in LABELS:
        cells = "".join(f"{result['confusion'][true][pred]:>15}" for pred in LABELS)
        print(f"  {true:14}{cells}")
    for row in result["misclassified"]:
        print(f"  MISS {row['id']}: {row['message']!r} true={row['label']} pred={row['pred']}")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=sorted(SETS), default="heldout")
    parser.add_argument("--classifier", choices=["current", "v1", "v0"], default="current")
    parser.add_argument("--summary-only", action="store_true", help="print accuracy and CI only")
    parser.add_argument("--no-write", action="store_true", help="do not write a results file")
    args = parser.parse_args(argv)

    path = SETS[args.set]
    now = datetime.now()
    result = {
        "set": args.set,
        "classifier": args.classifier,
        "timestamp": now.isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **score(load_set(path), get_classifier(args.classifier)),
    }
    print_summary(result, args.summary_only)

    if not args.no_write:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = now.strftime("%Y%m%dT%H%M%S")
        out = RESULTS_DIR / f"{stamp}_{args.set}_{args.classifier}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {out.relative_to(ROOT)}")
    return result


if __name__ == "__main__":
    main()
