"""Print before and after accuracy from results already written by run_eval.py.

    python eval/compare.py                     # held-out, v1 -> current
    python eval/compare.py --before v0
    python eval/compare.py --set dev

Uses the newest results file for each classifier, and refuses to compare runs
made on different versions of the dataset.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def latest(set_name: str, classifier: str) -> dict:
    # Timestamps sort lexically.
    paths = sorted(RESULTS_DIR.glob(f"*_{set_name}_{classifier}.json"))
    if not paths:
        raise SystemExit(
            f"No results for {set_name}/{classifier}. Run: "
            f"python eval/run_eval.py --set {set_name} --classifier {classifier}"
        )
    return json.loads(paths[-1].read_text())


def exact_mcnemar(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    total = fixed + broken
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(min(fixed, broken) + 1)) / 2**total
    return min(1.0, 2 * tail)


def compare(before: dict, after: dict) -> dict:
    if before["dataset_sha256"] != after["dataset_sha256"]:
        raise SystemExit("These results come from different versions of the dataset.")
    wrong_before = {row["id"]: row for row in before["misclassified"]}
    wrong_after = {row["id"]: row for row in after["misclassified"]}
    fixed = [wrong_before[i] for i in sorted(wrong_before.keys() - wrong_after.keys())]
    broken = [wrong_after[i] for i in sorted(wrong_after.keys() - wrong_before.keys())]
    return {
        "fixed": fixed,
        "broken": broken,
        "still_wrong": [wrong_after[i] for i in sorted(wrong_after.keys() & wrong_before.keys())],
        "mcnemar_p": exact_mcnemar(len(fixed), len(broken)),
    }


def _line(result: dict) -> str:
    low, high = result["wilson_95"]
    return (
        f"{result['classifier']:8} {result['correct']:>3}/{result['n']:<3} "
        f"{result['accuracy']:6.1%}   95% CI {low:.1%}-{high:.1%}   "
        f"commit {result['git_commit']}{' (dirty)' if result['git_dirty'] else ''}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", default="heldout", choices=["dev", "heldout"])
    parser.add_argument("--before", default="v1", choices=["v0", "v1"])
    parser.add_argument("--after", default="current", choices=["v0", "v1", "current"])
    args = parser.parse_args()

    before, after = latest(args.set, args.before), latest(args.set, args.after)
    diff = compare(before, after)
    print(f"Set: {args.set}")
    print("BEFORE " + _line(before))
    print("AFTER  " + _line(after))
    print(f"Change: {(after['accuracy'] - before['accuracy']) * 100:+.1f} points")
    print(
        f"Fixed {len(diff['fixed'])}, broken {len(diff['broken'])}; "
        f"exact McNemar p = {diff['mcnemar_p']:.3f}"
    )
    for title, rows in (
        ("Fixed", diff["fixed"]),
        ("Broken", diff["broken"]),
        ("Still wrong", diff["still_wrong"]),
    ):
        for row in rows:
            print(f"  {title}: {row['id']} {row['message']!r} true={row['label']} pred={row['pred']}")


if __name__ == "__main__":
    main()
