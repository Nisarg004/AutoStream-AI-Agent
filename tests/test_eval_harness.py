"""The eval numbers are only as good as the code and data behind them."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import compare
import run_eval


def _row(identifier: str, label: str, pred: str) -> dict:
    return {"id": identifier, "message": identifier, "label": label, "pred": pred}


def test_wilson_interval_matches_a_published_value():
    # Newcombe (1998), table I: 81/263 gives 0.2553-0.3662.
    low, high = run_eval.wilson_interval(81, 263)

    assert low == pytest.approx(0.2553, abs=1e-4)
    assert high == pytest.approx(0.3662, abs=1e-4)


def test_wilson_interval_stays_informative_at_the_extremes():
    assert run_eval.wilson_interval(30, 30)[1] == pytest.approx(1.0)
    assert run_eval.wilson_interval(30, 30)[0] < 1.0
    assert run_eval.wilson_interval(0, 30)[0] == pytest.approx(0.0, abs=1e-12)
    assert run_eval.wilson_interval(0, 0) == (0.0, 0.0)


def test_score_computes_precision_recall_and_confusion():
    items = [
        {"id": "a", "message": "g1", "label": "greeting"},
        {"id": "b", "message": "g2", "label": "greeting"},
        {"id": "c", "message": "p1", "label": "product_query"},
    ]
    predictions = {"g1": "greeting", "g2": "product_query", "p1": "product_query"}

    result = run_eval.score(items, predictions.get)

    assert result["correct"] == 2
    assert result["per_class"]["greeting"]["recall"] == 0.5
    assert result["per_class"]["greeting"]["precision"] == 1.0
    assert result["per_class"]["product_query"]["precision"] == 0.5
    assert result["per_class"]["high_intent"]["precision"] is None
    assert result["confusion"]["greeting"]["product_query"] == 1
    assert [row["id"] for row in result["misclassified"]] == ["b"]


def test_load_set_rejects_an_unknown_label(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "message": "m", "label": "sales"}) + "\n")

    with pytest.raises(ValueError):
        run_eval.load_set(bad)


def test_heldout_is_exactly_thirty_and_balanced():
    items = run_eval.load_set(run_eval.SETS["heldout"])

    assert len(items) == 30
    for label in run_eval.LABELS:
        assert sum(item["label"] == label for item in items) == 10


def test_dev_and_heldout_share_no_message():
    dev, heldout = (
        {item["message"].lower().strip() for item in run_eval.load_set(run_eval.SETS[name])}
        for name in ("dev", "heldout")
    )

    assert not dev & heldout


def test_heldout_matches_the_hash_in_every_recorded_result():
    """Changing the set after seeing results would break this."""
    actual = run_eval.hashlib.sha256(run_eval.SETS["heldout"].read_bytes()).hexdigest()

    recorded = {
        json.loads(path.read_text())["dataset_sha256"]
        for path in run_eval.RESULTS_DIR.glob("*_heldout_*.json")
    }
    assert recorded == {actual}


@pytest.mark.parametrize("classifier", ["v0", "v1", "current"])
def test_recorded_heldout_results_still_reproduce(classifier):
    """The README's numbers come from these files, so the code must still give them."""
    recorded = compare.latest("heldout", classifier)

    fresh = run_eval.score(
        run_eval.load_set(run_eval.SETS["heldout"]), run_eval.get_classifier(classifier)
    )

    assert fresh["predictions"] == recorded["predictions"]


def test_compare_separates_fixed_broken_and_still_wrong():
    before = {"dataset_sha256": "x", "misclassified": [_row("a", "greeting", "high_intent"), _row("b", "greeting", "high_intent")]}
    after = {"dataset_sha256": "x", "misclassified": [_row("b", "greeting", "high_intent"), _row("c", "greeting", "high_intent")]}

    diff = compare.compare(before, after)

    assert [row["id"] for row in diff["fixed"]] == ["a"]
    assert [row["id"] for row in diff["broken"]] == ["c"]
    assert [row["id"] for row in diff["still_wrong"]] == ["b"]


def test_compare_refuses_results_from_different_datasets():
    with pytest.raises(SystemExit):
        compare.compare(
            {"dataset_sha256": "x", "misclassified": []},
            {"dataset_sha256": "y", "misclassified": []},
        )


@pytest.mark.parametrize(
    "fixed,broken,expected",
    [(3, 0, 0.25), (0, 0, 1.0), (5, 5, 1.0), (11, 2, 0.0225)],
)
def test_exact_mcnemar(fixed, broken, expected):
    assert compare.exact_mcnemar(fixed, broken) == pytest.approx(expected, abs=1e-3)


def test_run_eval_runs_as_a_script_without_an_api_key():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env["OPENAI_API_KEY"] = ""

    done = subprocess.run(
        [sys.executable, "eval/run_eval.py", "--set", "dev", "--no-write"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )

    assert done.returncode == 0, done.stderr
    assert "Wilson 95% CI" in done.stdout
