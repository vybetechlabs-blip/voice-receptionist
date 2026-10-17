"""The eval numbers the README states, recomputed from the committed reports.

No model is called: eval.rescore re-scores the saved transcripts on the checker in this repo.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from eval.rescore import checker_fingerprint, score

pytestmark = pytest.mark.unit
RESULTS = Path(__file__).parent / "results"


def _row(report: str) -> tuple:
    r = score(RESULTS / report)
    return r["outcome"], r["clean"], r["total"], r["bad_data"], r["sloppy"]


def test_the_three_models_rescore_to_the_published_numbers():
    # (outcome-correct, clean-attempt, runs, blocked bad-data calls, blocked sloppy calls)
    assert _row("2026-09-25-gpt-6-luna-luna.json") == (45, 45, 45, 0, 0)
    assert _row("2026-09-25-gpt-4.1-mini-gpt41mini.json") == (41, 45, 45, 0, 0)
    assert _row("2026-09-25-gpt-oss-20b-enforced.json") == (43, 18, 45, 12, 34)


def test_the_historical_gpt_oss_report_keeps_its_original_headline():
    # Two checks were relaxed after this run; the report was left as written.
    headline = (RESULTS / "2026-09-25-gpt-oss-20b-enforced.md").read_text(encoding="utf-8")
    assert "Outcome-correct (gate): 39/45" in headline


def test_the_twelve_blocked_bad_data_calls_are_ten_invented_details_and_two_unwarranted_transfers():
    report = json.loads((RESULTS / "2026-09-25-gpt-oss-20b-enforced.json").read_text(encoding="utf-8"))
    counts: Counter = Counter()
    for run in report["runs"]:
        for _turn, tool, _args, result, kind in run["rejected_calls"]:
            if kind != "bad_data":
                continue
            invented = any("not something the caller said" in e for e in result.get("errors", []))
            counts[(tool, "invented caller detail" if invented else "unwarranted transfer")] += 1
    assert counts == {
        ("book_appointment", "invented caller detail"): 5,
        ("take_message", "invented caller detail"): 5,
        ("transfer_call", "unwarranted transfer"): 2,
    }


def test_the_comparison_file_was_scored_on_this_checker():
    comparison = (RESULTS / "2026-09-26-comparison.md").read_text(encoding="utf-8")
    assert f"`{checker_fingerprint()}`" in comparison
