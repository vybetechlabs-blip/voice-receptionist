"""Re-score saved eval.rerun reports under both tiers with today's checker, without calling any model.

    uv run python -m eval.rescore eval/results/2026-09-25-gpt-oss-20b-enforced.json
    uv run python -m eval.rescore a.json b.json c.json --out eval/results/<date>-comparison.md

outcome-correct (the gate) and clean-attempt (reported) are defined in eval/harness.py. Every
report is scored against the scenario files and harness on disk now, so reports written under
older checks are compared on equal terms; the printed checker fingerprint names that version.

Reports written before the second eval revision of 2026-09-25 lack the CRM snapshot and ended_by; for those
they are rebuilt from the tool results. The tools write only when they report booked, saved, or
transferring, so the rebuild is exact. The bad-data/sloppy breakdown is read from the report (the
tool bridge classified each rejection while the call ran); older reports show "n/a".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from agent.config import load_business
from agent.core import CallResult, Utterance
from agent.prompt import render_system_prompt
from agent.tool_bridge import ToolCall
from eval.harness import FROZEN_NOW, CrmChanges, Run, check_portable, load_scenarios, rejected_calls

EVAL = Path(__file__).parent


def checker_fingerprint() -> str:
    """Short hash of the harness and every scenario file: equal fingerprints, equal checker."""
    digest = hashlib.sha256()
    for f in [EVAL / "harness.py", *sorted((EVAL / "scenarios").glob("*.yaml"))]:
        digest.update(f.name.encode())
        digest.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:12]


def rebuild(saved: dict, scenario, system_prompt: str) -> Run:
    calls = [ToolCall(name, args, res, turn) for (turn, name, args, res) in saved["tool_calls"]]
    transcript = [Utterance(speaker, text, turn) for (turn, speaker, text) in saved["transcript"]]
    def went(status):
        return [c for c in calls if isinstance(c.result, dict) and c.result.get("status") == status]

    transfers = went("transferring")
    ended_by = saved.get("ended_by") or ("transfer" if transfers else "caller")
    crm = saved.get("crm")
    if crm:
        changes = CrmChanges(**crm)
    else:
        changes = CrmChanges(
            bookings_added=[c.result["booking"] for c in went("booked")],
            seed_bookings_intact=True,  # no tool can change or delete a booking
            messages_added=[c.result["message"] for c in went("saved")],
            transfers_logged=len(transfers),
            transfer_reasons=[str(c.arguments.get("reason", "")) for c in transfers],
        )
    return Run(scenario, CallResult("rescored", transcript, calls, ended_by, system_prompt), changes)


def score(path: Path) -> dict:
    """Score one report. Returns totals, per-scenario counts, the breakdown, and failing runs."""
    biz = load_business()
    scenarios = {s.id: s for s in load_scenarios()}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    prompt = render_system_prompt(biz, FROZEN_NOW)
    table: dict[str, list[tuple[bool, bool]]] = {}
    failing, kinds, known = [], Counter(), True
    for saved in data["runs"]:
        run = rebuild(saved, scenarios[saved["scenario"]], prompt)
        problems = check_portable(run, biz)
        table.setdefault(saved["scenario"], []).append((not problems, not rejected_calls(run)))
        if problems:
            first = problems[0].replace(chr(10), " ")[:200]
            failing.append(f"{saved['scenario']} run {saved.get('run', '?')}: {first}")
        if "breakdown" in saved:
            kinds.update(saved["breakdown"])
        else:
            known = False
    model = summary.get("model") or path.stem
    effort = summary.get("reasoning_effort")
    return {
        "report": path.name,
        "label": f"{model} (effort {effort})" if effort else model,
        "outcome": sum(r for rows in table.values() for r, _ in rows),
        "clean": sum(c for rows in table.values() for _, c in rows),
        "total": sum(len(rows) for rows in table.values()),
        "scenarios": {sid: (sum(r for r, _ in rows), sum(c for _, c in rows), len(rows)) for sid, rows in table.items()},
        "bad_data": kinds.get("bad_data", 0) if known else None,
        "sloppy": sum(v for k, v in kinds.items() if k.startswith("sloppy")) if known else None,
        "by_kind": dict(sorted(kinds.items())) if known else None,
        "failing": failing,
        "usage": {k: summary.get(k) for k in ("llm_requests", "prompt_tokens_total", "completion_tokens_total", "reasoning_tokens_total")},
    }


def _na(value) -> str:
    return "n/a" if value is None else f"{value:,}" if isinstance(value, int) else str(value)


def report_lines(r: dict) -> list[str]:
    kinds = ", ".join(f"{k} {v}" for k, v in r["by_kind"].items()) if r["by_kind"] else ("none" if r["by_kind"] == {} else "n/a")
    return [
        f"### {r['label']} (`{r['report']}`)",
        "",
        f"**Outcome-correct (gate): {r['outcome']}/{r['total']}. Clean-attempt (reported): {r['clean']}/{r['total']}.**",
        f"Blocked calls: {_na(r['bad_data'])} would have written bad data, {_na(r['sloppy'])} harmless but sloppy ({kinds}).",
        "",
        "| Scenario | Outcome-correct | Clean-attempt |",
        "|---|---|---|",
        *[f"| {sid} | {o}/{n} | {c}/{n} |" for sid, (o, c, n) in r["scenarios"].items()],
        "",
        "Outcome failures:" if r["failing"] else "Outcome failures: none.",
        *[f"- {line}" for line in r["failing"]],
        "",
    ]


def comparison_lines(results: list[dict]) -> list[str]:
    rows = [
        ("Outcome-correct (gate)", lambda r: f"**{r['outcome']}/{r['total']}**"),
        ("Clean-attempt (reported)", lambda r: f"{r['clean']}/{r['total']}"),
        ("Blocked: would have written bad data", lambda r: _na(r["bad_data"])),
        ("Blocked: harmless but sloppy", lambda r: _na(r["sloppy"])),
        ("LLM requests", lambda r: _na(r["usage"]["llm_requests"])),
        ("Tokens in", lambda r: _na(r["usage"]["prompt_tokens_total"])),
        ("Tokens out (of which reasoning)", lambda r: f"{_na(r['usage']['completion_tokens_total'])} ({_na(r['usage']['reasoning_tokens_total'])})"),
    ]
    order = [s.id for s in load_scenarios()]
    labels = [r["label"] for r in results]
    heads = [f"{r['label']} `{Path(r['report']).stem}`" if labels.count(r["label"]) > 1 else r["label"] for r in results]
    lines = [
        "| | " + " | ".join(heads) + " |",
        "|---|" + "---|" * len(results),
        *[f"| {name} | " + " | ".join(cell(r) for r in results) + " |" for name, cell in rows],
    ]
    for sid in order:
        cells = []
        for r in results:
            o, c, n = r["scenarios"].get(sid, (0, 0, 0))
            cells.append(f"{o}/{n} (clean {c}/{n})" if n else "not run")
        lines.append(f"| {sid} | " + " | ".join(cells) + " |")
    return lines


def rescore(path: Path) -> tuple[dict, list[str]]:
    """One report: (totals, markdown lines). Kept for callers of the single-report form."""
    r = score(path)
    return {"outcome": r["outcome"], "clean": r["clean"], "total": r["total"]}, report_lines(r)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.rescore", description=__doc__.splitlines()[0])
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, help="also write the markdown here (never overwrites)")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    results = [score(p) for p in args.reports]
    lines = [f"Checker fingerprint `{checker_fingerprint()}` (eval/harness.py + eval/scenarios/*.yaml on disk now).", ""]
    if len(results) > 1:
        lines += ["## Side by side, same checker", "", *comparison_lines(results), ""]
    for r in results:
        lines += report_lines(r)
    text = chr(10).join(lines)
    print(text)
    if args.out:
        if args.out.exists():
            print(f"NOT WRITTEN: {args.out} exists (reports are never overwritten).", file=sys.stderr)
            return 1
        args.out.write_text(text + chr(10), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
