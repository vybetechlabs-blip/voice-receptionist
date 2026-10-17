"""Rerun every scenario against a real LLM, N times each, and write a scored report.

    uv run python -m eval.rerun --llm ollama --model gpt-oss:20b --repeat 3 --label structural
    uv run python -m eval.rerun --llm openai --model gpt-6-luna --reasoning-effort low --repeat 3 --label luna
    uv run python -m eval.rerun --llm openai --model gpt-4.1-mini --repeat 3 --label gpt41mini

A real model varies run to run, so each scenario is played ``--repeat`` times. Each run is scored
on both tiers (eval/harness.py):

- outcome-correct (the gate): records written, the required spoken lines, and the call ending
  properly;
- clean-attempt (reported): no tool call was rejected.

Every rejected call is classified as ``bad_data`` (it would have written wrong data: invented caller
details, or an unwarranted transfer) or harmless but sloppy (an early ``end_call``, an unnecessary
call that would have failed anyway, malformed arguments).

The hosted provider reads its key from OPENAI_API_KEY in the environment and never logs it.
Writes eval/results/<date>-<model>[-<label>].md and .json; an existing report is never overwritten.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from loguru import logger

from agent.config import load_business
from agent.providers import ProviderUnavailable, openai_llm_choice, resolve_llm
from eval.harness import check_portable, load_scenarios, play, rejected_calls, rejection_breakdown

RESULTS = Path(__file__).parent / "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.rerun", description=__doc__.splitlines()[0])
    parser.add_argument("--llm", default="ollama", choices=("ollama", "scripted", "openai"))
    parser.add_argument("--model", help="model id for the chosen provider (default: config)")
    parser.add_argument("--ollama-model", help="same as --model for --llm ollama (kept for old commands)")
    parser.add_argument("--reasoning-effort", help="e.g. low (default: providers.openai.llm_reasoning_effort for the configured model)")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--only", action="append", help="scenario id (repeatable)")
    parser.add_argument("--label", default="", help="suffix for the report name, e.g. 'luna'")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    biz = load_business()
    model_arg = args.model or args.ollama_model
    provider = resolve_llm(biz, args.llm, model_arg if args.llm == "ollama" else None)
    effort = args.reasoning_effort
    reported_effort = effort  # what the request carried
    if provider == "ollama":
        model = model_arg or biz.providers.ollama.model
    elif provider == "openai":
        chosen, reported_effort, _ = openai_llm_choice(biz, model_arg, effort)
        model = chosen or "pipecat-default"
    else:
        model = provider
    scenarios = [s for s in load_scenarios() if not args.only or s.id in args.only]

    rows, runs_out = [], []
    for sc in scenarios:
        outcomes, cleans, failures, kinds = 0, 0, [], Counter()
        for i in range(args.repeat):
            t0 = time.monotonic()
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    run = play(sc, biz, provider, Path(tmp), model=model_arg, reasoning_effort=effort)
                except ProviderUnavailable as missing:  # e.g. OPENAI_API_KEY not set in this shell
                    print(f"ABORTED: {missing}", file=sys.stderr)
                    return 2
            if run.result.ended_by == "error":
                # Fail fast: a model/key/parameter problem fails every call the same way.
                print(f"ABORTED: the LLM returned an error on {sc.id} run {i + 1}:", file=sys.stderr)
                print(f"  {run.result.error}", file=sys.stderr)
                print("  Nothing was written to eval/results. Check the model id, key, and flags.", file=sys.stderr)
                return 2
            problems = check_portable(run, biz)  # outcome-correct: the gate
            rejected = rejected_calls(run)  # clean-attempt: reported, not gated
            breakdown = rejection_breakdown(run)
            outcomes += not problems
            cleans += not rejected
            kinds.update(breakdown)
            if problems:
                failures.append(problems)
            agent_turns = [u for u in run.result.transcript if u.speaker == "agent" and u.turn > 0]
            runs_out.append(
                {
                    "scenario": sc.id,
                    "run": i + 1,
                    "passed": not problems,
                    "failures": problems,
                    "clean": not rejected,
                    "rejected_calls": [(c.turn, c.name, c.arguments, c.result, c.note) for c in rejected],
                    "breakdown": dict(breakdown),
                    "ended_by": run.result.ended_by,
                    "crm": dataclasses.asdict(run.crm),
                    "seconds": round(time.monotonic() - t0, 1),
                    "transcript": [(u.turn, u.speaker, u.text) for u in run.result.transcript],
                    "tool_calls": [(c.turn, c.name, c.arguments, c.result) for c in run.result.tool_calls],
                    "usage": run.result.usage,
                    "caller_turns": sum(1 for u in run.result.transcript if u.speaker == "caller"),
                    "agent_chars": sum(len(u.text) for u in agent_turns),
                }
            )
            print(
                f"{sc.id:34s} run {i + 1}: outcome {'PASS' if not problems else 'FAIL'}, "
                f"clean {'yes' if not rejected else 'no'} {dict(breakdown) or ''}",
                flush=True,
            )
        rows.append((sc, outcomes, cleans, failures, kinds))

    usage = [u for r in runs_out for u in r["usage"]]
    turns = sum(r["caller_turns"] for r in runs_out) or 1
    totals = Counter()
    for *_, kinds in rows:
        totals.update(kinds)
    bad = totals.get("bad_data", 0)
    sloppy = sum(v for k, v in totals.items() if k.startswith("sloppy"))
    def mean(key):
        return round(statistics.mean(u.get(key) or 0 for u in usage), 1) if usage else None

    summary = {
        "provider": provider,
        "model": model,
        "reasoning_effort": reported_effort,
        "date": datetime.now().isoformat(timespec="minutes"),
        "repeat": args.repeat,
        "runs_outcome_correct": sum(r["passed"] for r in runs_out),
        "runs_clean": sum(r["clean"] for r in runs_out),
        "runs_total": len(runs_out),
        "rejections": {"bad_data": bad, "sloppy": sloppy, "by_kind": dict(totals)},
        "llm_requests": len(usage),
        "llm_requests_per_caller_turn": round(len(usage) / turns, 2),
        "prompt_tokens_total": sum(u.get("prompt_tokens") or 0 for u in usage),
        "completion_tokens_total": sum(u.get("completion_tokens") or 0 for u in usage),
        "reasoning_tokens_total": sum(u.get("reasoning_tokens") or 0 for u in usage),
        "prompt_tokens_per_request_mean": mean("prompt_tokens"),
        "completion_tokens_per_request_mean": mean("completion_tokens"),
        "agent_chars_per_caller_turn": round(sum(r["agent_chars"] for r in runs_out) / turns, 1),
    }

    RESULTS.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")
    stem = f"{datetime.now():%Y-%m-%d}-{slug}" + (f"-{args.label}" if args.label else "")
    if (RESULTS / f"{stem}.md").exists():
        # Never overwrite an earlier report (a same-day rerun once clobbered the baseline).
        stem += f"-{datetime.now():%H%M%S}"
    (RESULTS / f"{stem}.json").write_text(
        json.dumps({"summary": summary, "runs": runs_out}, indent=2, default=str), encoding="utf-8"
    )
    command = " ".join(["uv run python -m eval.rerun", *(argv if argv is not None else sys.argv[1:])])
    lines = [
        f"# Eval rerun: {model}{f' (reasoning effort {reported_effort})' if reported_effort else ''}, {summary['date']}",
        "",
        f"`{command}`",
        "",
        f"**Outcome-correct (gate): {summary['runs_outcome_correct']}/{summary['runs_total']}. "
        f"Clean-attempt (reported): {summary['runs_clean']}/{summary['runs_total']}.**",
        "",
        f"Rejected tool calls: **{bad} would have written bad data**, {sloppy} harmless but sloppy "
        f"({', '.join(f'{k} {v}' for k, v in sorted(totals.items())) or 'none'}).",
        "",
        "| Scenario | Outcome-correct | Clean-attempt | Bad-data blocks | Sloppy | First outcome failure |",
        "|---|---|---|---|---|---|",
    ]
    for sc, ok, clean, fails, kinds in rows:
        first = fails[0][0].replace("|", "/").replace(chr(10), " ")[:150] if fails else ""
        sl = sum(v for k, v in kinds.items() if k.startswith("sloppy"))
        lines.append(
            f"| {sc.id} | {ok}/{args.repeat} | {clean}/{args.repeat} | {kinds.get('bad_data', 0)} | {sl} | {first} |"
        )
    lines += [
        "",
        "Measured usage (tokens as the provider reported them; cost comes from the provider's usage page):",
        "",
        f"- LLM requests: {summary['llm_requests']} ({summary['llm_requests_per_caller_turn']} per caller turn)",
        f"- Tokens: {summary['prompt_tokens_total']:,} input, {summary['completion_tokens_total']:,} output"
        f" ({summary['reasoning_tokens_total']:,} of it reasoning)",
        f"- Per request (mean): {summary['prompt_tokens_per_request_mean']} input, "
        f"{summary['completion_tokens_per_request_mean']} output",
        f"- Agent characters spoken per caller turn: {summary['agent_chars_per_caller_turn']}",
        "",
        f"Full transcripts, tool calls, rejections, and usage: `{stem}.json`.",
    ]
    (RESULTS / f"{stem}.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print(chr(10).join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
