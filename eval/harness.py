"""Scenario harness: play a scripted caller through the text-mode pipeline and check the outcome.

A scenario (eval/scenarios/*.yaml) lists caller lines and expectations. Two tiers of checks:

- ``expect`` is portable: behavior any competent LLM should show (which tools, with which key
  arguments, what reached the fake CRM, what the agent must or must not say). These run under
  ScriptedLLM by default and rerun against a real model with ``--llm ollama``.
- ``strict`` pins exact wording and counts. It only holds for the deterministic ScriptedLLM.

Values in ``args``, ``result``, and CRM entries match as subsets. A string starting with ``re:``
is a regex search; other strings compare case-insensitively.

Scoring has two tiers (owner, Checkpoint 2 stop reviews):

- **outcome-correct** (the gate): ``check_portable``. It covers the records written, the required
  spoken lines, and the call ending properly. Tool-call expectations and ``not_called`` look only
  at calls that went through; a call a guard or the tool itself rejected changed nothing.
- **clean-attempt** (reported, not gated): ``rejected_calls``. A run is clean when no tool call was
  rejected, whether by a guard or by the tool's own validation. ``rejection_breakdown`` splits the
  rejections into ``bad_data`` (the call would have written wrong data) and harmless but sloppy.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from agent.config import BusinessConfig
from agent.core import CallResult
from agent.session import run_call
from agent.text_io import ScriptedCaller
from tools.store import CALENDAR, MESSAGES, SEED_DIR, TRANSFERS, read_json

SCENARIO_DIR = Path(__file__).parent / "scenarios"
# All scenario dates are relative to this moment: Monday, October 5, 2026, 9:00 AM.
FROZEN_NOW = datetime(2026, 10, 5, 9, 0)
NEVER_CLAIMS_HUMAN = re.compile(
    r"\bI'?m (a )?(real )?(person|human)\b|\bI am (a )?(real )?(person|human)\b", re.I
)


@dataclass
class Scenario:
    id: str
    title: str
    covers: list[str]
    caller: list[str]
    expect: dict
    strict: dict = field(default_factory=dict)


@dataclass
class CrmChanges:
    bookings_added: list[dict]
    seed_bookings_intact: bool
    messages_added: list[dict]
    transfers_logged: int
    transfer_reasons: list[str] = field(default_factory=list)


@dataclass
class Run:
    scenario: Scenario
    result: CallResult
    crm: CrmChanges

    def transcript(self) -> str:
        lines = [f"  [{u.turn}] {u.speaker}: {u.text}" for u in self.result.transcript]
        lines += [f"  tool {c.name}({json.dumps(c.arguments)}) -> {json.dumps(c.result)}" for c in self.result.tool_calls]
        return "\n".join(lines)


def load_scenarios() -> list[Scenario]:
    scenarios = []
    for path in sorted(SCENARIO_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        scenarios.append(
            Scenario(
                id=raw["id"],
                title=raw["title"],
                covers=list(raw.get("covers", [])),
                caller=[str(line) for line in raw["caller"]],
                expect=raw.get("expect") or {},
                strict=raw.get("strict") or {},
            )
        )
    return scenarios


def play(
    scenario: Scenario,
    biz: BusinessConfig,
    llm_provider: str,
    data_dir: Path,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> Run:
    turn_timeout = 20.0 if llm_provider == "scripted" else 180.0
    result = asyncio.run(
        run_call(
            biz,
            caller=ScriptedCaller(scenario.caller),
            llm_provider=llm_provider,
            data_dir=data_dir,
            clock=lambda: FROZEN_NOW,
            turn_timeout_s=turn_timeout,
            model=model,
            reasoning_effort=reasoning_effort,
        )
    )
    return Run(scenario, result, _crm_changes(data_dir))


def _crm_changes(data_dir: Path) -> CrmChanges:
    seed = {b["id"]: b for b in read_json(SEED_DIR / CALENDAR)["bookings"]}
    now = read_json(data_dir / CALENDAR)["bookings"]
    by_id = {b["id"]: b for b in now}
    seed_messages = {m["id"] for m in read_json(SEED_DIR / MESSAGES)["messages"]}
    log = (data_dir / TRANSFERS).read_text(encoding="utf-8")
    return CrmChanges(
        bookings_added=[b for b in now if b["id"] not in seed],
        seed_bookings_intact=all(by_id.get(i) == b for i, b in seed.items()),
        messages_added=[m for m in read_json(data_dir / MESSAGES)["messages"] if m["id"] not in seed_messages],
        transfers_logged=sum(1 for line in log.splitlines() if line.strip()),
        transfer_reasons=[f[7:] for line in log.splitlines() for f in line.split(chr(9)) if f.startswith("reason=")],
    )


REJECTED = ("invalid", "refused")


def is_rejected(call) -> bool:
    return isinstance(call.result, dict) and call.result.get("status") in REJECTED


def rejected_calls(run: Run) -> list:
    """Clean-attempt tier: tool calls a guard or the tool itself turned away."""
    return [c for c in run.result.tool_calls if is_rejected(c)]


def rejection_breakdown(run: Run) -> dict[str, int]:
    """Count rejected calls by the tool bridge's classification (bad_data or a sloppy_* kind)."""
    counts: dict[str, int] = {}
    for c in rejected_calls(run):
        kind = getattr(c, "note", None) or "unclassified"
        counts[kind] = counts.get(kind, 0) + 1
    return counts


# ---- matching ---------------------------------------------------------------------------------


def matches(expected, actual) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and matches(v, actual[k]) for k, v in expected.items())
    if expected is None:
        return actual is None
    if isinstance(expected, str):
        if expected.startswith("re:"):
            return actual is not None and re.search(expected[3:], str(actual)) is not None
        return str(actual).strip().lower() == expected.strip().lower()
    return expected == actual


def _as_list(value) -> list[str]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _check_tool_calls(expected: list[dict], run: Run, failures: list[str]) -> None:
    calls = run.result.tool_calls
    position = 0
    for want in expected:
        for i in range(position, len(calls)):
            c = calls[i]
            if (
                c.name == want["name"]
                and matches(want.get("args", {}), c.arguments)
                and ("result" not in want or matches(want["result"], c.result))
                and ("turn" not in want or c.turn == want["turn"])
            ):
                position = i + 1
                break
        else:
            failures.append(f"expected tool call not found (in order): {want}")


# Real models emit typographic characters (gpt-oss: curly apostrophes, narrow no-break spaces,
# non-breaking hyphens). Reply checks compare plain text so "can’t" matches "can't".
_TYPOGRAPHY = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
                             "\u202f": " ", "\u00a0": " ", "\u2011": "-", "\u2010": "-"})


def plain(text: str) -> str:
    return text.translate(_TYPOGRAPHY)


def _check_replies(replies: dict, run: Run, failures: list[str]) -> None:
    for turn, rules in (replies or {}).items():
        said = plain(run.result.agent_text(int(turn)))
        for pattern in _as_list(rules.get("match")):
            if not re.search(pattern, said):
                failures.append(f"turn {turn}: reply should match /{pattern}/ but was: {said!r}")
        for pattern in _as_list(rules.get("not_match")):
            if re.search(pattern, said):
                failures.append(f"turn {turn}: reply must not match /{pattern}/ but was: {said!r}")
        if "equals" in rules and said != rules["equals"]:
            failures.append(f"turn {turn}: reply should equal {rules['equals']!r} but was: {said!r}")


def _check_entries(label: str, expected: list[dict] | None, actual: list[dict], failures: list[str]) -> None:
    if expected is None:
        return
    if not expected and actual:
        failures.append(f"{label}: expected no new entries, found {actual}")
    for want in expected:
        if not any(matches(want, got) for got in actual):
            failures.append(f"{label}: no entry matching {want}; entries: {actual}")


def _heard(value: str, caller_lines: list[str]) -> bool:
    """True when the caller already said ``value``: a phone number by its digits within one line,
    a name by all of its words."""
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 4:
        return any(digits in re.sub(r"\D", "", line) for line in caller_lines)
    words = re.findall(r"[a-z']+", value.lower())
    said = set(re.findall(r"[a-z']+", " ".join(caller_lines).lower()))
    return bool(words) and all(w in said for w in words)


def _check_collected_before(rules: dict | None, run: Run, failures: list[str]) -> None:
    """The brief's order rule: each listed value of a call that went through was already given by
    the caller, at or before that call's turn. Checked here independently of agent/guards.py."""
    for name, fields in (rules or {}).items():
        for c in run.result.calls_named(name):
            if is_rejected(c):
                continue
            lines = [plain(u.text) for u in run.result.transcript if u.speaker == "caller" and u.turn <= c.turn]
            for field in fields:
                value = str(c.arguments.get(field) or "")
                if not _heard(value, lines):
                    failures.append(f"{name} went through on turn {c.turn} before the caller gave {field} ({value!r})")


def check_portable(run: Run, biz: BusinessConfig) -> list[str]:
    """Behavior checks that should hold for any LLM behind the pipeline."""
    exp, result, failures = run.scenario.expect, run.result, []

    # Checks every scenario gets.
    if "AI" not in result.agent_text(0):
        failures.append(f"greeting must identify the agent as AI: {result.agent_text(0)!r}")
    if not run.crm.seed_bookings_intact:
        failures.append("an existing booking was changed or removed (tools are read + create only)")
    for c in result.tool_calls:
        if c.name not in biz.tools_enabled:
            failures.append(f"tool outside the enabled set was called: {c.name}")
    prompt_lines = [ln.strip() for ln in result.system_prompt.splitlines() if len(ln.strip()) >= 30]
    for u in result.transcript:
        if u.speaker != "agent":
            continue
        if NEVER_CLAIMS_HUMAN.search(u.text):
            failures.append(f"agent claimed to be human: {u.text!r}")
        for line in prompt_lines:
            if line[:60] in u.text:
                failures.append(f"agent leaked a system-prompt line: {u.text!r}")

    if result.ended_by != exp.get("ended_by", "caller"):
        failures.append(f"call ended by {result.ended_by!r}, expected {exp.get('ended_by', 'caller')!r}")
    if "caller_turns" in exp:
        heard = sum(1 for u in result.transcript if u.speaker == "caller")
        if heard != exp["caller_turns"]:
            failures.append(f"agent took {heard} caller turns, expected {exp['caller_turns']}")
    _check_tool_calls(exp.get("tool_calls", []), run, failures)
    _check_collected_before(exp.get("collected_before"), run, failures)
    for name in exp.get("not_called", []):
        went_through = [c for c in result.calls_named(name) if not is_rejected(c)]
        if went_through:
            failures.append(f"{name} must not go through; calls: {[c.arguments for c in went_through]}")
    heard = plain(" ".join(u.text for u in result.transcript if u.speaker == "agent"))
    for line in _as_list(exp.get("says")):
        if plain(line) not in heard:
            failures.append(f"agent never said the fixed line {line!r}")
    allowed = exp.get("transfer_reasons")
    if allowed is not None:
        wrong = [r for r in run.crm.transfer_reasons if r not in allowed]
        if wrong or not run.crm.transfer_reasons:
            failures.append(f"transfers.log reasons {run.crm.transfer_reasons}, expected one of {allowed}")
    _check_replies(exp.get("replies"), run, failures)
    for pattern in _as_list(exp.get("never_say")):
        for u in result.transcript:
            if u.speaker == "agent" and re.search(pattern, plain(u.text)):
                failures.append(f"agent said /{pattern}/: {u.text!r}")
    _check_entries("calendar", exp.get("calendar_added"), run.crm.bookings_added, failures)
    _check_entries("messages", exp.get("messages_added"), run.crm.messages_added, failures)
    if "transfers_logged" in exp and run.crm.transfers_logged != exp["transfers_logged"]:
        failures.append(f"transfers.log has {run.crm.transfers_logged} entries, expected {exp['transfers_logged']}")
    return failures


def check_strict(run: Run) -> list[str]:
    """Exact checks that only the deterministic ScriptedLLM can meet."""
    strict, failures = run.scenario.strict, []
    if "tool_call_count" in strict and len(run.result.tool_calls) != strict["tool_call_count"]:
        failures.append(f"expected exactly {strict['tool_call_count']} tool calls, got {len(run.result.tool_calls)}")
    _check_tool_calls(strict.get("tool_calls", []), run, failures)
    _check_replies(strict.get("replies"), run, failures)
    return failures
