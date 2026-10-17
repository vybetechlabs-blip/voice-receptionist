"""Summarize the committed voice-latency probes in eval/results/latency/.

    uv run python -m eval.latency_summary

Each JSON file is one scripted call made with eval/voice_probe.py against a running voice server:
the time from the end of the caller's speech to the first agent audio, per caller turn. In the
`latency` script the first three caller turns are FAQ questions and the fourth is the closing
goodbye. The README's latency block is this script's output; eval/test_latency_summary.py fails
when the two differ, so the README shows only numbers this script reproduces.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

LATENCY_DIR = Path(__file__).parent / "results" / "latency"
FAQ_TURNS = 3


@dataclass(frozen=True)
class Row:
    label: str
    files: tuple[str, ...]


# Every row: Piper voice, Whisper base.en, gpt-oss:20b, one machine. Each row adds one change.
PIPER_ROWS = (
    Row("Whisper on CPU, FAQ through a tool call", ("20260925-104150-latency.json",)),
    Row("+ GPU Whisper", ("20260925-113548-latency.json", "20260925-113657-latency.json", "20260925-113802-latency.json")),
    Row(
        "+ FAQ in the prompt, transfer guard, fixed spoken lines, Smart Turn fallback 1.5 s",
        (
            "20260925-123218-latency.json",
            "20260925-123323-latency.json",
            "20260925-123425-latency.json",
            "20260925-123555-latency.json",
            "20260925-123710-latency.json",
        ),
    ),
    Row(
        "+ Whisper warm-up, end_call guard, transfer reasons from the caller's words",
        (
            "20260925-151022-latency.json",
            "20260925-151132-latency.json",
            "20260925-151237-latency.json",
            "20260925-151341-latency.json",
            "20260925-151555-latency.json",
            "20260925-151701-latency.json",
            "20260925-151809-latency.json",
            "20260925-151919-latency.json",
        ),
    ),
)

# One call each on the first stack, which used Kokoro and predates every row above.
KOKORO_FIRST_STACK = (
    Row("scripted LLM (rules, no model)", ("20260925-103529-booking.json",)),
    Row("gpt-oss:20b", ("20260925-103806-latency.json",)),
)

BARGE_IN_FILES = ("20260925-123808-barge-in.json", "20260925-123847-barge-in.json", "20260925-123926-barge-in.json")


@dataclass(frozen=True)
class RowStats:
    label: str
    calls: int
    faq_turns: int
    faq_min: float
    faq_max: float
    faq_median: float
    closing_min: float
    closing_max: float


def _load(name: str) -> dict:
    return json.loads((LATENCY_DIR / name).read_text(encoding="utf-8"))


def _turn_latencies(name: str) -> list[float | None]:
    return [turn["latency_s"] for turn in _load(name)["turns"]]


def summarize(rows: tuple[Row, ...] = PIPER_ROWS) -> list[RowStats]:
    stats = []
    for row in rows:
        faq: list[float] = []
        closing: list[float] = []
        for name in row.files:
            turns = _turn_latencies(name)
            faq += [t for t in turns[:FAQ_TURNS] if t is not None]
            closing += [t for t in turns[FAQ_TURNS:] if t is not None]
        stats.append(
            RowStats(row.label, len(row.files), len(faq), min(faq), max(faq), statistics.median(faq), min(closing), max(closing))
        )
    return stats


def _span(low: float, high: float) -> str:
    return f"{low:.2f}" if f"{low:.2f}" == f"{high:.2f}" else f"{low:.2f} to {high:.2f}"


def render_table() -> str:
    lines = [
        "| Configuration (each row adds to the one above) | Calls | FAQ turns | FAQ turn range (s) | FAQ turn median (s) | Closing turn (s) |",
        "|---|---|---|---|---|---|",
    ]
    for s in summarize():
        lines.append(
            f"| {s.label} | {s.calls} | {s.faq_turns} | {_span(s.faq_min, s.faq_max)} | {s.faq_median:.2f} | {_span(s.closing_min, s.closing_max)} |"
        )
    return "\n".join(lines)


def render_notes() -> str:
    barge = [_load(name)["barge_in"] for name in BARGE_IN_FILES]
    stopped = [b["agent_audio_stopped_after_caller_started_s"] for b in barge]
    reply = [b["reply_latency_s"] for b in barge]
    first = []
    for row in KOKORO_FIRST_STACK:
        turns = [t for name in row.files for t in _turn_latencies(name) if t is not None]
        first.append(f"{row.label} {_span(min(turns), max(turns))} s (median {statistics.median(turns):.2f}, {len(turns)} turns)")
    return "\n".join(
        [
            f"Barge-in, {len(barge)} calls on the \"FAQ in the prompt\" configuration: agent audio stopped {_span(min(stopped), max(stopped))} s "
            f"after the caller started speaking; the reply to the interruption took {_span(min(reply), max(reply))} s.",
            "",
            "Before the changes above, one call each on the first stack (Kokoro voice, Whisper on CPU) measured: " + "; ".join(first)
            + ". That stack is not the current configuration, so Kokoro voice latency on the current code is unmeasured.",
        ]
    )


def render() -> str:
    return render_table() + "\n\n" + render_notes()


def main() -> int:
    print(render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
