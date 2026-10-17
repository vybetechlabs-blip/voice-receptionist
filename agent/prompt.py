"""Render prompts/receptionist.md from the business config."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from agent.config import REPO_ROOT, WEEKDAYS, BusinessConfig

PROMPT_PATH = REPO_ROOT / "prompts" / "receptionist.md"
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _clock(t) -> str:
    hour = t.hour % 12 or 12
    text = f"{hour}" if t.minute == 0 else f"{hour}:{t.minute:02d}"
    return f"{text} {'AM' if t.hour < 12 else 'PM'}"


def describe_hours(biz: BusinessConfig) -> str:
    parts = []
    for day in WEEKDAYS:
        h = biz.hours[day]
        label = day.capitalize()
        parts.append(f"{label} closed" if h is None else f"{label} {_clock(h.open)} to {_clock(h.close)}")
    return "; ".join(parts) + "."


def describe_services(biz: BusinessConfig) -> str:
    return "; ".join(f"{s.name} ({s.duration_minutes} minutes)" for s in biz.services) + "."


NEVER_GUESS = "Never guess prices, fees, timing, or policies, even if you think you know."


def answering_rule(biz: BusinessConfig) -> str:
    """How the agent answers questions: from the lookup_faq tool, or from the FAQ in this prompt."""
    if "lookup_faq" in biz.tools_enabled:
        return (
            "Answer questions with lookup_faq. Say only what the tool returns. If it returns nothing, "
            f"say you don't have that information and offer to take a message for a callback. {NEVER_GUESS}"
        )
    return (
        "Answer questions only from the FAQ below, in its words. If the FAQ doesn't cover it, say you "
        f"don't have that information and offer to take a message for a callback. {NEVER_GUESS}"
    )


def faq_block(biz: BusinessConfig) -> str:
    """The FAQ inline (lookup_faq off), so answering a question costs no tool round trip."""
    if "lookup_faq" in biz.tools_enabled:
        return ""
    lines = ["", "FAQ (the only facts you may give about prices, fees, timing, and policies):"]
    lines += [f"- {f.question} {f.answer}" for f in biz.faq]
    return chr(10).join(lines)


def render_system_prompt(biz: BusinessConfig, now: datetime, path: Path = PROMPT_PATH) -> str:
    values = {
        "business_name": biz.spoken_name,
        "business_kind": biz.kind,
        "today": f"{now.strftime('%A, %B')} {now.day}, {now.year} ({now.date().isoformat()})",
        "hours": describe_hours(biz),
        "services": describe_services(biz),
        "answering": answering_rule(biz),
        "faq_block": faq_block(biz),
    }
    template = path.read_text(encoding="utf-8")
    missing = sorted({m for m in _PLACEHOLDER.findall(template) if m not in values})
    if missing:
        raise KeyError(f"{path.name}: no value for placeholder(s) {missing}")
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template).strip()
