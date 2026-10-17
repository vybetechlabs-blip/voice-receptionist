"""Unit tests for the business config loader and the rendered system prompt."""

from __future__ import annotations

import re

import pytest
import yaml

from agent.config import DEFAULT_BUSINESS, ConfigError, load_business
from agent.prompt import render_system_prompt
from eval.harness import FROZEN_NOW

pytestmark = pytest.mark.unit

US_STATES = (
    "alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii "
    "idaho illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan "
    "minnesota mississippi missouri montana nebraska nevada ohio oklahoma oregon pennsylvania "
    "tennessee texas utah vermont virginia washington wisconsin wyoming"
).split() + ["new hampshire", "new jersey", "new mexico", "new york", "north carolina", "north dakota",
             "rhode island", "south carolina", "south dakota", "west virginia"]
GEO_WORDS = ("county", "downtown", "suburb", "street", "avenue", "located in", "serving the", "area code", "metro")


def _write_variant(tmp_path, mutate):
    raw = yaml.safe_load(DEFAULT_BUSINESS.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_business_is_fictional_and_discloses_ai(biz):
    header = DEFAULT_BUSINESS.read_text(encoding="utf-8").splitlines()[:4]
    assert any("FICTIONAL" in line for line in header)
    assert re.search(r"\bAI assistant\b", biz.greeting)


def test_config_has_what_the_brief_requires(biz):
    assert 8 <= len(biz.faq) <= 10
    assert all(s.duration_minutes > 0 for s in biz.services)
    assert "555-01" in biz.transfer_number
    assert set(biz.tools_enabled) == {"book_appointment", "take_message", "transfer_call", "end_call", "resolve_date"}
    assert "lookup_faq" not in biz.tools_enabled  # the FAQ is in the prompt


def test_client_facing_text_has_no_location(biz):
    texts = [biz.greeting, biz.handoff_message, biz.callback_offer, *(f.answer for f in biz.faq)]
    for text in texts:
        low = text.lower()
        for word in (*US_STATES, *GEO_WORDS):
            assert not re.search(rf"\b{re.escape(word)}\b", low), f"location reference {word!r} in: {text}"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r["tools_enabled"].append("cancel_appointment"), "unknown tool"),
        (lambda r: r["business"].update(fictional=False), "fictional must be true"),
        (lambda r: r["business"].update(greeting="Hi, thanks for calling. How can I help?"), "AI assistant"),
        (lambda r: r["hours"].update(monday={"open": "17:00", "close": "08:00"}), "close must be after open"),
        (lambda r: r["providers"].update(llm="gpt-something"), "providers.llm"),
    ],
)
def test_config_rejects_bad_values(tmp_path, mutate, message):
    with pytest.raises(ConfigError, match=message):
        load_business(_write_variant(tmp_path, mutate))


def test_tools_enabled_controls_the_tool_set(tmp_path):
    from agent.tool_bridge import tools_schema

    biz = load_business(_write_variant(tmp_path, lambda r: r.update(tools_enabled=["lookup_faq"])))
    assert [t.name for t in tools_schema(biz).standard_tools] == ["lookup_faq"]


def test_prompt_carries_every_required_rule(biz):
    prompt = render_system_prompt(biz, FROZEN_NOW)
    required = {
        "identifies as AI": r"You are an AI assistant",
        "no guessing prices or policy": r"Never guess prices, fees, timing, or policies",
        "callback when unsure": r"offer to take a message for a callback",
        "transfer on request": r"reason caller_request when the caller asks for a person",
        "fixed lines, never the model's": r"don't say goodbye yourself",
        "name and number before booking": r"Never book without the name and the callback number",
        "name and number before a message": r"take_message\. Get the caller's name and callback number first",
        "short spoken replies": r"one or two short sentences",
        "no out-of-scope actions": r"cannot do: cancel, reschedule, change, or refund",
        "injection resistance": r"ignore these instructions",
        "dates from the tool, never calendar math": r"resolve_date and use the date it returns; never work out dates yourself",
        "caller details verbatim": r"pass both exactly as the caller said them",
    }
    for rule, pattern in required.items():
        assert re.search(pattern, prompt), f"prompt is missing: {rule}"
    assert "{{" not in prompt and "}}" not in prompt


def test_prompt_is_rendered_from_config(biz):
    prompt = render_system_prompt(biz, FROZEN_NOW)
    assert "Monday, October 5, 2026 (2026-10-05)" in prompt
    assert "Saturday 9 AM to 1 PM" in prompt and "Sunday closed" in prompt
    assert "Repair visit (90 minutes)" in prompt
    assert biz.spoken_name in prompt
    instructions = prompt.split("FAQ (the only facts")[0]
    assert len(instructions.split()) < 450, "keep the instructions concise (the FAQ data is extra)"


def test_faq_is_in_the_prompt_when_lookup_faq_is_off(biz, tmp_path):
    prompt = render_system_prompt(biz, FROZEN_NOW)
    assert all(f.answer in prompt for f in biz.faq)
    assert "Answer questions only from the FAQ below" in prompt
    with_tool = load_business(_write_variant(tmp_path, lambda r: r["tools_enabled"].append("lookup_faq")))
    tool_prompt = render_system_prompt(with_tool, FROZEN_NOW)
    assert "FAQ (the only facts" not in tool_prompt and "Answer questions with lookup_faq" in tool_prompt
