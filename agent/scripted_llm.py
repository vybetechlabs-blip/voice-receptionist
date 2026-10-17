"""ScriptedLLM: a deterministic placeholder for the LLM stage.

It reads the same LLMContext a real model reads and answers through the same Pipecat machinery:
text frames for speech, real function calls for tools. The reply is chosen by keyword rules keyed
to intent (booking, message, question, person, out of scope, injection), so the pipeline, the tool
contracts, and the eval harness can be tested without a model or a key.

It is a test double, not a dialogue manager to ship. Its rules cover the eval scenarios and
ordinary phrasings of them, nothing more.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from datetime import time as Time
from itertools import count

from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from agent.config import WEEKDAYS, BusinessConfig
from agent.guards import is_goodbye
from tools.booking import find_service
from tools.dates import extract_date  # shared with the resolve_date tool
from tools.faq import match_faq
from tools.matching import digits

# ---- intent patterns (checked against the caller's latest utterance, in this order) ----------

INJECTION = re.compile(
    r"\b(ignore|disregard|forget)\b.{0,30}\b(instructions|rules|prompt|programming)\b"
    r"|\bsystem prompt\b|\bdeveloper mode\b|\byou are now\b|\bpretend (to be|you are)\b"
    r"|\breveal\b.{0,20}\b(prompt|instructions)\b",
    re.I,
)
IDENTITY = re.compile(
    r"\b(are you|am i (talking|speaking) (to|with)|is this)\b.{0,12}"
    r"\b(real|human|person|robot|bot|machine|ai|computer|recording)\b",
    re.I,
)
HUMAN = re.compile(
    r"\b(talk|speak)\s+(to|with)\s+(a |an |the |some )?(real )?"
    r"(person|human|someone|somebody|representative|manager|owner|people)\b"
    r"|\b(real person|human being|live person|representative|operator)\b|\btransfer me\b",
    re.I,
)
OUT_OF_SCOPE = re.compile(
    r"\b(cancel\w*|refund\w*|reschedul\w*|discount\w*|coupon\w*|money back|delete|"
    r"(change|move) (my|the) (appointment|booking|visit))\b",
    re.I,
)
MESSAGE_INTENT = re.compile(
    r"\b(leave|take) a message\b|\bpass (a message|it|that|this) along\b|\bhave (someone|somebody) call\b",
    re.I,
)
BOOKING_VERBS = re.compile(
    r"\b(book\w*|schedul\w*|appointment|set up a visit|come out|come look|come by|"
    r"send (someone|somebody|a tech\w*)|openings?|availability)\b",
    re.I,
)
QUESTION_START = re.compile(
    r"^\W*(what|when|where|how|do|does|are|is|can|could|will|would|who|which|why)\b", re.I
)
YES = re.compile(
    r"^\W*(yes|yeah|yep|sure|please|ok|okay|sounds good|that works|that would be great)\b", re.I
)
CLOSING = re.compile(
    r"\b(that's all|that is all|that's it|nothing else|no thanks|no thank you|bye|goodbye)\b", re.I
)
PICK_FIRST = re.compile(r"\b(the first one|first one|earliest|the earliest one)\b", re.I)

ASK_NAME = "Can I get your name?"
ASK_NUMBER = "And what's the best callback number for you?"
ASK_DAY = "What day works for you?"
ASK_MESSAGE = "Sure. What would you like me to pass along?"

_TIME_AMPM = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s?m\b\.?", re.I)
_TIME_COLON = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_TIME_BARE = re.compile(r"\b(?:at|around|by|for)\s+(\d{1,2})\b(?!\s*(?:[:/]|am|pm|a\.m|p\.m))|\b(\d{1,2})\s*o'?clock\b", re.I)
_PHONE = re.compile(r"(?:\+?1[\s.-]*)?(?:\(?\d{3}\)?[\s.-]*)?\d{3}[\s.-]*\d{4}\b")
# Intro words match in any case; the name itself must be capitalized ("I'm calling" is not a name).
_NAME_INTRO = re.compile(
    r"\b(?i:my name is|my name's|name is|this is|it's|it is|i'm|i am)\s+"
    r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,2})"
)
_FILLER = re.compile(r"^\W*(sure|yeah|yes|ok|okay|it's|it is|this is|my name is|name's|i'm|i am)\b[\s,.]*", re.I)


# ---- conversation events ------------------------------------------------------------------------


@dataclass
class Event:
    kind: str  # "user" | "agent" | "call" | "result"
    text: str = ""
    name: str = ""
    args: dict = field(default_factory=dict)
    result: object = None


def _content(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def to_events(messages: list) -> list[Event]:
    events: list[Event] = []
    call_names: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            events.append(Event("user", text=_content(msg).strip()))
        elif role == "assistant":
            if _content(msg).strip():
                events.append(Event("agent", text=_content(msg).strip()))
            for call in msg.get("tool_calls") or []:
                fn = call.get("function", {})
                call_names[call.get("id", "")] = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                events.append(Event("call", name=fn.get("name", ""), args=args))
        elif role == "tool":
            try:
                result = json.loads(_content(msg))
            except json.JSONDecodeError:
                result = _content(msg)
            events.append(Event("result", name=call_names.get(msg.get("tool_call_id", ""), ""), result=result))
    return events


# ---- slot extraction ----------------------------------------------------------------------------


def extract_phone(text: str) -> str | None:
    m = _PHONE.search(text)
    return m.group(0).strip() if m and len(digits(m.group(0))) >= 7 else None


def extract_name(text: str, asked: bool) -> str | None:
    m = _NAME_INTRO.search(text)
    if m:
        return m.group(1)
    if asked:
        rest = _FILLER.sub("", _PHONE.sub("", text)).strip(" ,.!")
        words = rest.split()
        if 1 <= len(words) <= 3 and all(w[:1].isupper() for w in words):
            return " ".join(words)
    return None


def extract_time(text: str) -> Time | None:
    if m := _TIME_AMPM.search(text):
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        if 1 <= hour <= 12 and minute < 60:
            hour = hour % 12 + (12 if m.group(3).lower() == "p" else 0)
            return Time(hour, minute)
    if re.search(r"\bnoon\b", text, re.I):
        return Time(12, 0)
    if m := _TIME_COLON.search(text):
        return _infer_ampm(int(m.group(1)), int(m.group(2)))
    if m := _TIME_BARE.search(text):
        return _infer_ampm(int(m.group(1) or m.group(2)), 0)
    return None


def _infer_ampm(hour: int, minute: int) -> Time | None:
    if hour > 23:
        return None
    if 1 <= hour <= 6:  # "at 2" during business hours means 2 PM
        hour += 12
    return Time(hour, minute)


# ---- the decision ------------------------------------------------------------------------------


@dataclass
class Reply:
    text: str | None = None
    tool: str | None = None
    args: dict | None = None


@dataclass
class Flow:
    kind: str  # "booking" | "message"
    start: int  # index into events
    message_text: str = ""


def decide(messages: list, biz: BusinessConfig, now: datetime) -> Reply:
    """Choose the next reply or tool call from the conversation so far."""
    events = to_events(messages)
    if not events:
        return Reply(text=None)
    last = events[-1]
    if last.kind == "result":
        return _after_tool(last, events, biz)
    if last.kind != "user":
        return Reply(text=None)

    text = last.text
    spoken = biz.spoken_name
    if not text:
        return Reply(text=_steer(spoken))
    if INJECTION.search(text):
        return Reply(text=f"I can't do that. I can help you book a service visit, answer a question about {spoken}, or take a message.")
    if IDENTITY.search(text):
        return Reply(text=f"I'm an AI assistant for {spoken}, not a person. I can book a visit, answer questions, or take a message, and I can transfer you to someone on the team anytime.")
    if HUMAN.search(text):
        return Reply(tool="transfer_call", args={"reason": "caller_request"})
    if OUT_OF_SCOPE.search(text):
        # The transfer tool speaks the fixed "can't do that myself" line for cancel/refund/billing.
        return Reply(tool="transfer_call", args={"reason": _out_of_scope_reason(text)})

    flow = _current_flow(events, biz, now)
    if flow and flow.kind == "booking":
        return _continue_booking(flow, events, biz, now)
    if flow and flow.kind == "message":
        return _continue_message(flow, events)

    if is_goodbye(text):
        return Reply(tool="end_call", args={})  # the tool speaks the fixed sign-off
    if "?" in text or QUESTION_START.search(text):
        if "lookup_faq" in biz.tools_enabled:
            return Reply(tool="lookup_faq", args={"question": text})
        entry = match_faq(biz, text)  # the FAQ is in the prompt: answer without a tool call
        return Reply(text=entry.answer if entry else biz.callback_offer)
    return Reply(text=_steer(spoken))


def _out_of_scope_reason(text: str) -> str:
    low = text.lower()
    if re.search(r"refund|money back", low):
        return "refund"
    if re.search(r"\bbill|charge|invoice", low):
        return "billing"
    if re.search(r"cancel|reschedul|change|move", low):
        return "cancel"
    return "out_of_scope"


def _steer(spoken: str) -> str:
    return f"I'm the assistant for {spoken}, so I can book a service visit, answer a question, or take a message. What can I do for you?"


def _is_booking_request(text: str, biz: BusinessConfig, today: date) -> bool:
    if BOOKING_VERBS.search(text):
        return True
    has_when = extract_date(text, today) is not None or extract_time(text) is not None
    return has_when and find_service(biz, text) is not None


def _handled_elsewhere(text: str) -> bool:
    return bool(INJECTION.search(text) or IDENTITY.search(text) or HUMAN.search(text) or OUT_OF_SCOPE.search(text))


def _current_flow(events: list[Event], biz: BusinessConfig, now: datetime) -> Flow | None:
    flow: Flow | None = None
    pending_question = ""
    for i, e in enumerate(events):
        if e.kind == "user" and not _handled_elsewhere(e.text):
            prev_agent = _previous_agent_text(events, i)
            if MESSAGE_INTENT.search(e.text):
                content = MESSAGE_INTENT.sub("", e.text)
                has_content = len(re.findall(r"[A-Za-z']+", content)) >= 6
                flow = Flow("message", i, e.text if has_content else "")
            elif _is_booking_request(e.text, biz, now.date()) and not (flow and flow.kind == "booking"):
                flow = Flow("booking", i)
            elif flow is None and YES.search(e.text) and prev_agent == biz.callback_offer:
                flow = Flow("message", i, f"Callback requested about: {pending_question}")
            elif flow and flow.kind == "message" and not flow.message_text and prev_agent == ASK_MESSAGE:
                flow.message_text = e.text
        elif e.kind == "call" and e.name == "lookup_faq":
            pending_question = str(e.args.get("question", ""))
        elif e.kind == "agent" and e.text == biz.callback_offer:
            asked = [x for x in events[:i] if x.kind == "user"]
            pending_question = asked[-1].text if asked else pending_question
        elif e.kind == "result" and isinstance(e.result, dict):
            done = (e.name == "book_appointment" and e.result.get("status") == "booked") or (
                e.name == "take_message" and e.result.get("status") == "saved"
            )
            if done:
                flow = None
    return flow


def _previous_agent_text(events: list[Event], index: int) -> str:
    for e in reversed(events[:index]):
        if e.kind == "agent":
            return e.text
        if e.kind == "user":
            return ""
    return ""


def _caller_details(events: list[Event]) -> tuple[str | None, str | None]:
    """Name and callback number from anywhere in the call (the caller shouldn't repeat them)."""
    name = number = None
    for i, e in enumerate(events):
        if e.kind == "user":
            asked_name = _previous_agent_text(events, i).endswith(ASK_NAME)
            name = extract_name(e.text, asked_name) or name
            number = extract_phone(e.text) or number
    return name, number


def _continue_booking(flow: Flow, events: list[Event], biz: BusinessConfig, now: datetime) -> Reply:
    service = when_date = when_time = None
    last_alternatives: list[dict] = []
    for e in events[flow.start:]:
        if e.kind == "user":
            service = find_service(biz, e.text) or service
            when_date = extract_date(e.text, now.date()) or when_date
            t = extract_time(e.text)
            if t is None and last_alternatives and PICK_FIRST.search(e.text):
                first = last_alternatives[0]
                when_date = date.fromisoformat(first["date"])
                t = Time.fromisoformat(first["time"])
            when_time = t or when_time
        elif e.kind == "result" and e.name == "book_appointment" and isinstance(e.result, dict):
            last_alternatives = e.result.get("alternatives") or []
    name, number = _caller_details(events)

    if service is None:
        names = ", ".join(s.name.lower() for s in biz.services)
        return Reply(text=f"What do you need us to come out for? We do {names}.")
    if when_date is None:
        return Reply(text=ASK_DAY)
    if when_time is None:
        hours = biz.hours[WEEKDAYS[when_date.weekday()]]
        day = f"{when_date.strftime('%A, %B')} {when_date.day}"
        if hours is None:
            return Reply(text=f"We're closed on {when_date.strftime('%A')}s. What other day works for you?")
        return Reply(text=f"What time works on {day}? We have visits from {_clock(hours.open)} to {_clock(hours.close)}.")
    if not name:
        return Reply(text=ASK_NAME)
    if not number:
        return Reply(text=ASK_NUMBER)
    return Reply(
        tool="book_appointment",
        args={
            "date": when_date.isoformat(),
            "time": when_time.strftime("%H:%M"),
            "service": service.name,
            "caller_name": name,
            "callback_number": number,
        },
    )


def _continue_message(flow: Flow, events: list[Event]) -> Reply:
    if not flow.message_text:
        return Reply(text=ASK_MESSAGE)
    name, number = _caller_details(events)
    if not name:
        return Reply(text=ASK_NAME)
    if not number:
        return Reply(text=ASK_NUMBER)
    return Reply(
        tool="take_message",
        args={"caller_name": name, "callback_number": number, "message": flow.message_text},
    )


def _after_tool(result_event: Event, events: list[Event], biz: BusinessConfig) -> Reply:
    name, result = result_event.name, result_event.result
    if name == "lookup_faq":
        if isinstance(result, dict) and result.get("answer"):
            return Reply(text=result["answer"])
        return Reply(text=biz.callback_offer)
    if not isinstance(result, dict):
        return Reply(text="Sorry, something went wrong on my end. I can take a message for the office.")
    status = result.get("status")
    if status == "invalid":
        errors = " ".join(result.get("errors", []))
        if "caller_name" in errors:
            return Reply(text=ASK_NAME)
        if "callback_number" in errors:
            return Reply(text=ASK_NUMBER)
        return Reply(text="Sorry, I didn't catch all of that. Could you tell me the day and time again?")
    if name == "book_appointment" and status == "booked":
        b = result["booking"]
        return Reply(text=f"You're all set: {b['service']} on {result['spoken']}. We'll call {b['callback_number']} if anything changes. Anything else I can help with?")
    if name == "book_appointment" and status == "unavailable":
        alts = result.get("alternatives") or []
        if not alts:
            return Reply(text=f"Sorry, {result['reason']}, and I don't see an opening in the next two weeks. I can take a message for the office to call you back.")
        return Reply(text=f"Sorry, {result['reason']}. The next openings are {_join_slots(alts)}. Would one of those work?")
    if name == "take_message" and status == "saved":
        m = result["message"]
        first = m["caller_name"].split()[0]
        return Reply(text=f"Thanks, {first}. I'll pass that along, and someone from the office will call you back at {m['callback_number']}. Anything else I can help with?")
    return Reply(text="Okay. Anything else I can help with?")


def _join_slots(alts: list[dict]) -> str:
    """'Tuesday, October 6 at 11 AM, 12 PM, or 3 PM' when they share a day."""
    spoken = [a["spoken"] for a in alts]
    days = [s.split(" at ")[0] for s in spoken]
    if len(set(days)) == 1 and len(spoken) > 1:
        times = [s.split(" at ")[1] for s in spoken]
        return f"{days[0]} at {_or(times)}"
    return _or(spoken)


def _or(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f", or {items[-1]}"


def _clock(t: Time) -> str:
    hour = t.hour % 12 or 12
    text = f"{hour}" if t.minute == 0 else f"{hour}:{t.minute:02d}"
    return f"{text} {'AM' if t.hour < 12 else 'PM'}"


# ---- the Pipecat service ----------------------------------------------------------------------


class ScriptedLLM(LLMService):
    """Pipecat LLM service that answers with decide(). Same frames and function calls as a model."""

    def __init__(
        self,
        biz: BusinessConfig,
        clock: Callable[[], datetime],
        system_instruction: str | None = None,
        **kwargs,
    ):
        # The prompt is carried like any provider's, though the scripted rules never read it.
        settings = LLMSettings(
            model="scripted",
            system_instruction=system_instruction,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
        )
        super().__init__(settings=settings, **kwargs)
        self._biz = biz
        self._clock = clock
        self._ids = count(1)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            reply = decide(frame.context.messages, self._biz, self._clock())
            if reply.text:
                await self.push_frame(LLMTextFrame(reply.text))
            if reply.tool:
                await self.run_function_calls(
                    [
                        FunctionCallFromLLM(
                            function_name=reply.tool,
                            tool_call_id=f"scripted_{next(self._ids)}",
                            arguments=reply.args or {},
                            context=frame.context,
                        )
                    ]
                )
        finally:
            await self.push_frame(LLMFullResponseEndFrame())
