"""Phase 0 text placeholders for the STT and TTS stages, and the callers that feed them.

TextInputSTT turns each typed caller line into a user message. Phase 0 has no audio, so there is
no turn detection: a line is a finished turn. Phase 1 swaps this stage for a real STT service and
lets the user context aggregator's Smart Turn strategy decide when the caller has finished.

TextOutputTTS "speaks" by handing each finished agent reply to a callback (the runner prints it,
the eval harness records it). It also feeds a TurnTracker, so a text caller waits for the agent's
whole turn, including any tool calls and the reply after them, before sending the next line.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from pipecat.frames.frames import (
    EndFrame,
    EndWorkerFrame,
    Frame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    StartFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class Caller(Protocol):
    async def next_line(self) -> str | None:
        """The caller's next utterance, or None when they hang up."""


class ScriptedCaller:
    """Replays a fixed list of caller lines (eval harness)."""

    def __init__(self, lines: Iterable[str]):
        self._lines = list(lines)

    async def next_line(self) -> str | None:
        return self._lines.pop(0) if self._lines else None


class StdinCaller:
    """Reads caller lines from stdin on a daemon thread, so a pending read never blocks exit."""

    HANG_UP = {"/quit", "/exit", "/hangup"}

    def __init__(self, prompt: str = "You: "):
        self._prompt = prompt
        self._interactive = sys.stdin.isatty()
        self._queue: asyncio.Queue[str | None] | None = None

    def _reader(self, loop: asyncio.AbstractEventLoop) -> None:
        for raw in sys.stdin:
            # PowerShell 5.1 prefixes piped input with a UTF-8 BOM; it must not reach the LLM.
            line = raw.rstrip("\r\n").lstrip("﻿")
            loop.call_soon_threadsafe(self._queue.put_nowait, line)
        loop.call_soon_threadsafe(self._queue.put_nowait, None)

    async def next_line(self) -> str | None:
        if self._queue is None:
            self._queue = asyncio.Queue()
            threading.Thread(target=self._reader, args=(asyncio.get_running_loop(),), daemon=True).start()
        while True:
            if self._interactive:
                print(self._prompt, end="", flush=True)
            line = await self._queue.get()
            if line is None or line.strip().lower() in self.HANG_UP:
                return None
            if line.strip():
                if not self._interactive:
                    print(f"{self._prompt}{line}", flush=True)
                return line


@dataclass
class TextIO:
    """Everything the text STT and TTS stages need, bundled for the provider factory."""

    caller: Caller
    turns: TurnTracker
    on_caller_line: Callable[[str], None]
    on_agent_text: Callable[[str], Awaitable[None] | None]
    turn_timeout_s: float
    on_timeout: Callable[[], None]


class TurnTracker:
    """Tracks whether the agent is mid-turn: LLM responding, tools running, or a rerun owed."""

    def __init__(self) -> None:
        self._llm_active = False
        self._pending_calls: set[str] = set()
        self._awaiting_rerun = False
        self._responded = False
        self._done = asyncio.Event()
        self.call_ending = False

    def caller_spoke(self) -> None:
        self._responded = False
        self._done.clear()

    def end_call(self) -> None:
        """The agent is ending the call (transfer): the caller must not send another line."""
        self.call_ending = True
        self._done.set()

    async def wait_for_agent(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._done.wait(), timeout)
            return True
        except TimeoutError:
            return False

    def observe(self, frame: Frame) -> None:
        if isinstance(frame, LLMFullResponseStartFrame):
            self._llm_active = True
            self._awaiting_rerun = False
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._llm_active = False
            self._responded = True
        elif isinstance(frame, FunctionCallsStartedFrame):
            # A system frame broadcast before the LLM response ends. The per-call
            # FunctionCallInProgressFrame comes later, from the call's own task, so it can't be
            # used to tell that the agent's turn is still going.
            self._pending_calls.update(fc.tool_call_id for fc in frame.function_calls)
        elif isinstance(frame, FunctionCallResultFrame):
            self._pending_calls.discard(frame.tool_call_id)
            run_llm = frame.run_llm
            if run_llm is None and frame.properties is not None:
                run_llm = frame.properties.run_llm
            self._awaiting_rerun = run_llm is not False
        elif isinstance(frame, TTSSpeakFrame):
            self._responded = True
        else:
            return
        if self._responded and not (self._llm_active or self._pending_calls or self._awaiting_rerun):
            self._done.set()


class TextInputSTT(FrameProcessor):
    """Placeholder STT stage: one caller line in, one user message out, after the agent's turn."""

    def __init__(
        self,
        caller: Caller,
        turns: TurnTracker,
        on_caller_line: Callable[[str], None] | None = None,
        turn_timeout_s: float = 30.0,
        on_timeout: Callable[[], None] | None = None,
    ):
        super().__init__(name="TextInputSTT")
        self._caller = caller
        self._turns = turns
        self._on_caller_line = on_caller_line
        self._turn_timeout_s = turn_timeout_s
        self._on_timeout = on_timeout
        self._task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame) and self._task is None:
            self._task = self.create_task(self._feed(), f"{self}::feed")
        elif isinstance(frame, EndFrame):
            await self._stop()

    async def cleanup(self):
        await super().cleanup()
        await self._stop()

    async def _stop(self) -> None:
        if self._task:
            await self.cancel_task(self._task)
            self._task = None

    async def _feed(self) -> None:
        while True:
            if not await self._turns.wait_for_agent(self._turn_timeout_s):
                if self._on_timeout:
                    self._on_timeout()
                break
            if self._turns.call_ending:
                return
            line = await self._caller.next_line()
            if line is None:
                break
            self._turns.caller_spoke()
            if self._on_caller_line:
                self._on_caller_line(line)
            await self.push_frame(
                LLMMessagesAppendFrame(messages=[{"role": "user", "content": line}], run_llm=True)
            )
        # Caller hung up (or the agent went silent): end the call gracefully.
        await self.push_frame(EndWorkerFrame(reason="caller hung up"))


class TextOutputTTS(FrameProcessor):
    """Placeholder TTS stage: collects each LLM reply and hands the finished text to ``speak``."""

    def __init__(self, speak: Callable[[str], Awaitable[None] | None], turns: TurnTracker):
        super().__init__(name="TextOutputTTS")
        self._speak = speak
        self._turns = turns
        self._parts: list[str] = []

    async def _say(self, text: str) -> None:
        result = self._speak(text)
        if asyncio.iscoroutine(result):
            await result

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._parts = []
        elif isinstance(frame, LLMTextFrame):
            self._parts.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            text = "".join(self._parts).strip()
            self._parts = []
            if text:
                await self._say(text)
        elif isinstance(frame, TTSSpeakFrame):
            # Spoken outside an LLM response (greeting, handoff). A real TTS reports this text to
            # the context itself; here we append it so the LLM sees what the caller heard.
            await self._say(frame.text)
            if frame.append_to_context:
                await self.push_frame(
                    LLMMessagesAppendFrame(
                        messages=[{"role": "assistant", "content": frame.text}], run_llm=False
                    ),
                    direction,
                )
            self._turns.observe(frame)
            return
        await self.push_frame(frame, direction)
        self._turns.observe(frame)
