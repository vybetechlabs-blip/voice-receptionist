"""What every call has, text or voice: the LLM with its tools, the context, and the call record.

Text calls (agent/session.py) and voice calls (agent/voice.py) differ only in their input and
output stages and in how turns are detected; everything between is built here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService

from agent.config import BusinessConfig
from agent.prompt import render_system_prompt
from agent.providers import make_llm
from agent.tool_bridge import ToolCall, register_tools, tools_schema
from tools import ToolEnv
from tools.store import ensure_data


@dataclass
class Utterance:
    speaker: str  # "agent" | "caller"
    text: str
    turn: int  # caller turn it belongs to; 0 is the greeting
    interrupted: bool = False  # voice only: the caller barged in while this was being spoken


@dataclass
class CallResult:
    llm: str
    transcript: list[Utterance] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    ended_by: str = "caller"  # "caller" | "transfer" | "closing" | "timeout" | "hangup" | "error"
    system_prompt: str = ""
    latency: list[dict] = field(default_factory=list)  # voice only: one record per agent reply
    usage: list[dict] = field(default_factory=list)  # LLM token usage, one record per LLM request
    error: str | None = None  # the LLM's error when ended_by == "error" (bad model id, key, parameter)

    def agent_text(self, turn: int) -> str:
        """Everything the agent said in response to caller turn ``turn`` (0 = greeting)."""
        return " ".join(u.text for u in self.transcript if u.speaker == "agent" and u.turn == turn)

    def calls_named(self, name: str) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.name == name]


class UsageObserver(BaseObserver):
    """Collects LLM token usage. Observers see a frame at every hop, so each is counted once."""

    def __init__(self, sink: list[dict]):
        super().__init__()
        self._sink = sink
        self._seen: set[int] = set()

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, MetricsFrame) and frame.id not in self._seen:
            self._seen.add(frame.id)
            for item in frame.data:
                if isinstance(item, LLMUsageMetricsData):
                    self._sink.append(item.value.model_dump(exclude_none=True))


@dataclass
class AgentCore:
    llm: LLMService
    context: LLMContext
    result: CallResult


def build_core(
    biz: BusinessConfig,
    *,
    llm_provider: str,
    data_dir: Path,
    clock: Callable[[], datetime],
    current_turn: Callable[[], int],
    on_call_ending: Callable[[str], None],
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> AgentCore:
    """Build the LLM (prompt as its system instruction), register the tools, and open a context."""
    ensure_data(data_dir)
    system_prompt = render_system_prompt(biz, clock())
    result = CallResult(llm=llm_provider, system_prompt=system_prompt)

    def ending(reason: str) -> None:
        result.ended_by = reason
        on_call_ending(reason)

    llm = make_llm(
        biz,
        provider=llm_provider,
        clock=clock,
        system_instruction=system_prompt,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    register_tools(
        llm,
        biz,
        ToolEnv(data_dir, biz, clock),
        result.tool_calls,
        current_turn=current_turn,
        on_call_ending=ending,
    )
    # The system prompt rides on the LLM service (system_instruction), per Pipecat 1.9+.
    context = LLMContext(tools=tools_schema(biz))
    return AgentCore(llm=llm, context=context, result=result)
