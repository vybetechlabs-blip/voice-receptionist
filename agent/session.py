"""A text call: typed caller lines in, printed (or recorded) agent replies out.

    caller -> TextInputSTT -> user context -> LLM (+ tools) -> TextOutputTTS -> assistant context

The same function runs the interactive text runner and every eval scenario. The voice version of
this pipeline lives in agent/voice.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from agent.config import BusinessConfig
from agent.core import CallResult, UsageObserver, Utterance, build_core
from agent.providers import make_stt, make_tts
from agent.text_io import Caller, TextIO, TurnTracker

__all__ = ["CallResult", "Utterance", "run_call"]


async def run_call(
    biz: BusinessConfig,
    *,
    caller: Caller,
    llm_provider: str,
    data_dir: Path,
    clock: Callable[[], datetime] = datetime.now,
    speak: Callable[[str], Awaitable[None] | None] | None = None,
    turn_timeout_s: float = 30.0,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> CallResult:
    """Run one text call to completion and return its transcript and tool calls."""
    turns = TurnTracker()
    turn = 0
    core = build_core(
        biz,
        llm_provider=llm_provider,
        data_dir=data_dir,
        clock=clock,
        current_turn=lambda: turn,
        on_call_ending=lambda reason: turns.end_call(),
        model=model,
        reasoning_effort=reasoning_effort,
    )
    result = core.result

    def on_caller_line(line: str) -> None:
        nonlocal turn
        turn += 1
        result.transcript.append(Utterance("caller", line, turn))

    async def on_agent_text(text: str) -> None:
        result.transcript.append(Utterance("agent", text, turn))
        if speak:
            out = speak(text)
            if asyncio.iscoroutine(out):
                await out

    def on_timeout() -> None:
        result.ended_by = "timeout"

    text_io = TextIO(caller, turns, on_caller_line, on_agent_text, turn_timeout_s, on_timeout)
    stt = make_stt(biz, "text", text_io=text_io)
    tts = make_tts(biz, "text", text_io=text_io)

    # Text mode has no audio, so no turn detection: each typed line is a whole turn. External
    # strategies keep the aggregator from loading the VAD and Smart Turn models it would never use.
    # Voice calls (agent/voice.py) use Pipecat's default: Silero VAD + Smart Turn v3.
    user_params = LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies())
    user_context, assistant_context = LLMContextAggregatorPair(core.context, user_params=user_params)

    pipeline = Pipeline([stt, user_context, core.llm, tts, assistant_context])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[UsageObserver(result.usage)],
        enable_rtvi=False,
        idle_timeout_secs=None,
    )
    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker, frame):
        # The LLM failed (wrong model id, bad key, unsupported parameter): end the call now rather
        # than let every turn wait out its timeout. Errors from other processors are left alone.
        if getattr(frame, "processor", None) is core.llm:
            result.ended_by = "error"
            result.error = str(getattr(frame, "error", frame))[:500]
            turns.end_call()
            await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await worker.queue_frame(TTSSpeakFrame(biz.greeting))
    await runner.run()
    return result
