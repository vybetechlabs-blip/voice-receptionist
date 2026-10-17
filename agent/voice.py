"""A voice call over browser WebRTC (Pipecat's SmallWebRTC transport).

    browser mic -> transport in -> STT -> user context (Silero VAD + Smart Turn v3)
        -> LLM (+ tools) -> TTS -> transport out -> assistant context

Turn detection is Pipecat 1.11.0's default user-turn strategy: the Silero VAD starts a turn (and
interrupts the agent if it is talking: barge-in), and the Smart Turn v3.2 model decides when the
caller has finished. Both models ship inside the pipecat-ai wheel. The only change from the
default is Smart Turn's fallback silence (providers.smart_turn.stop_secs in the business config).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from agent.config import BusinessConfig
from agent.core import CallResult, UsageObserver, Utterance, build_core
from agent.providers import make_stt, make_tts


async def run_voice_call(
    biz: BusinessConfig,
    connection: SmallWebRTCConnection,
    *,
    llm_provider: str,
    data_dir: Path,
    stt_provider: str | None = None,
    tts_provider: str | None = None,
    clock: Callable[[], datetime] = datetime.now,
    model: str | None = None,
    reasoning_effort: str | None = None,
    idle_timeout_s: float = 120.0,
) -> CallResult:
    """Run one voice call until the caller hangs up, the agent transfers, or the line goes idle."""
    turn = 0
    core = build_core(
        biz,
        llm_provider=llm_provider,
        data_dir=data_dir,
        clock=clock,
        current_turn=lambda: turn,
        on_call_ending=lambda reason: None,  # the tool bridge speaks the line and ends the call
        model=model,
        reasoning_effort=reasoning_effort,
    )
    result = core.result

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    stt = make_stt(biz, stt_provider)
    tts = make_tts(biz, tts_provider)
    # Pipecat's default turn strategies (VAD starts a turn; Smart Turn v3.2 ends it), with the
    # Smart Turn fallback silence taken from config (Pipecat's default is 3.0 s).
    stop_secs = float(biz.providers.options.get("smart_turn", {}).get("stop_secs", 3.0))
    turn_strategies = UserTurnStrategies(
        stop=[
            TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=stop_secs))
            )
        ]
    )
    user_context, assistant_context = LLMContextAggregatorPair(
        core.context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer(), user_turn_strategies=turn_strategies),
    )

    @user_context.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        nonlocal turn
        if message.content and message.content.strip():
            turn += 1
            result.transcript.append(Utterance("caller", message.content.strip(), turn))
            logger.info(f"caller [{turn}] ({type(strategy).__name__}): {message.content.strip()}")

    @assistant_context.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        if message.content and message.content.strip():
            result.transcript.append(
                Utterance("agent", message.content.strip(), turn, interrupted=message.interrupted)
            )
            logger.info(f"agent  [{turn}]{' (interrupted)' if message.interrupted else ''}: {message.content.strip()}")

    latency = UserBotLatencyObserver()

    @latency.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        record = {
            "turn": turn,
            "total_secs": round(breakdown.total_secs, 3),
            "measured_from": str(breakdown.measured_from) if breakdown.measured_from else None,
            "parts": list(breakdown.turn_contribution_lines()),
        }
        result.latency.append(record)
        logger.info(f"latency [{turn}] {record['total_secs']:.3f}s | " + " | ".join(record["parts"]))

    pipeline = Pipeline(
        [transport.input(), stt, user_context, core.llm, tts, transport.output(), assistant_context]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[latency, UsageObserver(result.usage)],
        enable_rtvi=False,
        idle_timeout_secs=idle_timeout_s,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await worker.queue_frame(TTSSpeakFrame(biz.greeting))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        if result.ended_by == "caller":
            result.ended_by = "hangup"
        await worker.cancel()

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker, frame):
        # Same fail-fast as the text path (agent/session.py): an LLM error (bad key, model id,
        # quota) ends the call and shows on the server console, instead of leaving silence.
        if getattr(frame, "processor", None) is core.llm:
            result.ended_by = "error"
            result.error = str(getattr(frame, "error", frame))[:500]
            logger.error(f"LLM error, call ended: {result.error}")
            await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()
    return result
