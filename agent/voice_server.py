"""Local voice test server: a page with a Call button, and the WebRTC signaling behind it.

    uv run python -m agent.voice_server                        # providers from config
    uv run python -m agent.voice_server --llm scripted --tts piper
    uv run python -m agent.voice_server --llm openai --model gpt-6-luna --reasoning-effort low
    then open http://127.0.0.1:7860 and press Call

The hosted LLM reads OPENAI_API_KEY from the environment of the shell that starts the server and
never logs it; the server refuses to start with --llm openai when the variable is empty.

Binds to 127.0.0.1 only. This is the local test page, not a public demo (web/ is a placeholder).
Each call's transcript, tool calls, and latency breakdown are written to logs/calls/ as JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from loguru import logger
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from agent.config import DEFAULT_BUSINESS, REPO_ROOT, BusinessConfig, load_business
from agent.providers import KEY_ENV, local_tts_extra_hint, local_tts_installed, openai_llm_choice, resolve_llm
from agent.voice import run_voice_call
from tools.store import DEFAULT_DATA_DIR

STATIC = Path(__file__).parent / "static"


@dataclasses.dataclass
class ServerSettings:
    biz: BusinessConfig
    llm: str
    stt: str
    tts: str
    data_dir: Path
    log_dir: Path
    model: str | None = None  # LLM model override for the chosen provider
    reasoning_effort: str | None = None


def _warm_ollama(settings: ServerSettings) -> None:
    """Load the model into GPU memory before the first call, so no turn pays the load time."""
    cfg = settings.biz.providers.ollama
    root = cfg.base_url.rstrip("/").removesuffix("/v1")
    body = json.dumps({"model": settings.model or cfg.model, "keep_alive": "30m"}).encode()
    request = urllib.request.Request(f"{root}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as resp:
        resp.read()


def _warm_whisper(settings: ServerSettings) -> None:
    """Pay GPU Whisper's one-time CUDA/cuBLAS setup at startup instead of on a caller's first turn.

    The cost is per process: a fresh model in an already-warm process transcribes at normal speed
    (measured 2026-09-25: 0.375 s cold vs 0.093 s warm).
    """
    import numpy as np

    from agent.providers import make_stt

    stt = make_stt(settings.biz, "whisper")
    model = getattr(stt, "_model", None)
    if model is not None:
        list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])


def llm_details(settings: ServerSettings) -> dict:
    """The LLM model, reasoning effort, and request extras the calls actually use."""
    if settings.llm == "ollama":
        return {"model": settings.model or settings.biz.providers.ollama.model, "reasoning_effort": settings.reasoning_effort}
    if settings.llm == "openai":
        model, effort, extra = openai_llm_choice(settings.biz, settings.model, settings.reasoning_effort)
        return {"model": model, "reasoning_effort": effort, "request_extras": extra}
    return {"model": settings.llm}


def create_app(settings: ServerSettings) -> FastAPI:
    handler = SmallWebRTCRequestHandler()
    calls: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.llm == "ollama":
            logger.info("warming Ollama model")
            await asyncio.to_thread(_warm_ollama, settings)
        if settings.stt == "whisper":
            logger.info("warming Whisper")
            await asyncio.to_thread(_warm_whisper, settings)
        yield
        await handler.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "call.html")

    async def run_call(connection: SmallWebRTCConnection) -> None:
        started = datetime.now()
        try:
            result = await run_voice_call(
                settings.biz,
                connection,
                llm_provider=settings.llm,
                stt_provider=settings.stt,
                tts_provider=settings.tts,
                data_dir=settings.data_dir,
                model=settings.model,
                reasoning_effort=settings.reasoning_effort,
            )
        except Exception:
            logger.exception("voice call failed")
            return
        record = {
            "started": started.isoformat(timespec="seconds"),
            "providers": {"stt": settings.stt, "llm": settings.llm, "tts": settings.tts, **llm_details(settings)},
            **{k: v for k, v in dataclasses.asdict(result).items() if k != "system_prompt"},
        }
        out = settings.log_dir / "calls" / f"{started:%Y%m%d-%H%M%S}-{connection.pc_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        logger.info(f"call ended ({result.ended_by}); record: {out}")

    @app.post("/api/offer")
    async def offer(request: SmallWebRTCRequest):
        async def on_connection(connection: SmallWebRTCConnection):
            task = asyncio.create_task(run_call(connection))
            calls.add(task)
            task.add_done_callback(calls.discard)

        return await handler.handle_web_request(request=request, webrtc_connection_callback=on_connection)

    @app.patch("/api/offer")
    async def ice_candidate(request: SmallWebRTCPatchRequest):
        await handler.handle_patch_request(request)
        return {"status": "success"}

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent.voice_server", description=__doc__.splitlines()[0])
    parser.add_argument("--business", type=Path, default=DEFAULT_BUSINESS)
    parser.add_argument("--llm", choices=("auto", "ollama", "scripted", "openai"))
    parser.add_argument("--model", help="LLM model for the chosen provider (default: config)")
    parser.add_argument("--ollama-model", help="same as --model for --llm ollama (kept for old commands)")
    parser.add_argument("--reasoning-effort", help="e.g. low (default: providers.openai.llm_reasoning_effort for the configured model)")
    parser.add_argument("--stt", help="override providers.stt")
    parser.add_argument("--tts", help="override providers.tts")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO",
               filter=lambda r: args.debug or r["name"].startswith("agent"))

    biz = load_business(args.business)
    model = args.model or args.ollama_model
    settings = ServerSettings(
        biz=biz,
        llm=resolve_llm(biz, args.llm, model if args.llm in (None, "auto", "ollama") else None),
        stt=args.stt or biz.providers.stt,
        tts=args.tts or biz.providers.tts,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        model=model,
        reasoning_effort=args.reasoning_effort,
    )
    for stage, provider in (("llm", settings.llm), ("stt", settings.stt), ("tts", settings.tts)):
        if provider in KEY_ENV and not os.environ.get(KEY_ENV[provider], "").strip():
            # Fail before the first call rather than on the caller's first turn.
            print(f"--{stage} {provider} needs {KEY_ENV[provider]} set in this shell. Not started.", file=sys.stderr)
            return 2
    if not local_tts_installed(settings.tts):
        print(f"{local_tts_extra_hint(settings.tts)} Not started.", file=sys.stderr)
        return 2
    details = llm_details(settings)
    shown = ", ".join(f"{k} {v}" for k, v in details.items() if v and k != "request_extras")
    caching = " | prompt caching explicit" if details.get("request_extras") else ""
    print(f"{biz.name} (FICTIONAL) | STT {settings.stt} | LLM {settings.llm} ({shown}){caching} | TTS {settings.tts}")
    print(f"Open http://127.0.0.1:{args.port} and press Call. Ctrl+C stops the server.")
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
