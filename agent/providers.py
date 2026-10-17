"""Provider factory: builds each pipeline stage (STT, LLM, TTS) from the business config.

Every stage is swappable here and nowhere else.

| Stage | Local, $0                         | Hosted (your key, via environment variable)           |
|-------|-----------------------------------|-------------------------------------------------------|
| STT   | text, whisper (faster-whisper)    | deepgram (DEEPGRAM_API_KEY), openai (OPENAI_API_KEY)  |
| LLM   | scripted, ollama, auto            | openai (OPENAI_API_KEY)                               |
| TTS   | text, kokoro (ONNX), piper        | cartesia (CARTESIA_API_KEY), elevenlabs (ELEVENLABS_API_KEY), openai (OPENAI_API_KEY) |

Keys are read from the environment only, never from files, and never logged. A hosted provider
without its key fails fast with ProviderUnavailable before any network call.
"""

from __future__ import annotations

import importlib.util
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from loguru import logger
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import LLMService

from agent.config import BusinessConfig, OllamaConfig
from agent.scripted_llm import ScriptedLLM
from agent.text_io import TextInputSTT, TextIO, TextOutputTTS

KEY_ENV = {
    "deepgram": "DEEPGRAM_API_KEY",
    "openai": "OPENAI_API_KEY",
    "cartesia": "CARTESIA_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
}


class ProviderUnavailable(RuntimeError):
    """A configured provider can't be built here: missing key, voice, or local service."""


def _key(provider: str) -> str:
    name = KEY_ENV[provider]
    value = os.environ.get(name, "").strip()
    if not value:
        raise ProviderUnavailable(
            f"{provider} needs {name} in the environment. Hosted providers use a key you supply; "
            f"none are configured in this build."
        )
    return value


def _opts(biz: BusinessConfig, provider: str) -> dict:
    return dict(biz.providers.options.get(provider, {}))


def _require(opts: dict, key: str, provider: str) -> str:
    value = str(opts.get(key) or "").strip()
    if not value:
        raise ProviderUnavailable(f"providers.{provider}.{key} is not set in the business config")
    return value


LOCAL_TTS_MODULE = {"kokoro": "kokoro_onnx", "piper": "piper"}


def local_tts_installed(provider: str) -> bool:
    """True unless `provider` is a local TTS engine whose optional extra isn't installed."""
    module = LOCAL_TTS_MODULE.get(provider)
    return module is None or importlib.util.find_spec(module) is not None


def local_tts_extra_hint(provider: str) -> str:
    """Why a local TTS engine can't be built: it lives in an optional extra that isn't installed."""
    return (
        f"tts '{provider}' needs the optional '{provider}' extra, which is not installed. "
        f"Install it with: uv sync --extra {provider} (licenses: see Third-party licenses in the README)."
    )


# ---- LLM ---------------------------------------------------------------------------------------


def ollama_available(cfg: OllamaConfig, model: str | None = None, timeout_s: float = 1.5) -> bool:
    """True when Ollama answers at the configured URL and the model is pulled."""
    model = model or cfg.model
    root = cfg.base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=timeout_s) as resp:
            models = json.load(resp).get("models", [])
    except (urllib.error.URLError, OSError, ValueError):
        return False
    names = {m.get("name", "") for m in models}
    return model in names or f"{model}:latest" in names


def resolve_llm(biz: BusinessConfig, override: str | None = None, ollama_model: str | None = None) -> str:
    # ollama_model: the model to check for when the provider is Ollama
    """Turn the configured (or overridden) LLM choice into a concrete provider name."""
    choice = override or biz.providers.llm
    cfg = biz.providers.ollama
    if choice == "auto":
        return "ollama" if ollama_available(cfg, ollama_model) else "scripted"
    if choice == "ollama" and not ollama_available(cfg, ollama_model):
        raise ProviderUnavailable(
            f"Ollama is not answering at {cfg.base_url} with model "
            f"'{ollama_model or cfg.model}' pulled. Start Ollama, or use the scripted LLM."
        )
    return choice


def make_llm(
    biz: BusinessConfig,
    *,
    provider: str,
    clock: Callable[[], datetime],
    system_instruction: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> LLMService:
    """``model`` overrides the configured model; ``reasoning_effort`` is sent only when given."""
    if provider == "scripted":
        return ScriptedLLM(biz, clock, system_instruction=system_instruction)
    if provider == "ollama":
        from pipecat.services.ollama.llm import OLLamaLLMService

        cfg = biz.providers.ollama
        extra = dict(cfg.extra)
        if reasoning_effort:
            extra["reasoning_effort"] = reasoning_effort
        return OLLamaLLMService(
            base_url=cfg.base_url,
            settings=OLLamaLLMService.Settings(
                model=model or cfg.model,
                system_instruction=system_instruction,
                extra=extra,
            ),
        )
    if provider == "openai":
        # The Responses API, not Chat Completions: gpt-6-luna supports function calling on Chat
        # Completions only with reasoning effort "none" (see the README decision log). HTTP variant with
        # store off: full context each request (like the Ollama path) and nothing kept server-side.
        from pipecat.services.openai.responses.llm import (
            OpenAIResponsesHttpLLMService,
            OpenAIResponsesReasoningConfig,
        )

        chosen, effort, extra = openai_llm_choice(biz, model, reasoning_effort)
        settings = OpenAIResponsesHttpLLMService.Settings(system_instruction=system_instruction, extra=extra)
        if chosen:
            settings.model = chosen
        if effort:
            settings.reasoning = OpenAIResponsesReasoningConfig(effort=effort)
        return OpenAIResponsesHttpLLMService(api_key=_key("openai"), settings=settings)
    raise ProviderUnavailable(f"unknown LLM provider '{provider}'")


def openai_llm_choice(biz: BusinessConfig, model: str | None, reasoning_effort: str | None) -> tuple[str | None, str | None, dict]:
    """The hosted model, reasoning effort, and request extras actually sent.

    The configured model (providers.openai.llm_model) carries its configured effort and extras
    (prompt caching). A ``model`` override is sent bare, so ``--model gpt-4.1-mini`` gets no
    reasoning or caching parameter it may not accept; ``reasoning_effort`` always wins when given.
    """
    opts = _opts(biz, "openai")
    default = str(opts.get("llm_model") or "") or None
    chosen = model or default
    if chosen and chosen == default:
        return chosen, reasoning_effort or opts.get("llm_reasoning_effort") or None, dict(opts.get("llm_extra") or {})
    return chosen, reasoning_effort, {}


# ---- STT ---------------------------------------------------------------------------------------


def _expose_nvidia_dlls() -> None:
    """Let ctranslate2 find cuBLAS from the nvidia-cublas-cu12 wheel (Windows only).

    The ctranslate2 wheel bundles cuDNN but not cuBLAS, and Windows does not search site-packages
    for DLLs, so the wheel's bin folders go on both the DLL search path and PATH.
    """
    if os.name != "nt":
        return
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    for root in list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []:
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            if str(bin_dir) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def make_stt(biz: BusinessConfig, provider: str | None = None, *, text_io: TextIO | None = None) -> FrameProcessor:
    provider = provider or biz.providers.stt
    if provider == "text":
        if text_io is None:
            raise ProviderUnavailable("the text STT stage needs a TextIO (caller and turn tracker)")
        return TextInputSTT(
            text_io.caller, text_io.turns, text_io.on_caller_line, text_io.turn_timeout_s, text_io.on_timeout
        )
    opts = _opts(biz, provider)
    if provider == "whisper":
        from pipecat.services.whisper.stt import WhisperSTTService

        # device/compute_type are explicit, never "auto". CUDA needs the cuBLAS wheel
        # (nvidia-cublas-cu12, a Windows dependency in pyproject.toml).
        device, compute_type = opts.get("device", "cpu"), opts.get("compute_type", "int8")
        if device == "cuda":
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                _expose_nvidia_dlls()
            else:
                logger.warning("no CUDA GPU found; Whisper falls back to cpu/int8")
                device, compute_type = "cpu", "int8"
        return WhisperSTTService(
            device=device,
            compute_type=compute_type,
            settings=WhisperSTTService.Settings(model=opts.get("model", "base.en")),
        )
    if provider == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService

        settings = DeepgramSTTService.Settings()
        if opts.get("model"):
            settings.model = opts["model"]
        return DeepgramSTTService(api_key=_key("deepgram"), settings=settings)
    if provider == "openai":
        from pipecat.services.openai.stt import OpenAISTTService

        settings = OpenAISTTService.Settings()
        if opts.get("stt_model"):
            settings.model = opts["stt_model"]
        return OpenAISTTService(api_key=_key("openai"), settings=settings)
    raise ProviderUnavailable(f"unknown STT provider '{provider}'")


# ---- TTS ---------------------------------------------------------------------------------------


def make_tts(biz: BusinessConfig, provider: str | None = None, *, text_io: TextIO | None = None) -> FrameProcessor:
    provider = provider or biz.providers.tts
    if provider == "text":
        if text_io is None:
            raise ProviderUnavailable("the text TTS stage needs a TextIO (speaker and turn tracker)")
        return TextOutputTTS(text_io.on_agent_text, text_io.turns)
    opts = _opts(biz, provider)
    if provider == "kokoro":
        try:
            from pipecat.services.kokoro.tts import KokoroTTSService
        except ImportError as exc:
            raise ProviderUnavailable(local_tts_extra_hint("kokoro")) from exc

        return KokoroTTSService(
            settings=KokoroTTSService.Settings(voice=_require(opts, "voice", "kokoro"))
        )
    if provider == "piper":
        try:
            from pipecat.services.piper.tts import PiperTTSService
        except ImportError as exc:
            raise ProviderUnavailable(local_tts_extra_hint("piper")) from exc

        return PiperTTSService(settings=PiperTTSService.Settings(voice=_require(opts, "voice", "piper")))
    if provider == "cartesia":
        from pipecat.services.cartesia.tts import CartesiaTTSService

        key = _key("cartesia")
        settings = CartesiaTTSService.Settings(voice=_require(opts, "voice", "cartesia"))
        if opts.get("model"):
            settings.model = opts["model"]
        return CartesiaTTSService(api_key=key, settings=settings)
    if provider == "elevenlabs":
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

        key = _key("elevenlabs")
        settings = ElevenLabsTTSService.Settings(voice=_require(opts, "voice", "elevenlabs"))
        if opts.get("model"):
            settings.model = opts["model"]
        return ElevenLabsTTSService(api_key=key, settings=settings)
    if provider == "openai":
        from pipecat.services.openai.tts import OpenAITTSService

        settings = OpenAITTSService.Settings()
        if opts.get("tts_model"):
            settings.model = opts["tts_model"]
        if opts.get("tts_voice"):
            settings.voice = opts["tts_voice"]
        return OpenAITTSService(api_key=_key("openai"), settings=settings)
    raise ProviderUnavailable(f"unknown TTS provider '{provider}'")
