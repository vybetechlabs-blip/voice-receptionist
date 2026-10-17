"""Download the local speech models the voice config names, and report their size and location.

    uv run python -m agent.fetch_models            # the STT and TTS engines the config names
    uv run python -m agent.fetch_models --only whisper

Pipecat would fetch these lazily on the first call; doing it here keeps downloads deliberate,
visible, and out of the latency measurements. Nothing here touches Ollama: LLM models are pulled
(or not) by you.

Kokoro and Piper are optional extras (`uv sync --extra kokoro`, `uv sync --extra piper`): their
fetchers need the extra installed and say so when it is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from agent.config import DEFAULT_BUSINESS, load_business
from agent.providers import local_tts_extra_hint


def _size(paths: list[Path]) -> str:
    total = sum(p.stat().st_size for path in paths for p in ([path] if path.is_file() else path.rglob("*")) if p.is_file())
    return f"{total / 1e6:.0f} MB"


def fetch_whisper(opts: dict) -> list[Path]:
    from faster_whisper.utils import download_model

    return [Path(download_model(opts.get("model", "base.en")))]


def fetch_kokoro(opts: dict) -> list[Path]:
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files

    model, voices = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx", KOKORO_CACHE_DIR / "voices-v1.0.bin"
    _ensure_model_files(model, voices)
    return [model, voices]


def fetch_piper(opts: dict) -> list[Path]:
    from piper.download_voices import download_voice
    from pipecat.services.piper.tts import PIPER_CACHE_DIR

    voice = opts["voice"]
    PIPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    download_voice(voice, PIPER_CACHE_DIR)
    return sorted(PIPER_CACHE_DIR.glob(f"{voice}.onnx*"))


FETCHERS = {"whisper": fetch_whisper, "kokoro": fetch_kokoro, "piper": fetch_piper}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent.fetch_models", description=__doc__.splitlines()[0])
    parser.add_argument("--business", type=Path, default=DEFAULT_BUSINESS)
    parser.add_argument("--only", choices=sorted(FETCHERS), action="append")
    args = parser.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    biz = load_business(args.business)
    configured = [stage for stage in (biz.providers.stt, biz.providers.tts) if stage in FETCHERS]
    names = args.only or configured
    if not names:
        print("nothing to fetch: the config names no local STT or TTS engine")
    for name in names:
        try:
            paths = FETCHERS[name](biz.providers.options.get(name, {}))
        except ImportError:
            if name in ("kokoro", "piper"):
                print(local_tts_extra_hint(name), file=sys.stderr)
                return 2
            raise
        print(f"{name:8s} {_size(paths):>8s}  {', '.join(str(p) for p in paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
