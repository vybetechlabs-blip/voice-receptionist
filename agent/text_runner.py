"""Talk to the receptionist by typing.

    uv run python -m agent.text_runner                 # LLM per config (auto: Ollama if up, else scripted)
    uv run python -m agent.text_runner --llm scripted  # deterministic placeholder LLM
    uv run python -m agent.text_runner --reset-data    # restore data/ from data/seed/ first

Type a line and press Enter for each caller turn. /quit (or end of input) hangs up. Input can be
piped in, one caller line per line, and the transcript prints the same way.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from agent.config import DEFAULT_BUSINESS, load_business
from agent.providers import resolve_llm
from agent.session import run_call
from agent.text_io import StdinCaller
from tools.store import DEFAULT_DATA_DIR, ensure_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent.text_runner", description=__doc__.splitlines()[0])
    parser.add_argument("--business", type=Path, default=DEFAULT_BUSINESS, help="business config YAML")
    parser.add_argument("--llm", choices=("auto", "ollama", "scripted"), help="override providers.llm")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="fake CRM folder")
    parser.add_argument("--reset-data", action="store_true", help="restore data files from data/seed/")
    parser.add_argument("--debug", action="store_true", help="show Pipecat logs")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # the Windows console default can't print every character
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "WARNING")

    biz = load_business(args.business)
    provider = resolve_llm(biz, args.llm)
    ensure_data(args.data_dir, reset=args.reset_data)

    print(f"{biz.name} (FICTIONAL) | LLM: {provider} | data: {args.data_dir}")
    print("Type as the caller. /quit hangs up.\n")

    result = asyncio.run(
        run_call(
            biz,
            caller=StdinCaller(),
            llm_provider=provider,
            data_dir=args.data_dir,
            speak=lambda text: print(f"Agent: {text}", flush=True),
            turn_timeout_s=120.0 if provider != "scripted" else 30.0,
        )
    )

    print(f"\n[call ended: {result.ended_by}]")
    for call in result.tool_calls:
        status = call.result.get("status") if isinstance(call.result, dict) else call.result
        print(f"[tool] {call.name} -> {status if status is not None else 'no FAQ match'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
