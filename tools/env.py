"""Per-session environment for the mocked tools.

The tools keep the plain signatures the LLM sees (e.g. ``take_message(caller_name, callback_number,
message)``). What they also need (where the fake CRM lives, which business, what time it is) comes
from the environment set with ``use_env``. The pipeline sets it around every tool call; tests set it
directly.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import BusinessConfig


@dataclass(frozen=True)
class ToolEnv:
    data_dir: Path
    business: BusinessConfig
    clock: Callable[[], datetime] = field(default=datetime.now)

    def now(self) -> datetime:
        return self.clock().replace(second=0, microsecond=0)


_current: contextvars.ContextVar[ToolEnv] = contextvars.ContextVar("tool_env")


@contextlib.contextmanager
def use_env(env: ToolEnv) -> Iterator[ToolEnv]:
    token = _current.set(env)
    try:
        yield env
    finally:
        _current.reset(token)


def get_env() -> ToolEnv:
    try:
        return _current.get()
    except LookupError:
        raise RuntimeError("tool called outside use_env(...): no data dir or business set") from None
