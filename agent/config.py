"""Load and validate the per-business config file (business/*.yaml).

The config is the abstraction: prompt, hours, services, FAQ, enabled tools, and pipeline providers
all come from this one file. Validation fails loudly at load time so a bad config never reaches a
call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUSINESS = REPO_ROOT / "business" / "steadfast-hvac.yaml"

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
LLM_PROVIDERS = ("auto", "ollama", "scripted", "openai")
STT_PROVIDERS = ("text", "whisper", "deepgram", "openai")
TTS_PROVIDERS = ("text", "kokoro", "piper", "cartesia", "elevenlabs", "openai")

_HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


class ConfigError(ValueError):
    """The business config file is missing a field or has an invalid value."""


@dataclass(frozen=True)
class Service:
    name: str
    duration_minutes: int
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class FaqEntry:
    id: str
    question: str
    answer: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class DayHours:
    open: time
    close: time


@dataclass(frozen=True)
class OllamaConfig:
    model: str
    base_url: str
    extra: dict = field(default_factory=dict)  # merged into each request, e.g. reasoning_effort


@dataclass(frozen=True)
class Providers:
    stt: str  # voice-mode stages; the text runner and the eval always use text in and out
    tts: str
    llm: str
    ollama: OllamaConfig
    options: dict[str, dict] = field(default_factory=dict)  # per-provider settings, e.g. whisper


@dataclass(frozen=True)
class BookingRules:
    slot_minutes: int
    max_alternatives: int
    search_days: int


@dataclass(frozen=True)
class BusinessConfig:
    name: str
    spoken_name: str
    kind: str
    greeting: str
    hours: dict[str, DayHours | None]  # weekday name -> hours, None when closed
    services: tuple[Service, ...]
    booking: BookingRules
    faq: tuple[FaqEntry, ...]
    transfer_number: str
    handoff_message: str
    cant_do_line: str  # spoken by transfer_call for cancel / refund / billing
    closing_line: str  # spoken by end_call; the model never produces the sign-off
    callback_offer: str
    tools_enabled: tuple[str, ...]
    providers: Providers
    source_path: Path = field(default=DEFAULT_BUSINESS, compare=False)

    def service_named(self, name: str) -> Service | None:
        """Exact (case-insensitive) service-name lookup."""
        wanted = name.strip().lower()
        return next((s for s in self.services if s.name.lower() == wanted), None)


def _req(mapping: dict, key: str, where: str):
    if not isinstance(mapping, dict) or key not in mapping or mapping[key] in (None, ""):
        raise ConfigError(f"{where}: missing required field '{key}'")
    return mapping[key]


def _parse_hhmm(value: str, where: str) -> time:
    m = _HHMM.match(str(value))
    if not m:
        raise ConfigError(f"{where}: '{value}' is not HH:MM (24-hour)")
    return time(int(m.group(1)), int(m.group(2)))


def _keywords(raw, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw or not all(isinstance(k, str) and k.strip() for k in raw):
        raise ConfigError(f"{where}: keywords must be a non-empty list of strings")
    return tuple(k.strip().lower() for k in raw)


def load_business(path: str | Path = DEFAULT_BUSINESS) -> BusinessConfig:
    """Read a business YAML file and return a validated BusinessConfig."""
    # Imported here: tools/ imports this module's types, so a top-level import would be circular.
    from tools import TOOLS

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    biz = _req(raw, "business", str(path))
    if biz.get("fictional") is not True:
        raise ConfigError("business.fictional must be true: this build only runs fictional businesses")
    greeting = " ".join(str(_req(biz, "greeting", "business")).split())
    if not re.search(r"\bAI\b", greeting):
        raise ConfigError("business.greeting must identify the agent as an AI assistant")

    hours_raw = _req(raw, "hours", str(path))
    hours: dict[str, DayHours | None] = {}
    for day in WEEKDAYS:
        entry = _req(hours_raw, day, "hours")
        if entry == "closed":
            hours[day] = None
            continue
        opens = _parse_hhmm(_req(entry, "open", f"hours.{day}"), f"hours.{day}.open")
        closes = _parse_hhmm(_req(entry, "close", f"hours.{day}"), f"hours.{day}.close")
        if closes <= opens:
            raise ConfigError(f"hours.{day}: close must be after open")
        hours[day] = DayHours(opens, closes)

    services = []
    for i, s in enumerate(_req(raw, "services", str(path))):
        where = f"services[{i}]"
        duration = int(_req(s, "duration_minutes", where))
        if duration <= 0:
            raise ConfigError(f"{where}: duration_minutes must be positive")
        services.append(
            Service(str(_req(s, "name", where)), duration, _keywords(s.get("keywords"), where))
        )

    b = _req(raw, "booking", str(path))
    booking = BookingRules(
        slot_minutes=int(_req(b, "slot_minutes", "booking")),
        max_alternatives=int(_req(b, "max_alternatives", "booking")),
        search_days=int(_req(b, "search_days", "booking")),
    )

    faq = []
    for i, f in enumerate(_req(raw, "faq", str(path))):
        where = f"faq[{i}]"
        faq.append(
            FaqEntry(
                id=str(_req(f, "id", where)),
                question=" ".join(str(_req(f, "question", where)).split()),
                answer=" ".join(str(_req(f, "answer", where)).split()),
                keywords=_keywords(f.get("keywords"), where),
            )
        )

    transfer = _req(raw, "transfer", str(path))

    tools_enabled = tuple(_req(raw, "tools_enabled", str(path)))
    unknown = [t for t in tools_enabled if t not in TOOLS]
    if unknown:
        raise ConfigError(f"tools_enabled: unknown tool(s) {unknown}; allowed: {sorted(TOOLS)}")

    p = _req(raw, "providers", str(path))
    o = _req(p, "ollama", "providers")
    providers = Providers(
        stt=str(_req(p, "stt", "providers")),
        tts=str(_req(p, "tts", "providers")),
        llm=str(_req(p, "llm", "providers")),
        ollama=OllamaConfig(
            model=str(_req(o, "model", "providers.ollama")),
            base_url=str(_req(o, "base_url", "providers.ollama")),
            extra=dict(o.get("extra") or {}),
        ),
        options={
            name: dict(value)
            for name, value in p.items()
            if name not in ("stt", "tts", "llm", "ollama") and isinstance(value, dict)
        },
    )
    for value, allowed, key in (
        (providers.stt, STT_PROVIDERS, "stt"),
        (providers.tts, TTS_PROVIDERS, "tts"),
        (providers.llm, LLM_PROVIDERS, "llm"),
    ):
        if value not in allowed:
            raise ConfigError(f"providers.{key}: '{value}' not one of {allowed}")

    return BusinessConfig(
        name=str(_req(biz, "name", "business")),
        spoken_name=str(biz.get("spoken_name") or biz["name"]),
        kind=str(_req(biz, "kind", "business")),
        greeting=greeting,
        hours=hours,
        services=tuple(services),
        booking=booking,
        faq=tuple(faq),
        transfer_number=str(_req(transfer, "number", "transfer")),
        handoff_message=" ".join(str(_req(transfer, "handoff_message", "transfer")).split()),
        cant_do_line=" ".join(str(_req(transfer, "cant_do_line", "transfer")).split()),
        closing_line=" ".join(str(_req(raw, "closing_line", str(path))).split()),
        callback_offer=" ".join(str(_req(raw, "callback_offer", str(path))).split()),
        tools_enabled=tools_enabled,
        providers=providers,
        source_path=path,
    )
