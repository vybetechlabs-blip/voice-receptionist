"""Whole-word keyword matching shared by lookup_faq, service lookup, and the scripted LLM."""

from __future__ import annotations

import re

_NON_WORD = re.compile(r"[^a-z0-9/'\-]+")


def normalize(text: str) -> str:
    """Lowercase and reduce punctuation to single spaces, keeping hyphens, slashes, apostrophes."""
    text = text.lower().replace("’", "'")
    return " ".join(_NON_WORD.sub(" ", text).split())


def matched_phrases(text: str, phrases: tuple[str, ...] | list[str]) -> list[str]:
    """Return the phrases that occur in ``text`` as whole words (after normalizing both)."""
    padded = f" {normalize(text)} "
    return [p for p in phrases if f" {normalize(p)} " in padded]


def digits(text: str) -> str:
    return re.sub(r"\D", "", text)
