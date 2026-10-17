"""lookup_faq: keyword match against the business config's FAQ."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tools.env import get_env
from tools.matching import matched_phrases

if TYPE_CHECKING:
    from agent.config import BusinessConfig, FaqEntry


def lookup_faq(question: str) -> dict | None:
    """Look up the answer to a caller's question in the business FAQ.

    Returns nothing when the FAQ does not cover the question. Then do not answer from your own
    knowledge: say you don't have that information and offer a callback.

    Args:
        question: The caller's question, in their words.
    """
    best = match_faq(get_env().business, question)
    if best is None:
        return None
    return {"id": best.id, "question": best.question, "answer": best.answer}


def match_faq(biz: BusinessConfig, question: str) -> FaqEntry | None:
    """The FAQ entry with the most whole-word keyword hits in ``question``, or None."""
    best, best_score = None, 0
    for entry in biz.faq:
        score = len(matched_phrases(question, entry.keywords))
        if score > best_score:
            best, best_score = entry, score
    return best
