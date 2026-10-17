"""Calendar words to dates: the resolve_date tool, and the parser it shares with ScriptedLLM.

LLMs get weekday arithmetic wrong (the 2026-09-25 gpt-oss:20b rerun booked "Wednesday" on a
Sunday). resolve_date does the calendar math in code, so the model never has to.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from tools.env import get_env

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_MONTHS = {
    m: i
    for i, names in enumerate(
        (
            ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
            ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
            ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for m in names
}
_MONTH_DAY = re.compile(
    r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I
)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_SLASH_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")
_WEEKDAY = re.compile(r"\b(" + "|".join(WEEKDAYS) + r")\b", re.I)
_NEXT_WEEKDAY = re.compile(r"\bnext\s+(" + "|".join(WEEKDAYS) + r")\b", re.I)


def extract_date(text: str, today: date) -> date | None:
    """Best single reading of a day in ``text``: ISO, month + day, M/D, today, tomorrow, weekday."""
    low = text.lower()
    if m := _ISO_DATE.search(text):
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    if m := _MONTH_DAY.search(text):
        return _future(today, _MONTHS[m.group(1).lower()], int(m.group(2)))
    if m := _SLASH_DATE.search(text):
        month, day = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return _future(today, month, day)
    if re.search(r"\btomorrow\b", low):
        return today + timedelta(days=1)
    if re.search(r"\btoday\b", low):
        return today
    if m := _WEEKDAY.search(text):
        ahead = (WEEKDAYS.index(m.group(1).lower()) - today.weekday()) % 7 or 7
        return today + timedelta(days=ahead)
    return None


def _future(today: date, month: int, day: int) -> date | None:
    try:
        d = date(today.year, month, day)
        return d if d >= today else date(today.year + 1, month, day)
    except ValueError:
        return None


def spoken_day(d: date) -> str:
    return f"{d.strftime('%A, %B')} {d.day}"


def resolve_date(phrase: str) -> dict:
    """Turn the caller's words for a day into a calendar date. Use this for every date; never work out dates yourself.

    Returns the date in YYYY-MM-DD form with its weekday and whether the office is open that day,
    or asks you to clarify with the caller when the words are unclear or could mean two days.

    Args:
        phrase: The caller's words for the day, for example "tomorrow", "Wednesday", or "October 9".
    """
    env = get_env()
    today = env.now().date()
    phrase = " ".join(phrase.split())

    if m := _NEXT_WEEKDAY.search(phrase):
        # "next Wednesday" can mean the coming one or the one after; ask when both are ahead.
        coming = extract_date(m.group(1), today)
        if coming is not None and (coming - today).days < 7:
            later = coming + timedelta(days=7)
            return {
                "status": "unclear",
                "ask": f"Ask whether they mean {spoken_day(coming)} or {spoken_day(later)}.",
                "options": [coming.isoformat(), later.isoformat()],
            }
    weekday = _WEEKDAY.search(phrase)
    if weekday and not _MONTH_DAY.search(phrase) and weekday.group(1).lower() == WEEKDAYS[today.weekday()]:
        later = today + timedelta(days=7)
        return {
            "status": "unclear",
            "ask": f"Ask whether they mean today, {spoken_day(today)}, or {spoken_day(later)}.",
            "options": [today.isoformat(), later.isoformat()],
        }

    day = extract_date(phrase, today)
    if day is None:
        return {"status": "unclear", "ask": "Ask the caller which day they mean, as a weekday or a date."}
    hours = env.business.hours[WEEKDAYS[day.weekday()]]
    return {
        "status": "ok",
        "date": day.isoformat(),
        "weekday": day.strftime("%A"),
        "spoken": spoken_day(day),
        "in_the_past": day < today,
        "office_hours": "closed" if hours is None else f"{hours.open:%H:%M} to {hours.close:%H:%M}",
    }
