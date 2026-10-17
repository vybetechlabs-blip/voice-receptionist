"""book_appointment: check the mock calendar and write the visit, or offer open alternatives."""

from __future__ import annotations

import re
from datetime import date as Date
from datetime import datetime, timedelta
from datetime import time as Time
from typing import TYPE_CHECKING

from tools.env import get_env
from tools.matching import digits, matched_phrases
from tools.store import CALENDAR, next_id, read_json, write_json

if TYPE_CHECKING:
    from agent.config import BusinessConfig, Service

WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def book_appointment(
    date: str, time: str, service: str, caller_name: str, callback_number: str
) -> dict:
    """Book a service visit, or get open alternatives if the requested time is not available.

    Only call this after the caller has given their name and a callback number.

    Args:
        date: Visit date as YYYY-MM-DD.
        time: Start time as HH:MM on a 24-hour clock, for example 14:00.
        service: The service to book, using one of the business's service names.
        caller_name: The caller's name.
        callback_number: A phone number where the office can reach the caller.
    """
    env = get_env()
    biz = env.business
    errors: list[str] = []

    if not caller_name.strip():
        errors.append("caller_name is missing: ask the caller for their name")
    if len(digits(callback_number)) < 7:
        errors.append("callback_number is missing or too short: ask for a phone number")
    svc = find_service(biz, service) if service.strip() else None
    if svc is None:
        errors.append("service must be one of: " + ", ".join(s.name for s in biz.services))
    try:
        day = Date.fromisoformat(date.strip())
    except ValueError:
        errors.append("date must be YYYY-MM-DD")
        day = None
    start_time = _parse_hhmm(time)
    if start_time is None:
        errors.append("time must be HH:MM on a 24-hour clock")
    if errors:
        return {"status": "invalid", "errors": errors}

    start = datetime.combine(day, start_time)
    now = env.now()
    path = env.data_dir / CALENDAR
    calendar = read_json(path)
    bookings: list[dict] = calendar["bookings"]

    problem = slot_problem(biz, bookings, start, svc.duration_minutes, now)
    if problem:
        alternatives = open_slots(
            biz, bookings, max(start, now), svc.duration_minutes, biz.booking.max_alternatives
        )
        return {
            "status": "unavailable",
            "reason": problem,
            "requested": {"date": day.isoformat(), "time": start.strftime("%H:%M")},
            "service": svc.name,
            "alternatives": [
                {"date": s.date().isoformat(), "time": s.strftime("%H:%M"), "spoken": spoken_slot(s)}
                for s in alternatives
            ],
        }

    booking = {
        "id": next_id("BK", bookings),
        "date": day.isoformat(),
        "time": start.strftime("%H:%M"),
        "service": svc.name,
        "duration_minutes": svc.duration_minutes,
        "caller_name": " ".join(caller_name.split()),
        "callback_number": callback_number.strip(),
        "created_at": now.isoformat(timespec="minutes"),
    }
    bookings.append(booking)
    write_json(path, calendar)
    return {"status": "booked", "booking": booking, "spoken": spoken_slot(start)}


def find_service(biz: BusinessConfig, text: str) -> Service | None:
    """Exact service name first, then the longest whole-word keyword match."""
    exact = biz.service_named(text)
    if exact:
        return exact
    best, best_len = None, 0
    for svc in biz.services:
        for phrase in matched_phrases(text, (*svc.keywords, svc.name.lower())):
            if len(phrase) > best_len:
                best, best_len = svc, len(phrase)
    return best


def slot_problem(
    biz: BusinessConfig, bookings: list[dict], start: datetime, duration: int, now: datetime
) -> str | None:
    """Why ``start`` cannot be booked, or None when it is open."""
    end = start + timedelta(minutes=duration)
    weekday = WEEKDAY_NAMES[start.weekday()]
    hours = biz.hours[weekday]
    if start < now:
        return "that time has already passed"
    if hours is None:
        return f"the office is closed on {weekday.capitalize()}s"
    opens = datetime.combine(start.date(), hours.open)
    closes = datetime.combine(start.date(), hours.close)
    if start < opens or end > closes:
        return f"that is outside business hours on {weekday.capitalize()}"
    if (start - opens).seconds // 60 % biz.booking.slot_minutes:
        return f"visits start every {biz.booking.slot_minutes} minutes from opening"
    if _overlaps(bookings, start, end):
        return "that time is already booked"
    return None


def open_slots(
    biz: BusinessConfig, bookings: list[dict], after: datetime, duration: int, limit: int
) -> list[datetime]:
    """The first ``limit`` open start times at or after ``after``, within the search window."""
    found: list[datetime] = []
    step = timedelta(minutes=biz.booking.slot_minutes)
    length = timedelta(minutes=duration)
    for offset in range(biz.booking.search_days + 1):
        day = after.date() + timedelta(days=offset)
        hours = biz.hours[WEEKDAY_NAMES[day.weekday()]]
        if hours is None:
            continue
        t = datetime.combine(day, hours.open)
        closes = datetime.combine(day, hours.close)
        while t + length <= closes and len(found) < limit:
            if t >= after and not _overlaps(bookings, t, t + length):
                found.append(t)
            t += step
        if len(found) >= limit:
            break
    return found


def spoken_slot(when: datetime) -> str:
    """'Tuesday, October 6 at 11 AM' (no zero padding, which Windows strftime can't drop)."""
    hour = when.hour % 12 or 12
    clock = f"{hour}" if when.minute == 0 else f"{hour}:{when.minute:02d}"
    suffix = "AM" if when.hour < 12 else "PM"
    return f"{when.strftime('%A, %B')} {when.day} at {clock} {suffix}"


def _overlaps(bookings: list[dict], start: datetime, end: datetime) -> bool:
    for b in bookings:
        b_start = datetime.combine(Date.fromisoformat(b["date"]), _parse_hhmm(b["time"]))
        b_end = b_start + timedelta(minutes=int(b["duration_minutes"]))
        if start < b_end and b_start < end:
            return True
    return False


def _parse_hhmm(value: str) -> Time | None:
    m = _HHMM.match(value.strip())
    return Time(int(m.group(1)), int(m.group(2))) if m else None
