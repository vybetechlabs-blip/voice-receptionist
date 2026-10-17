"""take_message: append a caller's message to the mock CRM."""

from __future__ import annotations

from tools.env import get_env
from tools.matching import digits
from tools.store import MESSAGES, next_id, read_json, write_json


def take_message(caller_name: str, callback_number: str, message: str) -> dict:
    """Save a message for the office to call the caller back.

    Only call this after the caller has given their name and a callback number.

    Args:
        caller_name: The caller's name.
        callback_number: A phone number where the office can reach the caller.
        message: What the caller wants the office to know, in the caller's words.
    """
    env = get_env()
    errors: list[str] = []
    if not caller_name.strip():
        errors.append("caller_name is missing: ask the caller for their name")
    if len(digits(callback_number)) < 7:
        errors.append("callback_number is missing or too short: ask for a phone number")
    if not message.strip():
        errors.append("message is empty: ask what the caller wants the office to know")
    if errors:
        return {"status": "invalid", "errors": errors}

    path = env.data_dir / MESSAGES
    store = read_json(path)
    entry = {
        "id": next_id("MSG", store["messages"]),
        "caller_name": " ".join(caller_name.split()),
        "callback_number": callback_number.strip(),
        "message": " ".join(message.split()),
        "created_at": env.now().isoformat(timespec="minutes"),
    }
    store["messages"].append(entry)
    write_json(path, store)
    return {"status": "saved", "message": entry}
