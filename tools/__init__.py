"""Mocked receptionist tools: plain functions with typed signatures and docstrings.

The same functions are registered as LLM function calls by agent/tool_bridge.py. The set is read
and create only (resolve_date only reads the clock; end_call writes nothing). There is deliberately no cancel, delete,
refund, or update tool; adding one is out of scope for this build.
"""

from tools.booking import book_appointment
from tools.closing import end_call
from tools.dates import resolve_date
from tools.env import ToolEnv, get_env, use_env
from tools.faq import lookup_faq
from tools.messages import take_message
from tools.transfer import transfer_call

TOOLS = {
    "book_appointment": book_appointment,
    "take_message": take_message,
    "transfer_call": transfer_call,
    "lookup_faq": lookup_faq,
    "resolve_date": resolve_date,  # read-only: calendar math in code, never in the model
    "end_call": end_call,  # writes nothing: speaks the fixed sign-off and hangs up
}

__all__ = [
    "TOOLS",
    "ToolEnv",
    "book_appointment",
    "end_call",
    "get_env",
    "lookup_faq",
    "resolve_date",
    "take_message",
    "transfer_call",
    "use_env",
]
