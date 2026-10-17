"""end_call: close the call with the business's fixed sign-off. Writes nothing to the CRM."""

from __future__ import annotations

from tools.env import get_env


def end_call() -> dict:
    """End the call when the caller says goodbye or has nothing else. This speaks the closing line for you.

    Don't say goodbye yourself; the tool speaks the sign-off and hangs up.
    """
    return {"status": "ending", "spoken_line": get_env().business.closing_line, "ends_call": True}
