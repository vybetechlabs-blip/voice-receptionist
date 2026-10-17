"""transfer_call: log a handoff to a human. No real transfer happens in this build."""

from __future__ import annotations

from typing import Literal

from tools.env import get_env
from tools.store import TRANSFERS

TransferReason = Literal["caller_request", "cancel", "refund", "billing", "out_of_scope"]
# For these the agent can't do what was asked, so the fixed "can't do that myself" line is spoken.
CANT_DO_REASONS = ("cancel", "refund", "billing")


def transfer_call(reason: TransferReason) -> dict:
    """Hand the call to a person on the team. This speaks the handoff line and ends your part of the call.

    Use reason caller_request when the caller asks for a person. Use cancel, refund, billing, or
    out_of_scope when they want something you can't do yourself. Don't say a goodbye or handoff
    line of your own; the tool speaks it.

    Args:
        reason: Why the call is handed off: caller_request, cancel, refund, billing, or out_of_scope.
    """
    env = get_env()
    biz = env.business
    line = (
        f"{env.now().isoformat(timespec='minutes')}\ttransfer\tto={biz.transfer_number}"
        f"\treason={reason}\tmode=mock-no-real-transfer\n"
    )
    with (env.data_dir / TRANSFERS).open("a", encoding="utf-8", newline="\n") as log:
        log.write(line)
    return {
        "status": "transferring",
        "reason": reason,
        "transfer_to": biz.transfer_number,
        "spoken_line": biz.cant_do_line if reason in CANT_DO_REASONS else biz.handoff_message,
        "ends_call": True,
    }
