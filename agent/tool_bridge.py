"""Register the plain tool functions in tools/ as Pipecat LLM function calls.

The LLM-facing schema is generated from each function's signature and docstring, so the tool code
stays the single source of the contract. Every call is recorded for the eval harness.
"""

from __future__ import annotations

import inspect
import shutil
import tempfile
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docstring_parser
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import EndWorkerFrame, FunctionCallResultProperties, TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams, LLMService

from agent.config import BusinessConfig
from agent.guards import (
    caller_lines,
    caller_words,
    caller_words_errors,
    end_call_refusal,
    transfer_decision,
    unsaid,
)
from tools import TOOLS, ToolEnv, use_env

__all__ = ["FAQ_MISS", "ToolCall", "call_tool", "caller_words", "function_schema", "register_tools", "tools_schema", "unsaid"]

# What the LLM sees when lookup_faq finds nothing (the tool itself returns None).
FAQ_MISS = {
    "found": False,
    "instruction": (
        "The FAQ does not cover this question. Do not answer it from your own knowledge. "
        "Say you don't have that information and offer to take a message for a callback."
    ),
}


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: Any
    turn: int  # caller turn during which the call happened (0 = before the caller spoke)
    note: str | None = None  # for a rejected call: bad_data or a sloppy_* kind (never sent to the LLM)


def function_schema(fn: Callable) -> FunctionSchema:
    """Build a FunctionSchema from a plain function's typed signature and docstring.

    Parameters must be ``str`` or a ``Literal`` of strings (sent to the LLM as an enum).
    """
    doc = docstring_parser.parse(inspect.getdoc(fn) or "")
    param_docs = {p.arg_name: (p.description or "") for p in doc.params}
    hints = typing.get_type_hints(fn)
    properties, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        hint = hints.get(name)
        if hint is str:
            prop = {"type": "string"}
        elif typing.get_origin(hint) is typing.Literal and all(isinstance(v, str) for v in typing.get_args(hint)):
            prop = {"type": "string", "enum": list(typing.get_args(hint))}
        else:
            raise TypeError(f"{fn.__name__}.{name}: tool parameters must be str or a Literal of str")
        properties[name] = {**prop, "description": param_docs.get(name, "")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return FunctionSchema(
        name=fn.__name__,
        description=(doc.description or "").strip(),
        properties=properties,
        required=required,
    )


def tools_schema(biz: BusinessConfig) -> ToolsSchema:
    return ToolsSchema(standard_tools=[function_schema(TOOLS[name]) for name in biz.tools_enabled])


def call_tool(name: str, arguments: dict[str, Any], env: ToolEnv, said: str | list[str] | None = None) -> Any:
    """Run one tool with LLM-supplied arguments. Bad arguments come back as an 'invalid' result.

    ``said`` is what the caller has said so far (lines, or one string). When given, the guards in
    agent/guards.py run first: the fabrication guard on caller_name / callback_number, and the
    transfer guard on transfer_call. A refusal is returned and the tool never runs.
    """
    fn = TOOLS[name]
    params = inspect.signature(fn).parameters
    kwargs = {k: "" if v is None else str(v) for k, v in arguments.items() if k in params}
    missing = [k for k, p in params.items() if p.default is inspect.Parameter.empty and k not in kwargs]
    if missing:
        return {"status": "invalid", "errors": [f"missing argument(s): {', '.join(missing)}"]}
    if said is not None:
        lines = said if isinstance(said, list) else said.splitlines()
        if errors := caller_words_errors(kwargs, chr(10).join(lines)):
            return {"status": "invalid", "errors": errors}
        if name == "transfer_call":
            reason, refusal = transfer_decision(kwargs.get("reason", ""), lines)
            if refusal:
                return {"status": "refused", "errors": [refusal]}
            kwargs["reason"] = reason
        if name == "end_call" and (refusal := end_call_refusal(lines)):
            return {"status": "refused", "errors": [refusal]}
    with use_env(env):
        return fn(**kwargs)


def would_write(name: str, arguments: dict[str, Any], env: ToolEnv) -> bool:
    """Run the tool without the guards against a scratch copy of the CRM: would it have written?"""
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(env.data_dir, tmp, dirs_exist_ok=True)
        result = call_tool(name, arguments, ToolEnv(Path(tmp), env.business, env.clock), said=None)
    return isinstance(result, dict) and result.get("status") in ("booked", "saved", "transferring")


def classify_rejection(name: str, arguments: dict[str, Any], result: dict, env: ToolEnv) -> str:
    """Breakdown: bad_data (the call would have written wrong data) or harmless but sloppy."""
    errors = " ".join(result.get("errors", []))
    if name == "transfer_call" and result.get("status") == "refused":
        return "bad_data"  # an unwarranted transfer: a transfers.log entry and a dropped caller
    if name == "end_call":
        return "sloppy_early_end_call"
    if "is not something the caller said" in errors or "shorter than what the caller said" in errors:
        return "bad_data" if would_write(name, arguments, env) else "sloppy_unnecessary_call"
    return "sloppy_malformed_args"


def register_tools(
    llm: LLMService,
    biz: BusinessConfig,
    env: ToolEnv,
    calls: list[ToolCall],
    current_turn: Callable[[], int],
    on_call_ending: Callable[[str], None],
) -> None:
    """Register each enabled tool on ``llm``.

    transfer_call and end_call end the call: their fixed line (from config, never the model) is
    spoken, then the session ends. No real transfer happens in this build.
    """
    for name in biz.tools_enabled:

        async def handler(params: FunctionCallParams, name: str = name) -> None:
            arguments = dict(params.arguments)
            result = call_tool(name, arguments, env, said=caller_lines(params.context))
            note = None
            if isinstance(result, dict) and result.get("status") in ("invalid", "refused"):
                note = classify_rejection(name, arguments, result, env)
            calls.append(ToolCall(name, arguments, result, current_turn(), note))

            if isinstance(result, dict) and result.get("ends_call"):
                on_call_ending("transfer" if name == "transfer_call" else "closing")
                await params.result_callback(
                    result, properties=FunctionCallResultProperties(run_llm=False)
                )
                await params.llm.push_frame(TTSSpeakFrame(result["spoken_line"]))
                await params.llm.push_frame(EndWorkerFrame(reason=name))
                return

            await params.result_callback(FAQ_MISS if result is None else result)

        llm.register_function(name, handler)
