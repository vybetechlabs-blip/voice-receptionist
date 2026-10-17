"""Scripted caller scenarios through the full text-mode pipeline.

    uv run pytest eval/                                   # everything, ScriptedLLM
    uv run pytest eval/ -m scenario --llm ollama          # portable checks against a local model
    uv run pytest eval/ -m "booking and not strict"       # one area
"""

from __future__ import annotations

import pytest

from eval.harness import check_portable, check_strict, load_scenarios

SCENARIOS = load_scenarios()


def _params(scenarios):
    return [
        pytest.param(s, id=s.id, marks=[getattr(pytest.mark, tag) for tag in s.covers])
        for s in scenarios
    ]


def test_scenario_count():
    assert 12 <= len(SCENARIOS) <= 15, "12 to 15 scenarios are expected"


@pytest.mark.scenario
@pytest.mark.parametrize("scenario", _params(SCENARIOS))
def test_portable(scenario, biz, call_runs):
    run = call_runs(scenario)
    failures = check_portable(run, biz)
    assert not failures, "\n".join(failures) + "\n--- transcript\n" + run.transcript()


@pytest.mark.scenario
@pytest.mark.strict
@pytest.mark.parametrize("scenario", _params([s for s in SCENARIOS if s.strict]))
def test_strict(scenario, llm_provider, call_runs):
    if llm_provider != "scripted":
        pytest.skip("strict checks pin ScriptedLLM wording; a real model varies")
    run = call_runs(scenario)
    failures = check_strict(run)
    assert not failures, "\n".join(failures) + "\n--- transcript\n" + run.transcript()
