"""Shared pytest setup: the --llm switch, the business config, and a per-scenario call cache."""

from __future__ import annotations

import sys

import pytest
from loguru import logger

from agent.config import load_business
from agent.providers import ollama_available

logger.remove()
logger.add(sys.stderr, level="ERROR")


def pytest_addoption(parser):
    parser.addoption(
        "--llm",
        choices=("scripted", "ollama"),
        default="scripted",
        help="LLM behind the pipeline for scenario tests (default: deterministic ScriptedLLM)",
    )
    parser.addoption("--ollama-model", default=None, help="Ollama model for --llm ollama (default: config)")


@pytest.fixture(scope="session")
def biz():
    return load_business()


@pytest.fixture(scope="session")
def ollama_model(request, biz) -> str:
    return request.config.getoption("--ollama-model") or biz.providers.ollama.model


@pytest.fixture(scope="session")
def llm_provider(request, biz, ollama_model) -> str:
    provider = request.config.getoption("--llm")
    if provider == "ollama" and not ollama_available(biz.providers.ollama, ollama_model):
        pytest.skip(f"Ollama not reachable with model {ollama_model!r}")
    return provider


@pytest.fixture(scope="session")
def call_runs(biz, llm_provider, ollama_model, tmp_path_factory):
    """Play each scenario once per session; portable and strict checks share the run."""
    from eval.harness import play

    cache = {}

    def get(scenario):
        if scenario.id not in cache:
            cache[scenario.id] = play(
                scenario, biz, llm_provider, tmp_path_factory.mktemp(scenario.id), model=ollama_model
            )
        return cache[scenario.id]

    return get
