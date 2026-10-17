"""Provider factory tests: every configured stage builds, and hosted ones refuse to run without a key.

No test here makes a network call. Hosted providers are constructed with an obviously fake key
from a patched environment; construction does not contact the vendor.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from agent.config import DEFAULT_BUSINESS, load_business
from agent.providers import KEY_ENV, ProviderUnavailable, make_llm, make_stt, make_tts, openai_llm_choice, resolve_llm

pytestmark = pytest.mark.unit
FAKE_KEY = "test-key-not-real"


def _variant(tmp_path, mutate):
    raw = yaml.safe_load(DEFAULT_BUSINESS.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_business(path)


def _llm(biz, provider):
    return make_llm(biz, provider=provider, clock=lambda: None, system_instruction="x")


HOSTED = [
    ("stt", "deepgram"),
    ("stt", "openai"),
    ("tts", "cartesia"),
    ("tts", "elevenlabs"),
    ("tts", "openai"),
    ("llm", "openai"),
]


def _build(biz, stage, provider):
    return {"stt": make_stt, "tts": make_tts}[stage](biz, provider) if stage != "llm" else _llm(biz, provider)


@pytest.mark.parametrize("stage, provider", HOSTED)
def test_hosted_provider_without_key_fails_fast(biz, monkeypatch, stage, provider):
    for name in KEY_ENV.values():
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ProviderUnavailable, match=KEY_ENV[provider]):
        _build(biz, stage, provider)


@pytest.mark.parametrize("provider", ["cartesia", "elevenlabs"])
def test_hosted_tts_needs_a_voice_id(biz, monkeypatch, provider):
    monkeypatch.setenv(KEY_ENV[provider], FAKE_KEY)
    with pytest.raises(ProviderUnavailable, match=f"providers.{provider}.voice"):
        make_tts(biz, provider)


@pytest.mark.parametrize("stage, provider", HOSTED)
def test_hosted_provider_builds_with_key(tmp_path, monkeypatch, stage, provider):
    def mutate(raw):
        raw["providers"]["cartesia"]["voice"] = "test-voice"
        raw["providers"]["elevenlabs"]["voice"] = "test-voice"

    biz = _variant(tmp_path, mutate)
    monkeypatch.setenv(KEY_ENV[provider], FAKE_KEY)
    service = _build(biz, stage, provider)
    expected = {
        ("stt", "deepgram"): "DeepgramSTTService",
        ("stt", "openai"): "OpenAISTTService",
        ("tts", "cartesia"): "CartesiaTTSService",
        ("tts", "elevenlabs"): "ElevenLabsTTSService",
        ("tts", "openai"): "OpenAITTSService",
        ("llm", "openai"): "OpenAIResponsesHttpLLMService",
    }[(stage, provider)]
    assert type(service).__name__ == expected


def _cached(*paths: str) -> bool:
    return all(Path(p).expanduser().exists() for p in paths)


# The three tests below build a local speech service, which loads a downloaded model into memory.
# They are off unless asked for: run agent.fetch_models, then set VOICE_RECEPTIONIST_LOAD_MODELS=1.
needs_model_load = pytest.mark.skipif(
    os.environ.get("VOICE_RECEPTIONIST_LOAD_MODELS") != "1",
    reason="loads a speech model; set VOICE_RECEPTIONIST_LOAD_MODELS=1 after agent.fetch_models",
)


@needs_model_load
@pytest.mark.skipif(not _cached("~/.cache/huggingface/hub/models--Systran--faster-whisper-base.en"), reason="run agent.fetch_models first")
def test_local_whisper_builds(biz):
    assert type(make_stt(biz, "whisper")).__name__ == "WhisperSTTService"


@needs_model_load
@pytest.mark.skipif(not _cached("~/.cache/pipecat/kokoro-onnx/kokoro-v1.0.onnx"), reason="run agent.fetch_models first")
def test_local_kokoro_builds(biz):
    pytest.importorskip("kokoro_onnx", reason="optional extra: uv sync --extra kokoro")
    assert type(make_tts(biz, "kokoro")).__name__ == "KokoroTTSService"


@needs_model_load
@pytest.mark.skipif(not _cached("~/.cache/pipecat/piper/en_US-lessac-medium.onnx"), reason="run agent.fetch_models first")
def test_local_piper_builds(biz):
    pytest.importorskip("piper", reason="optional extra: uv sync --extra piper")
    assert type(make_tts(biz, "piper")).__name__ == "PiperTTSService"


@pytest.mark.parametrize("engine", ["kokoro", "piper"])
def test_local_tts_without_its_extra_says_how_to_install_it(biz, monkeypatch, engine):
    # A None entry in sys.modules makes the import raise ImportError, whether or not the extra is installed.
    monkeypatch.setitem(sys.modules, f"pipecat.services.{engine}.tts", None)
    with pytest.raises(ProviderUnavailable, match=f"uv sync --extra {engine}"):
        make_tts(biz, engine)


@pytest.mark.parametrize("engine", ["kokoro", "piper"])
def test_voice_server_refuses_to_start_without_the_local_tts_extra(monkeypatch, capsys, engine):
    from agent import voice_server

    monkeypatch.setattr(voice_server, "local_tts_installed", lambda provider: False)
    assert voice_server.main(["--llm", "scripted", "--tts", engine]) == 2
    assert f"uv sync --extra {engine}" in capsys.readouterr().err


def test_fetch_models_fetches_what_the_config_names(monkeypatch, capsys):
    from agent import fetch_models

    seen = []
    for name in ("whisper", "kokoro", "piper"):
        monkeypatch.setitem(fetch_models.FETCHERS, name, lambda opts, name=name: seen.append(name) or [Path(__file__)])
    assert fetch_models.main([]) == 0
    assert seen == ["whisper", "kokoro"]  # the config's STT and TTS; Piper only when asked for


@pytest.mark.parametrize("engine", ["kokoro", "piper"])
def test_fetch_models_without_the_extra_says_how_to_install_it(monkeypatch, capsys, engine):
    from agent import fetch_models

    def missing(opts):
        raise ImportError("Missing module: simulated")

    monkeypatch.setitem(fetch_models.FETCHERS, engine, missing)
    assert fetch_models.main(["--only", engine]) == 2
    assert f"uv sync --extra {engine}" in capsys.readouterr().err


def test_text_stages_need_text_io(biz):
    with pytest.raises(ProviderUnavailable, match="TextIO"):
        make_stt(biz, "text")
    with pytest.raises(ProviderUnavailable, match="TextIO"):
        make_tts(biz, "text")


def test_auto_falls_back_to_scripted_when_ollama_is_down(tmp_path):
    biz = _variant(tmp_path, lambda raw: raw["providers"]["ollama"].update(base_url="http://127.0.0.1:9/v1"))
    assert resolve_llm(biz) == "scripted"
    with pytest.raises(ProviderUnavailable, match="Ollama is not answering"):
        resolve_llm(biz, "ollama")


def test_ollama_url_avoids_localhost(biz):
    # "localhost" costs ~2 s per request on Windows (IPv6 first).
    assert "localhost" not in biz.providers.ollama.base_url


def test_voice_config_defaults(biz):
    # Kokoro is the config default; it lives in the optional [kokoro] extra (Piper is [piper]).
    assert (biz.providers.stt, biz.providers.tts) == ("whisper", "kokoro")
    assert biz.providers.options["whisper"] == {"model": "base.en", "device": "cuda", "compute_type": "float16"}


def test_voice_test_page_serves_ai_notice(biz, tmp_path):
    from fastapi.testclient import TestClient

    from agent.voice_server import ServerSettings, create_app

    settings = ServerSettings(biz, "scripted", "whisper", "kokoro", tmp_path, tmp_path)
    page = TestClient(create_app(settings)).get("/")
    assert page.status_code == 200
    assert "You are talking to an AI assistant." in page.text
    assert "Fictional business" in page.text


def test_openai_llm_uses_the_responses_api_with_model_and_effort(biz, monkeypatch):
    # gpt-6-luna supports function calling on Chat Completions only at effort "none"; the default
    # is effort "low", so the hosted LLM runs on the Responses API.
    monkeypatch.setenv(KEY_ENV["openai"], FAKE_KEY)
    luna = make_llm(biz, provider="openai", clock=lambda: None, system_instruction="x",
                    model="gpt-6-luna", reasoning_effort="low")
    assert type(luna).__name__ == "OpenAIResponsesHttpLLMService"
    assert luna._settings.model == "gpt-6-luna" and luna._settings.reasoning.effort == "low"
    mini = make_llm(biz, provider="openai", clock=lambda: None, system_instruction="x", model="gpt-4.1-mini")
    assert mini._settings.model == "gpt-4.1-mini"
    assert not getattr(mini._settings.reasoning, "effort", None)  # no reasoning parameter sent


def test_hosted_default_is_luna_low_with_explicit_prompt_caching(biz, monkeypatch):
    # gpt-6-luna at effort low is the hosted default; caching is explicit on the
    # Responses path. The request body is what Pipecat would send, built without a network call.
    monkeypatch.setenv(KEY_ENV["openai"], FAKE_KEY)
    body = _llm(biz, "openai")._build_response_params({"input": [], "instructions": "x"})
    assert body["model"] == "gpt-6-luna" and body["reasoning"] == {"effort": "low"}
    assert body["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}
    assert body["prompt_cache_key"] and body["store"] is False


def test_comparison_model_is_sent_bare(biz, monkeypatch):
    # 4.1-mini stays comparison-only and must get neither Luna's effort nor GPT-5.6+ cache options.
    monkeypatch.setenv(KEY_ENV["openai"], FAKE_KEY)
    mini = make_llm(biz, provider="openai", clock=lambda: None, system_instruction="x", model="gpt-4.1-mini")
    body = mini._build_response_params({"input": [], "instructions": "x"})
    assert body["model"] == "gpt-4.1-mini"
    assert "reasoning" not in body and not any(k.startswith("prompt_cache") for k in body)
    assert openai_llm_choice(biz, "gpt-6-luna", "medium")[1] == "medium"  # an explicit effort wins


def test_voice_server_refuses_to_start_a_hosted_llm_without_its_key(monkeypatch, capsys):
    from agent import voice_server

    monkeypatch.delenv(KEY_ENV["openai"], raising=False)
    assert voice_server.main(["--llm", "openai", "--stt", "text", "--tts", "text"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_voice_server_starts_luna_with_the_hosted_default(monkeypatch, capsys):
    # The hosted-LLM start command, wired end to end up to the web server (stubbed: nothing
    # listens, nothing reaches OpenAI).
    import uvicorn

    from agent import voice_server

    started = {}
    monkeypatch.setenv(KEY_ENV["openai"], FAKE_KEY)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.update(kw))
    argv = ["--llm", "openai", "--model", "gpt-6-luna", "--reasoning-effort", "low", "--stt", "text", "--tts", "text"]
    assert voice_server.main(argv) == 0
    banner = capsys.readouterr().out
    assert "model gpt-6-luna" in banner and "reasoning_effort low" in banner and "prompt caching explicit" in banner
    assert FAKE_KEY not in banner and started["host"] == "127.0.0.1"


def test_ollama_reasoning_effort_rides_in_the_request_extras(biz):
    llm = make_llm(biz, provider="ollama", clock=lambda: None, system_instruction="x",
                   model="gpt-oss:20b", reasoning_effort="low")
    assert llm._settings.model == "gpt-oss:20b" and llm._settings.extra["reasoning_effort"] == "low"
