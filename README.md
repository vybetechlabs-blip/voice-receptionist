# voice-receptionist

[![CI](https://github.com/vybetechlabs-blip/voice-receptionist/actions/workflows/ci.yml/badge.svg)](https://github.com/vybetechlabs-blip/voice-receptionist/actions/workflows/ci.yml)

## Problem and result
A phone assistant for a small business is trusted with the calendar, so a guess costs real money: a wrong day, an invented callback number, a refund nobody approved. voice-receptionist is an AI receptionist built on Pipecat 1.11.0 for one fictional business, Steadfast Heating & Air. It says up front that it is an AI, then books a visit, takes a message, or hands the call to a person. Its tools can only read and create, so there is no cancel, refund, delete or update. Guards check every tool call against what the caller actually said, and a name or number the caller never gave does not reach the calendar.

The eval has 15 scripted callers, each run 3 times (45 runs). gpt-6-luna passes 45 of 45. A local gpt-oss:20b passes 43 of 45, and its guards blocked 12 calls that would have written bad data: 10 invented caller details and 2 unwarranted transfers. On the local voice stack, FAQ answers start 1.46 s after the caller stops (median of 24 turns), under the 1.5 s target. That figure is one machine and one voice engine, so read the Latency section before relying on it. The suite has 235 tests.

### How it works
A call is a Pipecat pipeline. Speech or typed text goes to the model, the model calls tools, and the guards sit between the model and the tools.

```
text:   typed line ─► TextInputSTT ─┐                              ┌─► TextOutputTTS ─► printed
voice:  browser mic ─► WebRTC in ─► STT ─► user context ─► LLM ─► TTS ─► WebRTC out ─► browser
                                    (VAD + Smart Turn)     │ ▲
                                                           ▼ │
                                      tools/ (book_appointment, take_message,
                                              transfer_call, end_call, resolve_date) ─► data/ (fake CRM)
```

- **Config is the abstraction.** `business/steadfast-hvac.yaml` holds the greeting, hours, services, FAQ, transfer number, enabled tools and which provider runs each stage. `agent/config.py` validates it on load and refuses a business not marked fictional, a greeting without an AI disclosure, or an unknown tool.
- **Providers** are built in `agent/providers.py` and nowhere else.

  | Stage | Local | Hosted (your key, from an environment variable) |
  |---|---|---|
  | STT | `whisper` (faster-whisper; GPU float16 by default, CPU int8 fallback) | `deepgram`, `openai` |
  | LLM | `scripted` (deterministic rules, not a model), `ollama` (default model `gpt-oss:20b`), `auto` | `openai` (default `gpt-6-luna`, reasoning effort low) |
  | TTS | `kokoro` (`kokoro` extra), `piper` (`piper` extra) | `cartesia`, `elevenlabs`, `openai` |

  Without its key a hosted provider refuses before any network call. Cartesia and ElevenLabs also need a voice id from your account in the config.
- **Tools** are plain Python functions in `tools/` with typed signatures; `agent/tool_bridge.py` builds each LLM function schema from them.

  | Tool | Does | Refuses |
  |---|---|---|
  | `book_appointment(date, time, service, caller_name, callback_number)` | Books an open slot, or returns up to 3 open alternatives | Missing name or number, past times, closed days, off-grid or out-of-hours times, overlaps |
  | `take_message(caller_name, callback_number, message)` | Appends to the message log | Missing name, number or message |
  | `transfer_call(reason)` | Logs the transfer, speaks a fixed line, ends the call. `reason` is one of `caller_request`, `cancel`, `refund`, `billing`, `out_of_scope` | The transfer guard |
  | `end_call()` | Speaks the fixed closing line and ends the call | The end-call guard |
  | `resolve_date(phrase)` | Turns the caller's words for a day into an ISO date, weekday and open/closed, so the model never does calendar math | Asks for clarification when no day or two days fit |
  | `lookup_faq(question)` | Whole-word keyword match on the config FAQ. Off by default; see decision 5 | Returns nothing on a miss, and the agent offers a callback |

- **Guards** (`agent/guards.py`) run before the tool, check the call against the caller's own words, and refuse back to the model. Nothing refused reaches the fake CRM.
  - **Caller-words guard.** `caller_name` and `callback_number` must appear verbatim in what the caller said, ignoring case and spacing. A name must be the longest span they said: "Jordan" is refused when they said "Jordan Blake".
  - **Transfer guard.** `transfer_call` goes through only if the caller asked for a person or the reason is `cancel`, `refund`, `billing` or `out_of_scope`. It never goes through while the caller's latest line tries to override the agent's instructions.
  - **End-call guard.** `end_call` goes through only when the caller's latest line is a goodbye.
- **Fake CRM.** `data/seed/` is the version-controlled seed; working copies in `data/` are created from it and gitignored.

## Results
`eval/scenarios/*.yaml` holds 15 scripted callers: a valid booking, a taken slot, a missing callback number, hours, an FAQ service question, an FAQ fee, a price not in the FAQ, a message saved after a miss, a transfer request, rambling, cancel, refund and discount, prompt injection, a direct message, and "are you a person?" into a transfer. Each plays through the full text pipeline with the clock frozen at Monday 2026-10-05 09:00 and a throwaway copy of the fake CRM. Every scenario also checks that the greeting discloses AI, no seed booking changes, no tool outside the enabled set is used, the agent never claims to be human, and no system-prompt line is spoken.

Two scores per run. **Outcome-correct** is the gate: the records written, the required spoken lines, and the call ending properly. A call a guard refused counts as having changed nothing. **Clean-attempt** is reported only: no tool call was refused. Every refused call is classed as **bad data** (it would have written wrong data) or **harmless but sloppy** (an early `end_call`, an unnecessary call, malformed arguments).

| Model | Outcome-correct | Clean-attempt | Blocked: bad data | Blocked: sloppy |
|---|---|---|---|---|
| `gpt-6-luna`, effort low (hosted) | 45/45 | 45/45 | 0 | 0 |
| `gpt-4.1-mini` (hosted) | 41/45 | 45/45 | 0 | 0 |
| `gpt-oss:20b` (local) | 43/45 | 18/45 | 12 | 34 |

Reproduce the table without calling any model. The command re-scores the saved transcripts, and the full per-scenario table is in `eval/results/2026-09-26-comparison.md`:

```bash
uv run python -m eval.rescore eval/results/2026-09-25-gpt-oss-20b-enforced.json eval/results/2026-09-25-gpt-6-luna-luna.json eval/results/2026-09-25-gpt-4.1-mini-gpt41mini.json
```

- **About the 12 blocked calls.** Counting the `bad_data` entries in `2026-09-25-gpt-oss-20b-enforced.json` gives 5 `book_appointment` and 5 `take_message` calls with a name or number the caller never said, plus 2 `transfer_call` calls the guard refused (one injection, one "are you a person?" call). `eval/test_published_results.py` checks these counts.
- **About 43/45.** The gpt-oss:20b run was scored 39/45 when it ran, and its report `2026-09-25-gpt-oss-20b-enforced.md` still says so. Two checks asked for more than the requirement did (the order of the booking questions, and one exact wording of the callback offer), so they were relaxed and the saved transcripts re-scored 43/45. The checker's fingerprint, a hash of `eval/harness.py` and the scenario files, is `e674f1693299` in the comparison file, and a test fails if the checker stops matching it, so equal fingerprints mean the same checker.
- **What the results do not show.** Three runs per scenario, scripted text callers rather than audio, one fictional business. Luna's runs ran on the Responses API in text mode; its voice latency is unmeasured.

## Latency
End of caller speech to first agent audio, measured by `eval/voice_probe.py` over real WebRTC against a running voice server. **Measured once on the Piper configuration (`piper` extra), i7-8700K/RTX 4080; Kokoro and hosted-model voice latency unmeasured.** The model is `gpt-oss:20b` on the GPU, Whisper `base.en`, and each row adds one change to the row above.

<!-- latency:begin -->
| Configuration (each row adds to the one above) | Calls | FAQ turns | FAQ turn range (s) | FAQ turn median (s) | Closing turn (s) |
|---|---|---|---|---|---|
| Whisper on CPU, FAQ through a tool call | 1 | 3 | 1.38 to 2.67 | 2.16 | 4.89 |
| + GPU Whisper | 3 | 9 | 1.17 to 2.24 | 1.76 | 3.64 to 4.45 |
| + FAQ in the prompt, transfer guard, fixed spoken lines, Smart Turn fallback 1.5 s | 5 | 14 | 0.86 to 3.50 | 1.43 | 2.37 to 2.60 |
| + Whisper warm-up, end_call guard, transfer reasons from the caller's words | 8 | 24 | 0.87 to 2.89 | 1.46 | 2.29 to 2.66 |

Barge-in, 3 calls on the "FAQ in the prompt" configuration: agent audio stopped 0.56 to 0.59 s after the caller started speaking; the reply to the interruption took 1.31 to 1.39 s.

Before the changes above, one call each on the first stack (Kokoro voice, Whisper on CPU) measured: scripted LLM (rules, no model) 1.95 to 2.77 s (median 2.17, 4 turns); gpt-oss:20b 3.21 to 4.37 s (median 3.82, 4 turns). That stack is not the current configuration, so Kokoro voice latency on the current code is unmeasured.
<!-- latency:end -->

The probe files are in `eval/results/latency/` and `uv run python -m eval.latency_summary` prints the block above; `eval/test_latency_summary.py` fails if this block and the script disagree. The closing turn is slower because Smart Turn waits up to 1.5 s before ending a turn it judges unfinished (Pipecat's default is 3.0 s).

## Decision log
**1. A phased, $0 build with a latency stop rule.**
Chosen: build in phases and spend nothing until one passes. Phase 0 is a text pipeline with a scripted LLM, mocked tools and the eval, Phase 1 adds local voice and the hosted provider classes, and a public demo and a phone number come later. Stop and report when FAQ turns on the local stack run over 1.5 s. The rule fired on the first Whisper-on-CPU stack (2.16 s median on FAQ turns). Three steps followed: Whisper on the GPU (1.76 s), the FAQ moved into the prompt with the structural guards (1.43 s, under the target), and a Whisper warm-up (1.46 s).
Rejected: tuning past the stop, or moving to a hosted provider to hide the number.
Known limits: the target holds on one machine with one local model and one voice engine, and closing turns still take 2.29 to 2.66 s.

**2. A guard that checks tool arguments against the caller's own words.**
Chosen: before `book_appointment` or `take_message` runs, the name and number must appear verbatim in what the caller said, and a name must be the longest span they said. A refusal goes back to the model with an instruction to ask, or to pass the exact words.
Rejected: relying on the prompt. The first gpt-oss:20b run saved a message under an invented name and number (`eval/results/2026-09-25-gpt-oss-20b.md`), and later runs still tried 10 such calls, all blocked.
Known limits: verbatim also rejects a correctly heard number the model reformats ("555 0165" passed as "555-0165"), so the model has to retry.

**3. A transfer guard with enum reasons and fixed spoken lines.**
Chosen: `reason` is an enum in the tool schema. A transfer goes through for a request for a person or for cancel, refund, billing and out-of-scope reasons, and never while the caller's latest line tries to override the instructions. When the caller's own words say cancel, refund or billing, that becomes the reason. The hand-off, "I can't do that myself, but I can connect you with someone who can", and the closing line come from config, so the model never writes them.
Rejected: letting the model choose reason and wording. In the first gpt-oss:20b runs, prompt injection passed 0 of 3 because the model transferred the call; with the guard it passes 3 of 3, as do cancel and refund.
Known limits: the override check is a phrase list, so a new phrasing can get past it until it is added.

**4. Two scores: outcome-correct gates, clean-attempt is reported.**
Chosen: gate on what reached the records, and report separately how often a guard had to step in, with refused calls split into bad data and harmless but sloppy. `eval.rescore` re-scores any saved report on the current checker without calling a model.
Rejected: one pass/fail number, which would score a model that invented a number 12 times the same as one that never did. The table shows the difference: 43/45 with 12 blocked calls against 45/45 with none.
Known limits: relaxing the two checks moved gpt-oss:20b from 39/45 to 43/45 (see Results), and the older reports score differently today because they ran on older agent code.

**5. The FAQ lives in the prompt.**
Chosen: with 10 entries, the FAQ goes inline in the system prompt and `lookup_faq` is off by default. A tool call costs an extra model round trip. The step that moved the FAQ into the prompt, together with the transfer guard, fixed spoken lines and a 1.5 s Smart Turn fallback, took the FAQ-turn median from 1.76 s to 1.43 s (Latency table).
Rejected: a lookup tool for a FAQ this small.
Known limits: a longer FAQ makes every request larger. `lookup_faq` stays in the factory and comes back by adding it to `tools_enabled`.

**6. gpt-6-luna is the hosted default, gpt-oss:20b the local default, gpt-4.1-mini a comparison.**
Chosen: Luna at reasoning effort low, with prompt caching set explicitly, for hosted runs. It scored 45/45 with nothing blocked, and read 95.9% of its input tokens from the cache (157,869 of 164,578). Luna runs on the Responses API, because function calling on Chat Completions works for it only at reasoning effort none. gpt-oss:20b stays the default for local development because it costs nothing. gpt-4.1-mini is for comparison, selected with `--model`.
Rejected: gpt-4.1-mini as the default. It scored 41/45: two cancel requests got a message instead of a transfer, one question not in the FAQ was steered to an estimate booking, and one off-topic steer missed the check's words.
Known limits: three runs per scenario, and the amount billed for the hosted runs is not recorded. `COST-MODEL.md` has token counts and cost at published rates.

**7. Local speech engines are optional extras.**
Chosen: `uv sync` installs no GPL package, and the voice server needs one of the extras (`--extra kokoro`, `--extra piper`) or a hosted TTS. The config default stays `tts: kokoro` and fails with an install hint when its extra is absent. `eval/test_dependency_licenses.py` checks the default install.
Rejected: Kokoro in the default install. `kokoro-onnx` itself pulls `phonemizer` (GPL-3.0-or-later) and `espeakng-loader`, which bundles espeak-ng (GPL), so a Kokoro default would not have removed GPL from the default install. Piper is GPL-3.0-or-later too, so both engines moved out.
Known limits: a fresh clone cannot run the voice server until one extra is installed and its model is downloaded. Hosted TTS needs no extra.

## How AI was used
I built this to find out what it takes to let a voice agent answer a small business's phone without ever inventing a caller's details or booking the wrong day. The constraints were mine: a fictional HVAC company so no real data is involved, a $0 budget through the local build, a 1.5 second latency ceiling on the local stack, and a rule that any invented caller detail or unwarranted transfer must be blocked at the tool rather than caught later. I wrote the brief and Claude Code built the pipeline, the tools, the eval harness and the guards, stopping at each checkpoint for my review. The guards came out of what the evals exposed: the first local model invented callback numbers and names, so Claude Code proposed a fabrication guard that only accepts details the caller said verbatim, then a transfer guard with enumerated reasons after the model transferred calls it should have declined; I accepted both. Claude Code proposed splitting the score into two views when they disagreed; I set the rule that a correct record passes the gate and a clean transcript is reported separately. I ran the hosted model comparison myself with my own key, in my own shell. gpt-6-luna at low reasoning effort scored 45 of 45 with no guard trips, gpt-4.1-mini 41, and the local gpt-oss:20b 43 after I relaxed two checks to the brief's wording; I chose Luna as the hosted default and kept gpt-oss as the local development default. For the public release I moved both local TTS engines behind optional extras after a dependency check showed each carries GPL code, so the default install stays MIT-clean.

## Run it locally
You need [git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/). The first run installs the dependencies (about 1.6 GB on Windows, which includes the GPU Whisper wheels). It needs no key, no speech model and no LLM: the scripted LLM is a deterministic rules engine, not a model.

```bash
git clone https://github.com/vybetechlabs-blip/voice-receptionist.git
cd voice-receptionist
uv run python -m agent.text_runner --llm scripted
```

Type caller lines, for example "Hi, I need to schedule a furnace tune-up.", then "Wednesday at 2 pm works.", a name, and a number such as "555-0134"; `/quit` hangs up. The scripted rules recognize booking, hours, FAQ topics, messages and transfer requests, and answer anything else by asking what you need. Other commands, all keyless:

```bash
uv run pytest                     # 235 tests
uv run python -m eval.rescore eval/results/2026-09-25-gpt-6-luna-luna.json   # re-score a saved report, no model
uv run python -m eval.latency_summary                                       # the latency block above
uv run ruff check .
```

Three tests skip without downloaded speech models.
They build a local speech service, which loads a model into memory, so they stay off even when the models are present. After `agent.fetch_models`, set `VOICE_RECEPTIONIST_LOAD_MODELS=1` to run them.

Tested on Windows 11 with Python 3.12 only. The dependency pins and lockfile are cross-platform, but nothing here has been run on Linux or macOS.

### Offline demo
`uv run python -m agent.demo` runs the test suite and prints its summary line, then replays recorded tool calls through the guard layer. Each call goes through `call_tool`, the same path the live pipeline uses, and is checked against the caller's own lines. Nothing loads audio, speech models or an LLM, and the tools write to a scratch copy of the seed CRM, so the demo needs no key and no network and leaves `data/` alone.

The booking replay has the model trim the caller's name and then invent a callback number. The guard refuses both calls before the tool runs, and the third call, built from the caller's own words, books the visit. The transfer replay has the caller try to override the instructions, and the transfer is refused. `--no-tests` skips the suite and prints only the replay.

### Voice server (needs downloads and one speech engine)
The browser test page needs speech-to-text, a voice, and an LLM. Install one local voice (the licenses are under Third-party licenses), download the models, and start the server:

```bash
uv sync --extra kokoro        # or: uv sync --extra piper
uv run python -m agent.fetch_models
uv run python -m agent.voice_server   # open http://127.0.0.1:7860 and press Call
```

`uv run` keeps an installed extra, and a later plain `uv sync` removes it. `fetch_models` downloads what the config names: Whisper `base.en` (148 MB) and the Kokoro model (354 MB), or the Piper voice (63 MB) when the config says `tts: piper`. With `llm: auto` the server uses Ollama if it answers with `gpt-oss:20b` pulled (13.8 GB), and otherwise the scripted LLM. Whisper runs on CUDA by default and falls back to the CPU. Each call's transcript, tool calls and per-turn latency are written to `logs/calls/` (gitignored). The latency probe, `uv run python -m eval.voice_probe --script latency`, needs `--extra kokoro` as well because it synthesizes the caller's lines with Kokoro.

### Hosted models (your own key, costs money)
```bash
uv run python -m eval.rerun --llm openai --model gpt-6-luna --reasoning-effort low --repeat 3 --label luna
uv run python -m agent.voice_server --llm openai --model gpt-6-luna --reasoning-effort low
```
Both read `OPENAI_API_KEY` from your shell and never log it. The local equivalent is `uv run python -m eval.rerun --llm ollama --model gpt-oss:20b --repeat 3 --label <name>`.

## Third-party licenses
The code in this repository is MIT (`LICENSE`). Dependencies keep their own licenses, and nothing in this repository vendors them.

- **The default install contains no GPL package.** `eval/test_dependency_licenses.py` checks every installed distribution's declared license. It contains two weak-copyleft libraries that `pipecat-ai` pulls in, `num2words` (LGPL) and `soxr` (LGPL-2.1-or-later), and proprietary NVIDIA wheels (`nvidia-cublas-cu12` and its dependency `nvidia-cuda-nvrtc-cu12`, installed on Windows only, for GPU Whisper). Pipecat is BSD-2-Clause.
- **`uv sync --extra kokoro`** adds `kokoro-onnx` (MIT), which requires `phonemizer` (GPL-3.0-or-later) and `espeakng-loader`, a package that bundles espeak-ng (GPL-3.0-or-later).
- **`uv sync --extra piper`** adds `piper-tts` 1.8.0 (GPL-3.0-or-later), which bundles espeak-ng.
- **Model files** are downloaded by `agent.fetch_models` and have their own terms. The Piper voice `en_US-lessac-medium` has a model card that names no license and points to the Blizzard 2013 dataset terms. Read them before using that voice beyond local testing.

License: MIT, see [LICENSE](LICENSE). Steadfast Heating & Air is fictional, and any resemblance to a real business or person is coincidental.

## Status
**Status: Active, Phase 1 complete, public demo pending.** Phase 0 (text pipeline, mocked tools, eval) and Phase 1 (local browser voice, hosted provider classes, the guards, the three-model comparison) are done. The public browser demo (`web/` is a placeholder) and a phone number are not started. CI runs ruff and the keyless test suite on every push.
