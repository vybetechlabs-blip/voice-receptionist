# Cost model

Rules: every price or free-tier figure comes from the vendor's own page, with the
URL and the date read. Anything that couldn't be confirmed there is marked **unverified**. Nothing
is estimated. The per-minute rows below are arithmetic on those sourced prices and on this repo's
own measurements, with every assumption labeled. Compliance statements appear only as quoted from
a vendor page.

All vendor pages were read on **2026-09-25**. Quotes are verbatim from the pages. Two reads:
- **Research pass:** a read of all four vendors' pages. It used a browser for OpenAI, whose pages
  refuse plain fetches.
- **Spot-check:** a second, manual re-fetch. It confirmed the Deepgram figures and the Cartesia
  plan figures exactly. It could not see Cartesia's FAQ, so the two Cartesia FAQ figures are
  flagged below.

## Assumptions and measured inputs

| Input | Value | Basis |
|---|---|---|
| Caller turns per call-minute | 4 | **Assumption.** The scripted probe caller answers the instant the agent stops, so it runs faster than a real caller, who pauses and talks longer. LLM and TTS cost scale linearly with this number; STT does not |
| LLM requests per caller turn | 1.84 | **Measured.** `eval/results/2026-09-25-gpt-oss-20b-structural.json`: 45 text runs with the current agent (FAQ in the prompt, `resolve_date`, guards). Earlier runs: 1.61 (baseline), 1.78 (post-fix) |
| Prompt tokens per LLM request | 1,656.6 (mean) | **Measured**, same runs. System prompt (now with the FAQ inline) + tools + conversation so far. Earlier: 1,079.5 (baseline), 1,251.2 (post-fix) |
| Completion tokens per LLM request | 69.5 (mean) | **Measured**, same runs. The `gpt-oss` tokenizer counts are used as a stand-in for OpenAI's; a different model's counts will differ |
| Agent characters spoken per caller turn | 85.8 | **Measured**, same runs (baseline 94.3, post-fix 90.5) |
| Greeting | 148 characters, once per call | **Measured** from the config |
| STT audio billed per call-minute | 1 minute | **Assumption.** The caller's audio streams for the whole call |

Derived per call-minute: 7.36 LLM requests, about 12,193 input and 512 output tokens, and about
343 characters of agent speech (plus the greeting once).

## (a) Fully local, $0 cash cost

| Layer | Provider (as run) | Cash cost per minute | Notes |
|---|---|---|---|
| STT | faster-whisper `base.en`, GPU float16 (default; CPU int8 fallback) | $0 | 148 MB model, plus the cuBLAS wheel (553 MB download) |
| LLM | Ollama `gpt-oss:20b`, GPU | $0 | 13.8 GB, pulled locally |
| TTS | Kokoro v1.0 (ONNX, CPU; `kokoro` extra, the config default) or Piper `en_US-lessac-medium` (`piper` extra) | $0 | 354 MB / 63 MB |
| Turn detection | Smart Turn v3.2 + Silero VAD (bundled with `pipecat-ai` 1.11.0) | $0 | |

The local voices are for development and testing. A public demo would use a hosted TTS, so the TTS
rows in (b) and (c) apply to it, not the $0 row above.

Electricity and hardware amortization are not estimated (**unverified**). Measured voice latency,
with the machine it ran on, is in the README's latency section.

## (b) Hosted free tiers and signup credit

| Layer | Vendor | Free tier or credit (verbatim) | What it covers at the measured usage | Source |
|---|---|---|---|---|
| STT | Deepgram | "Free $200 Credit … then pay-as-you-go"; "this credit is available until you use it up" | 41,667 streaming minutes at the promotional $0.0048/min; 25,974 at the regular $0.0077/min | https://deepgram.com/pricing (research pass; credit line spot-checked) |
| TTS | Cartesia | Free: "$0 /mo … 20K credits / month". Credits: "1 credit equals 1 character" | About 58 call-minutes a month. The Free plan does not list the "Commercial use license" that Pro adds | https://cartesia.ai/pricing (plan figures spot-checked) |
| TTS | ElevenLabs | API page: Flash/Turbo characters included on Free, "20,000". General page: "Free $0 (10,000 credits)" | About 58 call-minutes at 20,000 characters; 29–58 at 10,000 credits and "0.5 and 1 credit per character". **The two vendor pages disagree.** The UI plan table marks Free without commercial use | https://elevenlabs.io/pricing/api, https://elevenlabs.io/pricing (research pass) |
| STT, LLM, TTS | OpenAI | No free tier or signup credit stated on the pricing pages | n/a | https://developers.openai.com/api/docs/pricing, https://openai.com/api/pricing/ (research pass) |

A public demo would be commercial use. Free tiers that don't list a commercial license may not fit
(an observation, not legal advice).

## (c) A plausible paid production stack

Unit prices (verbatim, read 2026-09-25):

| Layer | Vendor / model | Unit price (verbatim) | Cost per call-minute at the measured usage | Source |
|---|---|---|---|---|
| STT | Deepgram Nova-3 Monolingual, streaming | "Current price $0.0048/min Regular price $0.0077/min" ("Limited-time promotional rates on streaming", **end date unverified**) | $0.0048 (promo) / $0.0077 (regular) | https://deepgram.com/pricing (spot-checked) |
| STT | OpenAI `gpt-transcribe` | "$0.0045 / minute" | $0.0045 | OpenAI pricing (research pass) |
| STT | OpenAI `gpt-4o-mini-transcribe` | "$1.25 / $5.00 / $0.003 / minute" (the per-minute figure is the page's own estimate) | about $0.0030 | OpenAI pricing (research pass) |
| LLM | OpenAI `gpt-4o-mini` | "$0.15 / $0.075 / $0.60" per 1M tokens (input / cached / output) | $0.0021 | OpenAI pricing (research pass) |
| LLM | OpenAI `gpt-4.1-mini` | "$0.40 / $0.10 / $1.60" | $0.0057 on the `gpt-oss` stand-in; **$0.0027–$0.0034 on its own measured tokens** (see "Hosted LLM, measured" below) | OpenAI pricing (research pass); same rates on https://developers.openai.com/api/docs/models/gpt-4.1-mini (read 2026-09-25, re-read 2026-09-26) |
| LLM | OpenAI `gpt-5-mini` | "$0.25 / $0.025 / $2.00" | $0.0041 (it reasons, so real output tokens may run higher; **unverified** here) | OpenAI pricing (research pass) |
| LLM | OpenAI `gpt-6-luna`, reasoning effort low (**the hosted default**) | Input "$0.10", cached input "$0.01", cache writes "$0.125", output "$0.50" per 1M text tokens | **$0.0002–$0.0011 on its own measured tokens**, reasoning included (see "Hosted LLM, measured" below). The earlier $0.0015 used the `gpt-oss` stand-in | https://developers.openai.com/api/docs/models/gpt-6-luna (read 2026-09-25, re-read 2026-09-26) |
| TTS | Cartesia Sonic, Startup plan | "$49 /mo … 1.25M credits / month"; overage "At startup tier, it's $45 per 1M credits"\* | $0.0135 (included credits, fully used) / $0.0154 (overage rate) | https://cartesia.ai/pricing |
| TTS | ElevenLabs Flash / Turbo | "$0.05 Price per 1K characters" | $0.0172 | https://elevenlabs.io/pricing/api (research pass) |
| TTS | OpenAI `tts-1` | "$15.00 / 1M characters" | $0.0051 | OpenAI pricing (research pass) |
| TTS | OpenAI `gpt-4o-mini-tts` | "$0.60" text input and "$12.00" audio output per 1M tokens | **unverified** per minute (the page gives token prices only) | OpenAI pricing (research pass) |
| TTS | Deepgram Aura-2 | "$0.030/1k characters" | $0.0103 | https://deepgram.com/pricing (research pass) |

**Hosted eval runs:** the amount billed for the `gpt-6-luna` and `gpt-4.1-mini` eval runs is not recorded here. Costs below are arithmetic on published rates and on the token counts each report recorded. See "Hosted LLM, measured" below.

\* Cartesia's FAQ figures ("One minute of audio generation requires 750-800 credits" and the
overage rates) were read by the research pass. The spot-check could not see the FAQ, so
**re-confirm before relying on them**.

**Worked example:** Deepgram Nova-3 streaming + `gpt-4.1-mini` + Cartesia Startup comes to
**about $0.024 per call-minute** at the promotional STT rate, or **$0.027** at the regular rate.
That's roughly $0.072–$0.081 for a 3-minute call. (Figures use the current agent's measured usage.
The inline FAQ makes each request larger but removes tool round trips. Earlier runs came to
$0.023 / $0.026.)

> **Scope: STT + LLM + TTS only. This is not an all-in price; do not quote it as one.**
> It **excludes**:
> - telephony (phone numbers and carrier minutes),
> - orchestration hosting (the server or platform running the pipeline),
> - noise cancellation,
> - platform fees,
> - the Cartesia subscription floor when usage is below the included credits.
>
> It also takes no cached-input discount.

**Compliance, as quoted by the vendors** (not verified further):
- **Deepgram:** "We sign Business Associate Agreements (BAA) for Enterprise customers handling
  electronic Protected Health Information (ePHI)."
- **Cartesia:** the Enterprise tier lists "DPAs and BAAs for compliance".
- **ElevenLabs:** Enterprise lists "BAAs for HIPAA customers".

## Hosted LLM, measured (eval runs of 2026-09-25)

Rates per 1M text tokens, standard processing, short context, read **2026-09-26**:
- **`gpt-6-luna`:** input $0.10, cached input $0.01, cache writes $0.125, output $0.50. The model
  page adds: "Cached input tokens are priced at 10% of the uncached input token rate. Cache writes
  are billed at 1.25x the uncached input token rate." Sources:
  https://developers.openai.com/api/docs/models/gpt-6-luna and
  https://developers.openai.com/api/docs/pricing
- **`gpt-4.1-mini`:** input $0.40, cached input $0.10, output $1.60, no cache-write charge.
  Source: https://developers.openai.com/api/docs/models/gpt-4.1-mini
- **Accounting:** ordinary input = input − cached − cache writes, per the code on
  https://developers.openai.com/api/docs/guides/prompt-caching. Output includes reasoning tokens.

Tokens are as each report recorded them from the provider's usage fields: 45 text-mode runs per
model, 15 scenarios × 3. Scripted callers.

| Model | LLM requests (per caller turn) | Input tokens: total = ordinary + cached + cache writes | Output (reasoning) | Cost of the 45 runs at published rates (arithmetic, not a billed amount) |
|---|---|---|---|---|
| `gpt-6-luna`, effort low | 108 (1.333) | 164,578 = 324 + 157,869 (95.9%) + 6,385 (3.9%) | 3,441 (477) | $0.0041 |
| `gpt-4.1-mini` | 110 (1.325) | 165,605 = 117,093 + 48,512 (29.3%) + 0 | 2,760 (0) | $0.0561 |
| `gpt-oss:20b` (local, for reference) | 160 (1.975) | 270,121 (Ollama's own counts) | 12,159 | $0 |

**Projection: LLM cost per call-minute.** This is a projection, not a measurement. Assumptions:
- **Caller turns per minute: 4.** An assumption, the same as the table at the top.
- **Requests per turn, and input and output tokens per request:** as measured above. Luna: 1.333
  requests, 1,523.9 input and 31.9 output tokens. 4.1-mini: 1.325 requests, 1,505.5 input and 25.1
  output. That comes to about 8,100 input and 130–170 output tokens a minute.
- **The low end uses the eval's measured cache mix.** 45 calls ran back to back with one prompt.
  The prompt changes only by date, so live calls share the prefix. Caching is on by default for
  GPT-5.6 and later, and the hosted default now sets it explicitly (`llm_extra` in the business
  config): `prompt_cache_options` mode implicit, ttl `30m`, plus a `prompt_cache_key`.
- **The high end bills every input token at the worst rate:** Luna as cache writes (1.25×), and
  4.1-mini as ordinary input. Sparse demo traffic, with gaps over 30 minutes, sits between the two.

| Model | Per call-minute (low – high) | Per 3-minute demo call (low – high) |
|---|---|---|
| `gpt-6-luna`, effort low | $0.00020 – $0.0011 | $0.0006 – $0.0033 |
| `gpt-4.1-mini` | $0.0027 – $0.0034 | $0.0081 – $0.0102 |

> **Scope: LLM only. This is not an all-in price; do not quote it as one.** STT, TTS, hosting,
> transport, and platform fees are excluded. See the tables above.

Voice latency for the hosted stacks has not been measured.
