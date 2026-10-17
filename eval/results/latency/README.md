# Latency probes

One JSON file per scripted call made with `eval/voice_probe.py` against a running voice server
on 2026-09-25: the time from the end of the caller's speech to the first agent audio, per caller
turn. `uv run python -m eval.latency_summary` turns them into the table in the main README, and
`eval/test_latency_summary.py` checks that the README shows nothing else.

Machine: Intel i7-8700K, 32 GB RAM, NVIDIA RTX 4080 (also recorded in each file's `hardware` block).

These are the original probe files with one change: the `server_record` field, a local file path
to the server's own call log, was removed. The call logs themselves are not published.
