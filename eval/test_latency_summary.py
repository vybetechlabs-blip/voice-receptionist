"""The committed latency probes, their summary script, and the README block built from it.

The README states latency only as eval.latency_summary prints it. These tests recompute the
summary from the committed probe files and fail when the README block drifts from it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from eval import latency_summary

pytestmark = pytest.mark.unit
README = Path(__file__).parent.parent / "README.md"
BEGIN, END = "<!-- latency:begin -->", "<!-- latency:end -->"


def test_rows_recompute_from_the_committed_probe_files():
    cpu, gpu, prompt, last = latency_summary.summarize()
    assert (cpu.calls, cpu.faq_turns, cpu.faq_median) == (1, 3, pytest.approx(2.157))
    assert (gpu.calls, gpu.faq_turns, gpu.faq_median) == (3, 9, pytest.approx(1.758))
    assert (prompt.calls, prompt.faq_turns, prompt.faq_median) == (5, 14, pytest.approx(1.4305))
    assert (last.calls, last.faq_turns) == (8, 24)
    assert last.faq_median == pytest.approx(1.4625)
    assert (last.faq_min, last.faq_max) == (pytest.approx(0.872), pytest.approx(2.889))


def test_every_row_names_probe_files_that_exist():
    for row in (*latency_summary.PIPER_ROWS, *latency_summary.KOKORO_FIRST_STACK):
        for name in row.files:
            assert (latency_summary.LATENCY_DIR / name).is_file(), name
    for name in latency_summary.BARGE_IN_FILES:
        assert (latency_summary.LATENCY_DIR / name).is_file(), name


def test_probe_files_carry_no_local_paths_or_server_records():
    for path in sorted(latency_summary.LATENCY_DIR.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        json.loads(text)
        assert "server_record" not in text, path.name
        # A drive letter followed by a slash, a user-profile folder, or the workspace folder name.
        assert not re.search(r"[A-Za-z]:[\\/]|[\\/]Users[\\/]|AI_WOR[K]", text), path.name


def test_readme_latency_block_is_the_script_output():
    readme = README.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert BEGIN in readme and END in readme, "README is missing the latency block markers"
    block = readme.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    assert block == latency_summary.render().strip()
