"""The offline demo: each recorded call gets the verdict the demo claims, and data/ is never touched."""

from __future__ import annotations

import shutil
from pathlib import Path

from agent import demo
from tools import ToolEnv
from tools.store import SEED_DIR


def _env(biz, tmp_path: Path) -> ToolEnv:
    shutil.copytree(SEED_DIR, tmp_path, dirs_exist_ok=True)
    return ToolEnv(tmp_path, biz, clock=lambda: demo.FROZEN_NOW)


def test_booking_replay_rejects_twice_then_books(biz, tmp_path):
    verdicts = demo.replay(demo.REPLAYS[0], _env(biz, tmp_path))
    assert [v.split(" (")[0].split(",")[0] for v in verdicts] == [
        "REJECTED by the guard",
        "REJECTED by the guard",
        "ALLOWED",
    ]
    assert verdicts[-1].endswith("booked")


def test_transfer_replay_is_refused(biz, tmp_path):
    assert demo.replay(demo.REPLAYS[1], _env(biz, tmp_path)) == [
        "REJECTED by the guard (refused)"
    ]


def test_main_without_tests_leaves_data_alone(capsys):
    data = demo.ROOT / "data"
    before = {p.name: p.read_bytes() for p in data.glob("*") if p.is_file()}
    assert demo.main(["--no-tests"]) == 0
    after = {p.name: p.read_bytes() for p in data.glob("*") if p.is_file()}
    assert before == after
    out = capsys.readouterr().out
    assert "1. Test suite" not in out
    assert out.count("REJECTED") == 3
