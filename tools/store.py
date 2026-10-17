"""The fake CRM on disk: data/calendar.json, data/messages.json, data/transfers.log.

Seed copies live in data/seed/ and are version-controlled; the working copies in data/ are created
from them on first use and are gitignored, so running the agent never dirties the repo.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

CALENDAR = "calendar.json"
MESSAGES = "messages.json"
TRANSFERS = "transfers.log"
SEEDED = (CALENDAR, MESSAGES)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
SEED_DIR = DEFAULT_DATA_DIR / "seed"


def ensure_data(data_dir: Path, *, reset: bool = False) -> Path:
    """Create any missing working file from its seed. ``reset=True`` restores all of them."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in SEEDED:
        target = data_dir / name
        if reset or not target.exists():
            shutil.copyfile(SEED_DIR / name, target)
    log = data_dir / TRANSFERS
    if reset or not log.exists():
        log.write_text("", encoding="utf-8")
    return data_dir


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    """Write atomically so a crash mid-write never leaves a half-written CRM file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def next_id(prefix: str, existing: list[dict]) -> str:
    return f"{prefix}-{len(existing) + 1:04d}"
