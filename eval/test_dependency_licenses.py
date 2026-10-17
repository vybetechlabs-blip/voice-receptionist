"""The default install carries no GPL package.

The optional local voices (`--extra kokoro`, `--extra piper`) pull GPL-3 code, so this check only
applies to an environment where neither extra is installed, which is what `uv sync` gives and
what CI runs. Weak copyleft (LGPL) is allowed and listed in the README.
"""

from __future__ import annotations

import importlib.util
from importlib import metadata

import pytest

pytestmark = pytest.mark.unit
EXTRA_MODULES = ("kokoro_onnx", "piper", "phonemizer")


def _license_text(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    parts = [meta.get("License-Expression") or "", meta.get("License") or ""]
    parts += [c for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
    return " ".join(parts)


def _is_gpl(text: str) -> bool:
    upper = text.upper()
    if "LGPL" in upper or "LESSER" in upper:
        return False
    return "GPL" in upper or "GENERAL PUBLIC LICENSE" in upper


def test_license_check_recognizes_gpl_and_not_lgpl():
    assert _is_gpl("GPL-3.0-or-later")
    assert _is_gpl("License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)")
    assert not _is_gpl("LGPL-2.1-or-later")
    assert not _is_gpl("GNU Library or Lesser General Public License (LGPL)")
    assert not _is_gpl("MIT License")


def test_default_install_has_no_gpl_package():
    installed = [m for m in EXTRA_MODULES if importlib.util.find_spec(m)]
    if installed:
        pytest.skip(f"an optional voice extra is installed ({', '.join(installed)}); it pulls GPL code by design")
    gpl = sorted(
        f"{d.metadata['Name']} ({_license_text(d).strip()[:60]})" for d in metadata.distributions() if _is_gpl(_license_text(d))
    )
    assert gpl == []
