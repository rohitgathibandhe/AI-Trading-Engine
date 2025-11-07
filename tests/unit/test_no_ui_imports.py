from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


FORBIDDEN_SUBSTRINGS = ("market_ai.ui", "streamlit")


def test_test_suite_avoids_ui_imports():
    tests_dir = Path(__file__).resolve().parents[1]
    offenders = []
    for path in tests_dir.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        content = path.read_text()
        for needle in FORBIDDEN_SUBSTRINGS:
            if needle in content:
                offenders.append((path, needle))
    assert not offenders, f"UI dependencies referenced in tests: {offenders}"
