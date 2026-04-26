from __future__ import annotations

from datetime import datetime

from market_ai.dhan_wrapper import DhanWrapper


def test_dhan_wrapper_parses_epoch_timestamp_as_ist_naive() -> None:
    assert DhanWrapper._parse_timestamp(1776743100.0) == datetime(2026, 4, 21, 9, 15)


def test_dhan_wrapper_parses_epoch_timestamp_string_as_ist_naive() -> None:
    assert DhanWrapper._parse_timestamp("1776743400.0") == datetime(2026, 4, 21, 9, 20)
