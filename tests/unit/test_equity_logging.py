import csv
import json
from pathlib import Path

import pytest

from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import (
    LiveConfig,
    _record_equity_snapshot,
    _record_activity_event,
)

pytestmark = pytest.mark.unit


class _StubDhan:
    def __init__(self, funds: dict) -> None:
        self._funds = funds

    def get_funds(self) -> dict:
        return dict(self._funds)


def test_record_equity_snapshot_accumulates_exposures(tmp_path: Path) -> None:
    cfg = LiveConfig()
    cfg.equity_log_path = str(tmp_path / "equity.csv")

    funds = {
        "available": 100000,
        "collateral": 25000,
        "utilized": 5000,
        "withdrawable": 95000,
    }
    positions = [
        {"qty": 75, "avg_price": 120.0, "pnl": 500.0, "_raw": {"realizedProfit": 150.0}},
        {"qty": -75, "avg_price": 100.0, "pnl": -300.0, "_raw": {"realizedProfit": 0.0}},
    ]

    _record_equity_snapshot(cfg, _StubDhan(funds), positions)

    with open(cfg.equity_log_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    row = rows[0]
    assert float(row["available"]) == pytest.approx(100000)
    assert float(row["equity_estimate"]) == pytest.approx(130000)
    gross = float(row["gross_exposure"])
    expected_gross = 75 * 120.0 + 75 * 100.0
    assert gross == pytest.approx(expected_gross)
    assert float(row["unrealized"]) == pytest.approx(200.0)
    assert float(row["realized"]) == pytest.approx(150.0)


def test_record_activity_event_writes_feature_row(tmp_path: Path) -> None:
    cfg = LiveConfig()
    cfg.feature_log_path = str(tmp_path / "feature.csv")

    _record_activity_event(
        cfg,
        "test_alert",
        "Something happened",
        severity="warning",
        context={"foo": "bar"},
    )

    with open(cfg.feature_log_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    row = rows[0]
    assert row["strategy"] == "risk_event"
    ctx = json.loads(row["context"])
    assert ctx["label"] == "test_alert"
    assert ctx["message"] == "Something happened"
    assert ctx["severity"] == "warning"
