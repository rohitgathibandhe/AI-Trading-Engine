import csv
from pathlib import Path

import pytest

from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import (
    LiveConfig,
    _apply_exposure_throttles,
)

pytestmark = pytest.mark.unit


def _read_feature_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def test_delta_breach_blocks_and_resets(tmp_path: Path) -> None:
    feature_file = tmp_path / "feature.csv"
    cfg = LiveConfig(feature_log_path=str(feature_file))
    cfg.risk_max_portfolio_delta = 0.1
    cfg.risk_account_equity = 100_000.0
    cfg.risk_max_exposure_pct = 0.05

    _apply_exposure_throttles(cfg, net_delta=0.2, total_notional=1000.0)
    assert cfg._delta_breach_active is True
    assert cfg.enable_strangle_entry is False

    _apply_exposure_throttles(cfg, net_delta=0.05, total_notional=1000.0)
    assert cfg._delta_breach_active is False
    assert cfg.enable_strangle_entry is True

    rows = _read_feature_rows(feature_file)
    labels = [row["strategy"] for row in rows]
    assert "risk_event" in labels
