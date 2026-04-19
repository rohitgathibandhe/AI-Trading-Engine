from __future__ import annotations

import json
from datetime import datetime

import market_ai.intraday_defined_risk.collector as collector_module
from market_ai.intraday_defined_risk.collector import (
    _default_dhan_creds_candidates,
    _flatten_option_chain_payload,
    _load_dhan_creds,
    run_daily_chain_schedule,
)


def test_flatten_option_chain_payload_extracts_bid_ask_and_greeks() -> None:
    payload = {
        "data": {
            "last_price": 22456.2,
            "oc": {
                "22500.000000": {
                    "ce": {
                        "last_price": 45.5,
                        "oi": 1200,
                        "implied_volatility": 11.8,
                        "top_bid_price": 45.1,
                        "top_ask_price": 45.9,
                        "greeks": {"delta": 0.22},
                    },
                    "pe": {
                        "last_price": 70.4,
                        "oi": 1400,
                        "implied_volatility": 12.3,
                        "top_bid_price": 70.0,
                        "top_ask_price": 70.8,
                        "greeks": {"delta": -0.18},
                    },
                }
            },
        }
    }

    rows = _flatten_option_chain_payload(
        payload,
        timestamp=datetime(2026, 4, 4, 9, 30, 0),
        decision_time="09:30",
        expiry="2026-04-09",
    )

    assert len(rows) == 2
    ce = next(row for row in rows if row["option_type"] == "CALL")
    pe = next(row for row in rows if row["option_type"] == "PUT")
    assert ce["bid"] == 45.1
    assert ce["ask"] == 45.9
    assert ce["delta"] == 0.22
    assert ce["spot"] == 22456.2
    assert pe["bid"] == 70.0
    assert pe["ask"] == 70.8
    assert pe["delta"] == -0.18
    assert pe["delta_source"] == "PROVIDED"


def test_run_daily_chain_schedule_captures_targets_and_stops(monkeypatch, tmp_path) -> None:
    captured: list[tuple[str, str]] = []

    class DummyReport:
        def __init__(self, decision_time: str, stamp: datetime) -> None:
            self.decision_time = decision_time
            self.timestamp = stamp.isoformat()

        def to_dict(self) -> dict[str, str]:
            return {"decision_time": self.decision_time, "timestamp": self.timestamp}

    def fake_capture(*args, **kwargs):
        decision_time = kwargs["decision_time"]
        stamp = kwargs["timestamp"]
        captured.append((decision_time, stamp.strftime("%Y-%m-%d")))
        return DummyReport(decision_time, stamp)

    times = iter(
        [
            datetime(2026, 4, 4, 9, 30),   # Saturday, should be ignored
            datetime(2026, 4, 6, 9, 20),   # Monday pre first capture
            datetime(2026, 4, 6, 9, 30),
            datetime(2026, 4, 6, 10, 0),
            datetime(2026, 4, 6, 13, 0),
            datetime(2026, 4, 6, 15, 20),  # stop after one completed day
        ]
    )

    monkeypatch.setattr("market_ai.intraday_defined_risk.collector.capture_decision_time_snapshot", fake_capture)
    monkeypatch.setattr("market_ai.intraday_defined_risk.collector.time.sleep", lambda *_args, **_kwargs: None)

    reports = run_daily_chain_schedule(
        tmp_path,
        now_provider=lambda: next(times),
        poll_seconds=0.01,
        max_days=1,
    )

    assert [report.decision_time for report in reports] == ["09:30", "10:00", "13:00"]
    assert captured == [("09:30", "2026-04-06"), ("10:00", "2026-04-06"), ("13:00", "2026-04-06")]


def test_default_dhan_creds_candidates_prefer_ui_saved_creds_path() -> None:
    candidates = _default_dhan_creds_candidates()

    assert candidates[0].as_posix().endswith("/data_engine/market_ai/state/creds.json")


def test_load_dhan_creds_reads_ui_saved_creds_candidate(monkeypatch, tmp_path) -> None:
    creds = tmp_path / "market_ai" / "state" / "creds.json"
    creds.parent.mkdir(parents=True)
    creds.write_text(json.dumps({"client_id": "1000678336", "access_token": "token-from-ui"}))
    monkeypatch.delenv("DHAN_CLIENT_ID", raising=False)
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(collector_module, "_default_dhan_creds_candidates", lambda: [creds])

    client_id, token = _load_dhan_creds()

    assert client_id == "1000678336"
    assert token == "token-from-ui"
