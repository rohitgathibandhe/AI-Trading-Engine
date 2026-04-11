from __future__ import annotations

import csv
import json
from pathlib import Path

import scripts.paper_pnl_server as paper_server
from scripts.paper_pnl_server import load_positions


def _candle(ts: str, open_px: float, high_px: float, low_px: float, close_px: float, volume: float = 1000.0) -> dict:
    return {
        "timestamp": ts,
        "open": open_px,
        "high": high_px,
        "low": low_px,
        "close": close_px,
        "volume": volume,
    }


def test_load_positions_nets_closed_paper_trade_and_realizes_pnl(tmp_path: Path) -> None:
    blotter = tmp_path / "trade_blotter.csv"
    with blotter.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "trade_mode",
                "warn_only",
                "executed",
                "side",
                "order_type",
                "exchange_seg",
                "product_type",
                "security_id",
                "quantity",
                "price",
                "delta",
                "expiry",
                "strike",
                "tag",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-03-06T10:00:00+05:30",
                "trade_mode": "paper",
                "warn_only": 0,
                "executed": 0,
                "side": "SELL",
                "order_type": "MARKET",
                "exchange_seg": "NSE_FNO",
                "product_type": "MARGIN",
                "security_id": "",
                "quantity": 65,
                "price": 100.0,
                "delta": "",
                "expiry": "2026-03-10",
                "strike": 25500,
                "tag": "BEAR_CALL_SPREAD",
                "notes": "OPEN",
            }
        )
        writer.writerow(
            {
                "timestamp": "2026-03-06T10:22:00+05:30",
                "trade_mode": "paper",
                "warn_only": 0,
                "executed": 0,
                "side": "BUY",
                "order_type": "MARKET",
                "exchange_seg": "NSE_FNO",
                "product_type": "MARGIN",
                "security_id": "",
                "quantity": 65,
                "price": 80.0,
                "delta": "",
                "expiry": "2026-03-10",
                "strike": 25500,
                "tag": "BEAR_CALL_SPREAD",
                "notes": "CLOSE",
            }
        )

    out = load_positions(blotter, mode="paper")

    assert out["positions"] == []
    assert len(out["closed"]) == 1
    assert round(float(out["realized_pnl"]), 2) == 1300.0


def test_build_intraday_performance_payload_summarizes_history(tmp_path: Path, monkeypatch) -> None:
    hist = tmp_path / "intraday_ai_trade_history.jsonl"
    rows = [
        {
            "position_id": "P1",
            "trade_mode": "paper",
            "strategy_label": "Bear Call Credit Spread",
            "closed_at": "2026-03-06T10:30:00+05:30",
            "pnl_rs": 1200.0,
            "hold_minutes": 28.0,
            "reason": "TARGET_HIT",
            "result": "WIN",
            "expiry": "2026-03-10",
        },
        {
            "position_id": "P2",
            "trade_mode": "paper",
            "strategy_label": "Bull Put Credit Spread",
            "closed_at": "2026-03-06T12:15:00+05:30",
            "pnl_rs": -800.0,
            "hold_minutes": 35.0,
            "reason": "SL_HIT",
            "result": "LOSS",
            "expiry": "2026-03-10",
        },
        {
            "position_id": "L1",
            "trade_mode": "live",
            "strategy_label": "Ignore Live",
            "closed_at": "2026-03-06T13:00:00+05:30",
            "pnl_rs": 999.0,
            "hold_minutes": 20.0,
            "reason": "TARGET_HIT",
            "result": "WIN",
            "expiry": "2026-03-10",
        },
    ]
    hist.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(paper_server, "INTRADAY_AI_TRADE_HISTORY_JSONL", hist)

    out = paper_server._build_intraday_performance_payload(trade_mode="paper")

    assert out["total_trades"] == 2
    assert out["wins"] == 1
    assert out["losses"] == 1
    assert out["flats"] == 0
    assert out["win_rate_pct"] == 50.0
    assert out["avg_hold_minutes"] == 31.5
    assert out["realized_pnl_rs"] == 400.0
    assert out["exit_reasons"] == [
        {"reason": "SL_HIT", "count": 1},
        {"reason": "TARGET_HIT", "count": 1},
    ] or out["exit_reasons"] == [
        {"reason": "TARGET_HIT", "count": 1},
        {"reason": "SL_HIT", "count": 1},
    ]
    assert len(out["recent_trades"]) == 2


def test_load_intraday_learning_status_returns_payload(tmp_path: Path, monkeypatch) -> None:
    status = tmp_path / "intraday_ai_learning_status.json"
    status.write_text(
        json.dumps(
            {
                "status": "ACTIVE",
                "total_outcomes": 5,
                "wins": 3,
                "losses": 2,
                "flats": 0,
                "realized_pnl_rs": 145.5,
                "last_outcome_at": "2026-04-03T11:10:00+05:30",
                "last_assessment": {
                    "strategy_type": "CALL_CREDIT_SPREAD",
                    "confidence_adjust": 0.06,
                    "block_entry": False,
                    "negative_hits": 0,
                    "positive_hits": 2,
                    "reasons": ["LEARNING_POSITIVE_STRATEGY_TYPE_CALL_CREDIT_SPREAD"],
                },
                "top_negative_buckets": [],
                "top_positive_buckets": [{"bucket": "strategy_type", "label": "CALL_CREDIT_SPREAD", "count": 3, "avg_pnl_rs": 48.5, "win_rate": 0.6667}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_server, "INTRADAY_AI_LEARNING_STATUS_JSON", status)

    out = paper_server._load_intraday_learning_status()

    assert out["status"] == "ACTIVE"
    assert out["total_outcomes"] == 5
    assert out["last_assessment"]["positive_hits"] == 2
    assert out["top_positive_buckets"][0]["label"] == "CALL_CREDIT_SPREAD"


def test_fetch_ltp_lookup_includes_dhan_client_id_and_maps_401(tmp_path: Path, monkeypatch) -> None:
    creds = tmp_path / "creds.json"
    creds.write_text(json.dumps({"client_id": "CID123", "access_token": "TOK123"}), encoding="utf-8")
    monkeypatch.setattr(paper_server, "CREDS_FILE", creds)
    monkeypatch.setattr(paper_server, "_LAST_LTP_FETCH_TS", 0.0)
    monkeypatch.setattr(paper_server, "_NEXT_LTP_ALLOWED_TS", 0.0)

    captured: dict = {}

    class _Resp:
        status_code = 401

        def json(self):
            return {"errorMessage": "Unauthorized"}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(paper_server.requests, "post", _fake_post)

    lookup, status = paper_server._fetch_ltp_lookup({"NSE_FNO": [12345, 12345]})

    assert lookup == {}
    assert status == "auth-error: refresh dhan access token"
    assert captured["headers"]["client-id"] == "CID123"
    assert captured["headers"]["access-token"] == "TOK123"
    assert captured["json"]["dhanClientId"] == "CID123"
    assert captured["json"]["NSE_FNO"] == [12345]


def test_build_batman_bkm_review_payload_exposes_last_closed_basket(tmp_path: Path, monkeypatch) -> None:
    strategy_state = tmp_path / "strategy_state.json"
    strategy_state.write_text(
        json.dumps(
            {
                "BATMAN_BKM": {
                    "2026-05-26": {
                        "status": "CLOSED",
                        "opened_at": "2026-04-01T09:00:17+05:30",
                        "closed_at": "2026-04-01T09:18:58+05:30",
                        "reason": "DAILY_LOCK_RED",
                        "pnl": -15522.0,
                        "net_credit": 24173.5,
                        "credit_pct": 2.41735,
                        "quality_status": "PASS",
                        "quality_score": 100.0,
                        "quality_metrics": {
                            "center_offset": 7.95,
                            "short_distance_min": 992.05,
                            "short_width": 2000.0,
                            "outer_distance_ratio": 1.0,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    events = tmp_path / "batman_bkm_ai_events.jsonl"
    events.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-01T09:18:50+05:30",
                "basket_expiry": "2026-05-26",
                "market_bias": "NEUTRAL",
                "position_risk_side": "CENTER",
                "reasons": ["LOSS_BUILDING", "Pnl_DRAWDOWN_WATCH"],
                "confidence": 0.848,
                "score": 26.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_server, "STRATEGY_STATE", strategy_state)
    monkeypatch.setattr(paper_server, "BATMAN_BKM_AI_EVENTS_JSONL", events)

    out = paper_server._build_batman_bkm_review_payload(trade_mode="paper")

    latest = out["latest_closed"]
    assert out["summary"]["closed_count"] == 1
    assert latest["expiry"] == "2026-05-26"
    assert latest["pnl_rs"] == -15522.0
    assert latest["opened_pre_market"] is True
    assert latest["daily_lock_triggered"] is True
    assert latest["quality_status"] == "PASS"
    assert latest["held_minutes"] == 18.68
    assert "STRUCTURE_SHAPE_WAS_GOOD" in latest["lessons"]
    assert "BLOCK_BATMAN_ENTRY_BEFORE_09_15" in latest["improvements"]
    assert latest["close_context"]["market_bias"] == "NEUTRAL"


def test_build_batman_bkm_learning_payload_exposes_adaptive_state(tmp_path: Path, monkeypatch) -> None:
    status = tmp_path / "batman_bkm_learning_status.json"
    status.write_text(
        json.dumps(
            {
                "status": "ACTIVE",
                "total_outcomes": 4,
                "wins": 1,
                "losses": 3,
                "flats": 0,
                "realized_pnl_rs": -9100.5,
                "last_outcome_at": "2026-04-01T09:18:58+05:30",
                "last_outcome_id": "2026-05-26|1",
                "last_entry_assessment": {
                    "status": "BLOCK",
                    "block_entry": True,
                    "risk_score_adjust": 12.0,
                    "reasons": ["LEARNING_BLOCK_PRE_MARKET_ENTRY"],
                    "supportive_reasons": [],
                    "timestamp": "2026-04-02T09:14:00+05:30",
                },
                "top_negative_buckets": [
                    {"bucket": "entry_phase", "label": "PRE_MARKET", "count": 3, "avg_pnl_rs": -3200.0, "win_rate": 0.0}
                ],
                "top_positive_buckets": [
                    {"bucket": "quality_status", "label": "PASS", "count": 4, "avg_pnl_rs": 1200.0, "win_rate": 0.75}
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_server, "BATMAN_BKM_LEARNING_STATUS_JSON", status)

    out = paper_server._build_batman_bkm_learning_payload(trade_mode="paper")

    assert out["status"] == "ACTIVE"
    assert out["summary"]["total_outcomes"] == 4
    assert out["summary"]["realized_pnl_rs"] == -9100.5
    assert out["last_entry_assessment"]["status"] == "BLOCK"
    assert out["last_entry_assessment"]["block_entry"] is True
    assert out["top_negative_buckets"][0]["label"] == "PRE_MARKET"
    assert out["top_positive_buckets"][0]["label"] == "PASS"


def test_build_intraday_index_chart_payload_prefers_bear_call_after_pullback(monkeypatch) -> None:
    candles_5m = [
        _candle("2026-04-04T09:15:00+05:30", 23060, 23080, 23020, 23040),
        _candle("2026-04-04T09:20:00+05:30", 23040, 23055, 22995, 23005),
        _candle("2026-04-04T09:25:00+05:30", 23005, 23020, 22945, 22960),
        _candle("2026-04-04T09:30:00+05:30", 22960, 22980, 22905, 22920),
        _candle("2026-04-04T09:35:00+05:30", 22920, 22940, 22870, 22895),
        _candle("2026-04-04T09:40:00+05:30", 22895, 22925, 22880, 22910),
        _candle("2026-04-04T09:45:00+05:30", 22910, 22960, 22900, 22945),
        _candle("2026-04-04T09:50:00+05:30", 22945, 22955, 22895, 22900),
        _candle("2026-04-04T09:55:00+05:30", 22900, 22910, 22840, 22855),
    ]
    candles_15m = [
        _candle("2026-04-04T09:15:00+05:30", 23120, 23140, 22980, 23010),
        _candle("2026-04-04T09:30:00+05:30", 23010, 23035, 22870, 22905),
        _candle("2026-04-04T09:45:00+05:30", 22905, 22965, 22835, 22855),
    ]
    candles_60m = [
        _candle("2026-04-04T09:15:00+05:30", 23300, 23340, 22820, 22855),
        _candle("2026-04-04T10:15:00+05:30", 22855, 22910, 22780, 22820),
    ]
    advisor = {
        "pcr_bias": "BEARISH",
        "oi_build_bias": "BEARISH",
        "market_context": {
            "option_chain": {
                "call_wall_above": {"strike": 23050.0},
                "put_wall_below": {"strike": 22650.0},
            }
        },
    }

    out = paper_server._build_intraday_index_chart_payload_from_candles(
        candles_5m=candles_5m,
        candles_15m=candles_15m,
        candles_60m=candles_60m,
        advisor_status=advisor,
    )

    assert out["status"] == "OK"
    assert out["structure"]["market_bias"] == "BEARISH"
    assert out["plan"]["trade_idea"] == "Bear call spread"
    assert out["plan"]["posture"] in {"WAIT_PULLBACK", "WAIT_CONFIRMATION", "READY_AFTER_CONFIRMATION"}
    assert out["plan"]["short_strike_hint"] is not None
    assert out["plan"]["hedge_strike_hint"] is not None
    assert out["structure"]["regime"] in {"SWING_DOWN", "BEARISH_DRIFT"}
    assert "rsi_divergence" in out["structure"]
    assert "higher_tf_confluence" in out["structure"]
    assert "buyers" in out["levels"]
    assert "sellers" in out["levels"]
    assert isinstance(out["plan"]["required_confluences"], list)


def test_build_intraday_index_chart_payload_stands_aside_in_range() -> None:
    candles_5m = [
        _candle("2026-04-04T09:15:00+05:30", 23000, 23020, 22985, 23005),
        _candle("2026-04-04T09:20:00+05:30", 23005, 23018, 22992, 22998),
        _candle("2026-04-04T09:25:00+05:30", 22998, 23012, 22990, 23004),
        _candle("2026-04-04T09:30:00+05:30", 23004, 23016, 22996, 23002),
        _candle("2026-04-04T09:35:00+05:30", 23002, 23015, 22994, 23001),
        _candle("2026-04-04T09:40:00+05:30", 23001, 23014, 22993, 23006),
        _candle("2026-04-04T09:45:00+05:30", 23006, 23017, 22995, 23003),
        _candle("2026-04-04T09:50:00+05:30", 23003, 23013, 22996, 23000),
    ]
    candles_15m = [
        _candle("2026-04-04T09:15:00+05:30", 23000, 23020, 22985, 23004),
        _candle("2026-04-04T09:30:00+05:30", 23004, 23017, 22993, 23001),
        _candle("2026-04-04T09:45:00+05:30", 23001, 23015, 22995, 23000),
    ]
    candles_60m = [
        _candle("2026-04-04T09:15:00+05:30", 23000, 23025, 22980, 23002),
        _candle("2026-04-04T10:15:00+05:30", 23002, 23020, 22988, 23001),
    ]

    out = paper_server._build_intraday_index_chart_payload_from_candles(
        candles_5m=candles_5m,
        candles_15m=candles_15m,
        candles_60m=candles_60m,
        advisor_status={"pcr_bias": "NEUTRAL", "oi_build_bias": "NEUTRAL"},
    )

    assert out["status"] == "OK"
    assert out["structure"]["market_bias"] == "NEUTRAL"
    assert out["plan"]["trade_idea"] == "Stand aside / range read"
    assert out["plan"]["posture"] == "WAIT_EDGE_TEST"
    assert out["structure"]["higher_tf_confluence"]["status"] in {"MIXED", "PARTIAL", "ALIGNED"}
