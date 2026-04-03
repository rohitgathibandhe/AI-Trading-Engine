from __future__ import annotations

import csv
import json
from pathlib import Path

import scripts.paper_pnl_server as paper_server
from scripts.paper_pnl_server import load_positions


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
