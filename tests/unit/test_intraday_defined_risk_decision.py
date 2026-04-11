from __future__ import annotations

from copy import deepcopy

from market_ai.intraday_defined_risk.decision import make_decision


BASE_PAYLOAD = {
    "date": "2026-04-03",
    "time_ist": "09:55",
    "underlying": {
        "symbol": "NIFTY",
        "spot": 22420.0,
        "vwap": 22470.0,
        "or15_high": 22510.0,
        "or15_low": 22440.0,
        "ohlcv_5m": [
            {"t": "09:20", "o": 22500.0, "h": 22510.0, "l": 22480.0, "c": 22495.0, "v": 1000},
            {"t": "09:25", "o": 22495.0, "h": 22500.0, "l": 22470.0, "c": 22480.0, "v": 1000},
            {"t": "09:30", "o": 22480.0, "h": 22485.0, "l": 22455.0, "c": 22460.0, "v": 1000},
            {"t": "09:35", "o": 22460.0, "h": 22465.0, "l": 22435.0, "c": 22438.0, "v": 1000},
            {"t": "09:40", "o": 22438.0, "h": 22440.0, "l": 22410.0, "c": 22420.0, "v": 1000},
            {"t": "09:45", "o": 22420.0, "h": 22425.0, "l": 22390.0, "c": 22415.0, "v": 1000},
            {"t": "09:50", "o": 22415.0, "h": 22418.0, "l": 22395.0, "c": 22405.0, "v": 1000},
            {"t": "09:55", "o": 22405.0, "h": 22412.0, "l": 22398.0, "c": 22400.0, "v": 1000},
        ],
    },
    "volatility": {
        "india_vix": 16.0,
        "rv30_pct": 0.42,
        "atr_5m": 55.0,
    },
    "options_chain": {
        "expiry": "2026-04-09",
        "strikes": [
            {"k": 22500, "ce_ltp": 64.0, "pe_ltp": 41.0, "ce_bid": 63.0, "ce_ask": 65.0, "pe_bid": 40.0, "pe_ask": 42.0, "ce_delta": 0.31, "pe_delta": -0.12},
            {"k": 22600, "ce_ltp": 43.0, "pe_ltp": 28.0, "ce_bid": 42.0, "ce_ask": 44.0, "pe_bid": 27.0, "pe_ask": 29.0, "ce_delta": 0.22, "pe_delta": -0.09},
            {"k": 22700, "ce_ltp": 12.0, "pe_ltp": 18.0, "ce_bid": 11.5, "ce_ask": 12.5, "pe_bid": 17.5, "pe_ask": 18.5, "ce_delta": 0.07, "pe_delta": -0.05},
            {"k": 22200, "ce_ltp": 14.0, "pe_ltp": 10.0, "ce_bid": 13.5, "ce_ask": 14.5, "pe_bid": 9.5, "pe_ask": 10.5, "ce_delta": 0.05, "pe_delta": -0.07},
            {"k": 22300, "ce_ltp": 19.0, "pe_ltp": 20.0, "ce_bid": 18.5, "ce_ask": 19.5, "pe_bid": 19.0, "pe_ask": 21.0, "ce_delta": 0.08, "pe_delta": -0.22},
            {"k": 22400, "ce_ltp": 26.0, "pe_ltp": 37.0, "ce_bid": 25.0, "ce_ask": 27.0, "pe_bid": 36.0, "pe_ask": 38.0, "ce_delta": 0.11, "pe_delta": -0.31},
        ],
    },
    "risk": {
        "lot_size": 65,
        "max_risk_rupees": 10000,
        "max_margin_rupees": None,
        "slippage_points_assumption": 0.5,
    },
}


def test_downtrend_payload_generates_bear_call_spread() -> None:
    decision = make_decision(deepcopy(BASE_PAYLOAD))
    assert decision["decision"] == "TRADE"
    assert decision["strategy"] == "BEAR_CALL_CREDIT_SPREAD"
    assert decision["legs"][0]["action"] == "SELL"
    assert decision["legs"][0]["type"] == "CALL"
    assert decision["max_loss_rupees_per_lot"] > 0
    assert decision["recommended_lots"] >= 1


def test_stale_last_bar_forces_no_trade() -> None:
    payload = deepcopy(BASE_PAYLOAD)
    payload["time_ist"] = "10:00"
    decision = make_decision(payload)
    assert decision["decision"] == "NO_TRADE"
    assert any("stale" in reason.lower() for reason in decision["reasons"])


def test_range_before_1000_stays_no_trade() -> None:
    payload = deepcopy(BASE_PAYLOAD)
    payload["time_ist"] = "09:55"
    payload["underlying"]["spot"] = 22476.0
    payload["underlying"]["vwap"] = 22475.0
    payload["volatility"]["rv30_pct"] = 0.20
    payload["underlying"]["ohlcv_5m"] = [
        {"t": "09:30", "o": 22480.0, "h": 22490.0, "l": 22460.0, "c": 22472.0, "v": 1000},
        {"t": "09:35", "o": 22472.0, "h": 22488.0, "l": 22462.0, "c": 22478.0, "v": 1000},
        {"t": "09:40", "o": 22478.0, "h": 22489.0, "l": 22463.0, "c": 22470.0, "v": 1000},
        {"t": "09:45", "o": 22470.0, "h": 22486.0, "l": 22461.0, "c": 22479.0, "v": 1000},
        {"t": "09:50", "o": 22479.0, "h": 22487.0, "l": 22460.0, "c": 22471.0, "v": 1000},
        {"t": "09:55", "o": 22471.0, "h": 22485.0, "l": 22464.0, "c": 22476.0, "v": 1000},
    ]
    decision = make_decision(payload)
    assert decision["decision"] == "NO_TRADE"


def test_wide_bid_ask_rows_are_skipped() -> None:
    payload = deepcopy(BASE_PAYLOAD)
    payload["options_chain"]["strikes"][1]["ce_bid"] = 35.0
    payload["options_chain"]["strikes"][1]["ce_ask"] = 50.0
    decision = make_decision(payload)
    assert decision["decision"] == "NO_TRADE" or decision["legs"][0]["strike"] != 22600
