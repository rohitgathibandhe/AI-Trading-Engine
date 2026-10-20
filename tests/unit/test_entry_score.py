from datetime import datetime, time, timedelta

from market_ai.intraday_defined_risk import entry_score as es
from market_ai.intraday_defined_risk import strategy_selector as sel


def _bars(closes, day=datetime(2026, 10, 13, 9, 15)):
    out = []
    for k, c in enumerate(closes):
        out.append({"t": day + timedelta(minutes=5 * k), "o": c - 2, "h": c + 6, "l": c - 6, "c": c})
    return out


def test_features_need_a_swing_stop_and_score_is_a_probability():
    # a dip (swing low at bar 4) then a rally: a LONG has a structure stop, a SHORT (no swing high above) may not
    closes = [100, 98, 95, 92, 88, 91, 95, 99, 103, 107, 111]
    T = _bars(closes)
    f = es.features(T, len(T) - 1, 1, {"atr_d": 200, "dtrend": 1})
    assert f is not None and f["risk_pts"] >= 8 and f["dir"] == 1
    if es._model():
        assert 0 < es.prob(f) < 1


def test_evaluate_uses_only_completed_bars():
    T = _bars([100 + k for k in range(12)])
    now = T[-1]["t"] + timedelta(minutes=2)              # last bar still forming
    r = es.evaluate(T, now)
    assert r["at"] == T[-2]["t"].isoformat()


def _choice(cond, struct):
    c = sel.StrategyChoice(condition=cond, iv_regime="NORMAL", read=None, conviction=0.5)
    c.family, c.structures, c.rationale = sel.FAM_DIRECTIONAL_DEBIT, [struct], "x"
    return c


def test_gate_blocks_low_score_and_passes_high(monkeypatch):
    monkeypatch.setattr(sel, "_EDGE_ONLY", True)
    monkeypatch.setattr(sel, "_DEBIT_BLACKOUT", None)
    monkeypatch.setattr(sel, "_ENTRY_SCORE_MIN", 0.30)
    md = {"entry_score": {"long": 0.20, "short": 0.50}}
    assert sel._edge_only(_choice("BREAKOUT_UP", "CALL_DEBIT_SPREAD"), md, time(11, 0)).structures == []
    assert md["entry_score_skip"]
    md = {"entry_score": {"long": 0.20, "short": 0.50}}
    assert sel._edge_only(_choice("BREAKOUT_DOWN", "PUT_DEBIT_SPREAD"), md, time(11, 0)).structures == ["PUT_DEBIT_SPREAD"]
    md = {"entry_score": {}}                             # no score (no swing stop) = no trade
    assert sel._edge_only(_choice("BREAKOUT_DOWN", "PUT_DEBIT_SPREAD"), md, time(11, 0)).structures == []


def test_gate_off_by_default(monkeypatch):
    monkeypatch.setattr(sel, "_EDGE_ONLY", True)
    monkeypatch.setattr(sel, "_DEBIT_BLACKOUT", None)
    monkeypatch.setattr(sel, "_ENTRY_SCORE_MIN", 0.0)
    md = {"entry_score": {"long": 0.01}}
    assert sel._edge_only(_choice("BREAKOUT_UP", "CALL_DEBIT_SPREAD"), md, time(11, 0)).structures == ["CALL_DEBIT_SPREAD"]
