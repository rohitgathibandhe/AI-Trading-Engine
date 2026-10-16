from datetime import datetime, timedelta

from market_ai.intraday_defined_risk.data_models import OhlcvBar
from market_ai.intraday_defined_risk.features import swing_structure_confirmation
from market_ai.intraday_defined_risk.ops_runtime import _early_structure_override

# 2026-10-16 09:15-09:55 (5m): big first-candle drop, low 22,914 at 09:35, swing high 22,981 at 09:40,
# 09:55 close 23,027 breaks it — the structure turned up 20 min before two closes above the OR high.
_OHLC = [(23065.1, 23076.55, 22953.65, 22955.9), (22958.05, 22996.4, 22956.5, 22987.65),
         (22987.9, 22988.95, 22942.15, 22952.3), (22952.95, 22961.05, 22927.35, 22932.4),
         (22934.2, 22953.2, 22913.8, 22938.0), (22938.6, 22981.45, 22936.35, 22974.7),
         (22974.75, 22974.75, 22940.9, 22963.65), (22964.5, 22987.9, 22958.35, 22983.65),
         (22984.0, 23027.2, 22982.65, 23026.95)]


def _bars(rows):
    t0 = datetime(2026, 10, 16, 9, 15)
    return [OhlcvBar(timestamp=t0 + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=1.0)
            for i, (o, h, l, c) in enumerate(rows)]


def _vwap(bars):
    tp = [(b.high + b.low + b.close) / 3 for b in bars]
    return sum(tp) / len(tp)


def test_structure_turns_up_on_break_of_swing_high_after_session_low():
    bars = _bars(_OHLC)
    assert swing_structure_confirmation(bars[:-1], _vwap(bars[:-1])) == "UNCONFIRMED"   # 09:50 close: not yet
    assert swing_structure_confirmation(bars, _vwap(bars)) == "UP_CONFIRMED"            # 09:55 close 23,027


def test_structure_mirror_down():
    mirror = [(-o, -l, -h, -c) for (o, h, l, c) in _OHLC]
    bars = _bars(mirror)
    assert swing_structure_confirmation(bars, _vwap(bars)) == "DOWN_CONFIRMED"


def test_structure_needs_vwap_and_bars():
    assert swing_structure_confirmation(_bars(_OHLC[:3]), 22950.0) == "UNCONFIRMED"
    assert swing_structure_confirmation(_bars(_OHLC), None) == "UNCONFIRMED"


def test_gate_override_confirms_and_vetoes(monkeypatch):
    monkeypatch.delenv("EARLY_SWING_CONFIRM", raising=False)
    up = {"execution_5m_structure": "UP_CONFIRMED"}
    assert _early_structure_override(False, True, up, {}) == (True, False)     # bull-put now allowed
    assert _early_structure_override(True, False, up, {}) == (True, True)      # bear-call vetoed by up CoCH
    assert _early_structure_override(False, True, {}, {}) == (False, False)    # nothing changes without a read
    monkeypatch.setenv("EARLY_SWING_CONFIRM", "0")
    assert _early_structure_override(False, True, up, {}) == (False, False)    # toggle off = old behaviour


def test_stale_overnight_position_is_booked_not_erased(tmp_path, monkeypatch):
    """A position stranded past 15:15 was dropped at the next start with NO PAPER_EXIT — its P&L vanished."""
    import json
    from types import SimpleNamespace
    from market_ai.intraday_defined_risk import ops_runtime as ops

    paths = ops.OpsPaths(paper_state=tmp_path / "state.json", paper_trades=tmp_path / "trades.jsonl")
    paths.paper_state.write_text(json.dumps({"active_position": {"x": 1}, "last_mark_pnl_rupees": -484.25,
                                             "mfe_rupees": 205.0, "mae_rupees": -600.0}))
    stale = SimpleNamespace(entry_time=datetime.now() - timedelta(days=1), metadata={"playbook": "SEL_BREAKOUT_UP"},
                            structure=SimpleNamespace(strategy=SimpleNamespace(value="BULL_PUT_CREDIT_SPREAD")))
    monkeypatch.setattr(ops, "deserialize_open_position", lambda _d: stale)
    monkeypatch.setattr(ops, "save_paper_position", lambda *a, **k: None)
    assert ops.load_paper_position(paths, drop_stale=True) is None
    ev = [json.loads(l) for l in paths.paper_trades.read_text().splitlines()]
    assert ev and ev[-1]["event"] == "PAPER_EXIT" and ev[-1]["exit_reason"] == "STALE_OVERNIGHT_EXIT"
    assert ev[-1]["realized_paper_pnl"] == -484.25
