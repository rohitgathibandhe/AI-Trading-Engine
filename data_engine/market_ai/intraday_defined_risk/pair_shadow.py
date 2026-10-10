"""STRUCTURE-CHOICE SHADOW — the evidence the agent needs to learn WHICH structure to trade.

Goal (user, 2026-10-09): the agent should decide by itself, for a given directional read, whether to
BUY or SELL — bearish: put-debit vs bear-call; bullish: call-debit vs bull-put. That choice can only
be learned from outcomes of BOTH structures under the SAME conditions at the SAME moment. The
backtest can't supply that honestly (live-capture data is ~16-min cadence; the dense set's synthetic
4% spreads bias every selling result negative), and the shadow book enters at a fixed 09:35 with no
read. So on every directional read this module opens a GHOST PAIR — the debit AND the credit for that
direction — marks both on the same live chain every cycle until 15:15, and writes one row per pair
to state/pair_shadow.jsonl together with the market features at entry. That table is the training /
validation set for a structure-selection rule; nothing here changes what the agent trades.

Triggers:
  ENTRY   — the agent entered a directional trade (pairs its structure with the opposite one).
  READ    — a directional stand-aside the selector still had a bias on (no momentum, debit blackout,
            weak confluence...). Rate-limited: >= 30 min apart, <= 4 per direction per day.

Structures (200pt wide, same lots as the paired trade else 2):
  BEAR_CALL / BULL_PUT  short nearest 0.20 delta OTM, long 200 further OTM. Sold at bid / bought at ask.
  PUT_DEBIT / CALL_DEBIT long nearest 0.45 delta, short 200 further OTM. Bought at ask / sold at bid.
  Marked at the cost to CLOSE (real friction, never mid).

Exit candidates per structure (decide nothing live):
  RIDE_TO_CLOSE   hold to 15:15.
  SELLER_EXIT     credit: TP 50% of credit; stop 2x credit or spot within 50pts of the short strike
                  (strike-buffer-aware — no momentum stops; see project_exit_whipsaw_rootcause).
  SELLER_TIGHT    credit: as above, stop 1x credit.
  DEBIT_EXIT      debit: TP +60% of debit; stop -50% of debit.

Pure recorder: every public function is guarded and can never break the trading loop. Open pairs are
persisted to disk so a mid-session restart doesn't lose them.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

_STATE_DIR = Path(__file__).resolve().parent.parent / "state"
_OPEN_FILE = _STATE_DIR / "pair_shadow_open.json"
_LEDGER = _STATE_DIR / "pair_shadow.jsonl"

_CLOSE_HHMM = (15, 15)
_WIDTH_PTS = 200.0
_DEFAULT_LOTS = 2
_READ_MAX_PER_DIR_PER_DAY = 4
_READ_MIN_GAP_MINUTES = 30
_READ_MIN_CONVICTION = 0.45
_STRIKE_BUFFER_STOP_PTS = 50.0

# name: (option side, is_credit, target delta, delta band, OTM direction of the 2nd leg: +1 up / -1 down)
_SPECS = {
    "BEAR_CALL": ("CALL", True, 0.20, (0.12, 0.30), +1),
    "BULL_PUT": ("PUT", True, 0.20, (0.12, 0.30), -1),
    "PUT_DEBIT": ("PUT", False, 0.45, (0.30, 0.60), -1),
    "CALL_DEBIT": ("CALL", False, 0.45, (0.30, 0.60), +1),
}
PAIRS = {"BEARISH": ("PUT_DEBIT", "BEAR_CALL"), "BULLISH": ("CALL_DEBIT", "BULL_PUT")}
_STRAT_DIR = {"PUT_DEBIT": "BEARISH", "BEAR_CALL": "BEARISH", "CALL_DEBIT": "BULLISH", "BULL_PUT": "BULLISH"}

# Features captured at entry — the inputs a structure-selection rule could condition on.
_FEATURE_KEYS = ("trend_efficiency_ratio", "selector_bias", "selector_conviction", "selector_condition",
                 "selector_family", "selector_iv", "vol_regime", "india_vix", "distance_to_call_wall",
                 "distance_to_put_wall", "call_wall_oi_velocity", "opening_range_break_state",
                 "option_chain_pressure_state", "smart_money_bias", "daily_trend", "session_move_pts",
                 "last_hour_change_pct", "expected_move_pts")


def _load() -> dict:
    try:
        return json.loads(_OPEN_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _save(state: dict) -> None:
    _OPEN_FILE.write_text(json.dumps(state))


def _side(chain, side: str) -> dict[float, Any]:
    out = {}
    for q in chain.quotes:
        t = str(getattr(q.option_type, "value", q.option_type)).upper()
        if (side == "CALL" and t in ("CALL", "CE")) or (side == "PUT" and t in ("PUT", "PE")):
            out[float(q.strike)] = q
    return out


def strategy_key(strategy: str) -> str | None:
    s = (strategy or "").upper()
    for k in _SPECS:
        if k in s:
            return k
    return None


def _build(snapshot, name: str) -> dict | None:
    side, is_credit, tgt, band, wing = _SPECS[name]
    chain = snapshot.option_chain
    spot = float(chain.spot)
    quotes = _side(chain, side)
    # credit: the SHORT is OTM (calls above / puts below spot); debit: the LONG is near the money.
    def _ok(k, q):
        if q.delta is None or not (band[0] <= abs(q.delta) <= band[1]) or q.bid <= 0:
            return False
        if is_credit:
            return k > spot if side == "CALL" else k < spot
        return True
    cands = [(abs(abs(q.delta) - tgt), k) for k, q in quotes.items() if _ok(k, q)]
    if not cands:
        return None
    first = min(cands)[1]
    second = first + wing * _WIDTH_PTS
    a, b = quotes.get(first), quotes.get(second)
    if a is None or b is None:
        return None
    if is_credit:     # sell `first` at bid, buy `second` at ask
        px = float(a.bid) - float(b.ask or 0)
        legs = {"short": first, "long": second}
    else:             # buy `first` at ask, sell `second` at bid
        px = float(a.ask or a.ltp) - float(b.bid or 0)
        legs = {"long": first, "short": second}
    if px <= 0:
        return None
    return {"structure": name, "side": side, "credit": is_credit, **legs,
            "first_delta": round(abs(float(a.delta)), 3), "entry_pts": round(px, 2), "marks": []}


def _mark_pts(snapshot, leg: dict) -> float | None:
    """Value to CLOSE now, in points: credit -> cost to buy back; debit -> proceeds from selling."""
    q = _side(snapshot.option_chain, leg["side"])
    s, l = q.get(leg["short"]), q.get(leg["long"])
    if s is None or l is None:
        return None
    if leg["credit"]:
        ask_s = float(s.ask or s.ltp or 0)
        return None if ask_s <= 0 else max(0.0, ask_s - float(l.bid or 0))
    bid_l = float(l.bid or 0)
    return max(0.0, bid_l - float(s.ask or s.ltp or 0))


def _pnl(leg: dict, mark_pts: float, lot_value: float) -> float:
    if leg["credit"]:
        return (leg["entry_pts"] - mark_pts) * lot_value
    return (mark_pts - leg["entry_pts"]) * lot_value


def open_pair(snapshot, direction: str, trigger: str, *, metadata: dict | None = None,
              lots: int | None = None, lot_size: int | None = None, paired_entry: str | None = None,
              actual_structure: str | None = None, note: str = "") -> None:
    """Open a ghost pair (debit + credit) for `direction`. READ triggers are rate-limited. Never raises."""
    try:
        if direction not in PAIRS:
            return
        ts = snapshot.timestamp
        if (ts.hour, ts.minute) >= _CLOSE_HHMM:
            return
        day = ts.date().isoformat()
        state = _load()
        if state.get("day") != day:
            state = {"day": day, "open": [], "read_counts": {}, "read_last": {}}
        if trigger == "READ":
            last = state["read_last"].get(direction)
            if last and (ts - datetime.fromisoformat(last)).total_seconds() < _READ_MIN_GAP_MINUTES * 60:
                return
            if state["read_counts"].get(direction, 0) >= _READ_MAX_PER_DIR_PER_DAY:
                return
        legs = [b for b in (_build(snapshot, n) for n in PAIRS[direction]) if b is not None]
        if not legs:
            return
        spot = round(float(snapshot.option_chain.spot), 2)
        m = metadata or {}
        pair = {"pair_id": f"{ts.isoformat(timespec='seconds')}_{direction}", "session_date": day,
                "opened_at": ts.isoformat(timespec="seconds"), "direction": direction, "trigger": trigger,
                "actual_structure": actual_structure, "paired_entry": paired_entry, "note": note[:240],
                "spot": spot, "minutes_since_open": (ts.hour * 60 + ts.minute) - (9 * 60 + 15),
                "lot_value": float(lot_size or getattr(snapshot, "lot_size", 65)) * float(lots or _DEFAULT_LOTS),
                "features": {k: m.get(k) for k in _FEATURE_KEYS if m.get(k) is not None},
                "legs": legs, "spots": [{"t": ts.isoformat(timespec="seconds"), "spot": spot}]}
        for leg in legs:
            leg["marks"].append({"t": pair["opened_at"], "spot": spot, "pnl": 0.0})
        state["open"].append(pair)
        if trigger == "READ":
            state["read_counts"][direction] = state["read_counts"].get(direction, 0) + 1
            state["read_last"][direction] = pair["opened_at"]
        _save(state)
    except Exception:
        pass


def _credit_exit(leg: dict, lot_value: float, stop_mult: float):
    credit_rs = leg["entry_pts"] * lot_value
    for m in leg["marks"][1:]:
        if m["pnl"] >= 0.5 * credit_rs:
            return m["t"], m["pnl"], "TAKE_PROFIT_50"
        if m["pnl"] <= -stop_mult * credit_rs:
            return m["t"], m["pnl"], f"STOP_{stop_mult:g}X_CREDIT"
        breach = (m["spot"] >= leg["short"] - _STRIKE_BUFFER_STOP_PTS if leg["side"] == "CALL"
                  else m["spot"] <= leg["short"] + _STRIKE_BUFFER_STOP_PTS)
        if breach:
            return m["t"], m["pnl"], "STRIKE_BUFFER"
    last = leg["marks"][-1]
    return last["t"], last["pnl"], "CLOSE"


def _debit_exit(leg: dict, lot_value: float):
    debit_rs = leg["entry_pts"] * lot_value
    for m in leg["marks"][1:]:
        if m["pnl"] >= 0.6 * debit_rs:
            return m["t"], m["pnl"], "TAKE_PROFIT_60"
        if m["pnl"] <= -0.5 * debit_rs:
            return m["t"], m["pnl"], "STOP_50PCT"
    last = leg["marks"][-1]
    return last["t"], last["pnl"], "CLOSE"


def _finalize(pair: dict) -> None:
    lv = pair["lot_value"]
    out_legs = {}
    for leg in pair["legs"]:
        marks = leg["marks"]
        last = marks[-1]
        cands = {"RIDE_TO_CLOSE": {"exit_at": last["t"], "pnl_rupees": round(last["pnl"], 2)}}
        if leg["credit"]:
            for name, mult in (("SELLER_EXIT", 2.0), ("SELLER_TIGHT", 1.0)):
                t, p, why = _credit_exit(leg, lv, mult)
                cands[name] = {"exit_at": t, "pnl_rupees": round(p, 2), "reason": why}
            max_loss = (_WIDTH_PTS - leg["entry_pts"]) * lv
        else:
            t, p, why = _debit_exit(leg, lv)
            cands["DEBIT_EXIT"] = {"exit_at": t, "pnl_rupees": round(p, 2), "reason": why}
            max_loss = leg["entry_pts"] * lv
        out_legs[leg["structure"]] = {
            "first": leg.get("short") if leg["credit"] else leg.get("long"),
            "second": leg.get("long") if leg["credit"] else leg.get("short"),
            "first_delta": leg["first_delta"], "entry_pts": leg["entry_pts"],
            "entry_rupees": round(leg["entry_pts"] * lv, 2), "max_loss_rupees": round(max_loss, 2),
            "n_marks": len(marks), "mfe_rupees": round(max(m["pnl"] for m in marks), 2),
            "mae_rupees": round(min(m["pnl"] for m in marks), 2), "candidates": cands}
    spots = [m["spot"] for leg in pair["legs"] for m in leg["marks"]]
    row = {k: pair[k] for k in ("pair_id", "session_date", "opened_at", "direction", "trigger",
                                "actual_structure", "paired_entry", "note", "spot", "minutes_since_open",
                                "lot_value", "features")}
    row.update({"spot_at_close": pair["legs"][0]["marks"][-1]["spot"], "spot_high": max(spots),
                "spot_low": min(spots), "structures": out_legs})
    with _LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")


def reprice(snapshot) -> None:
    """Every cycle: mark each open leg at the value to close; finalize pairs at 15:15 / on a day roll."""
    try:
        state = _load()
        if not state.get("open"):
            return
        ts = snapshot.timestamp
        spot = round(float(snapshot.option_chain.spot), 2)
        same_day = lambda p: str(ts.date()) == p["session_date"]
        survivors = []
        for pair in state["open"]:
            if same_day(pair):
                for leg in pair["legs"]:
                    mk = _mark_pts(snapshot, leg)
                    if mk is not None:
                        leg["marks"].append({"t": ts.isoformat(timespec="seconds"), "spot": spot,
                                             "pnl": round(_pnl(leg, mk, pair["lot_value"]), 2)})
            if (ts.hour, ts.minute) >= _CLOSE_HHMM or not same_day(pair):
                _finalize(pair)
            else:
                survivors.append(pair)
        state["open"] = survivors
        _save(state)
    except Exception:
        pass


def read_direction(decision) -> str | None:
    """Directional read behind a NO_TRADE: the selector's bias if conviction clears the floor."""
    try:
        m = getattr(decision, "metadata", None) or {}
        bias = str(m.get("selector_bias") or "").upper()
        conv = float(m.get("selector_conviction") or 0.0)
        return bias if bias in PAIRS and conv >= _READ_MIN_CONVICTION else None
    except Exception:
        return None


def direction_of(strategy: str) -> str | None:
    k = strategy_key(strategy)
    return _STRAT_DIR.get(k) if k else None
