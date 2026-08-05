"""EXIT SHADOW BOOK — forward evidence for exit rules, at zero risk to the live path.

The problem this exists to solve: debit trades ride to the 15:15 flatten and keep ~22% of their
peak (Rs 12,087 peak -> Rs 2,615 realized over the first 7 trades). The obvious fix — a P&L trail —
is VALIDATED-NEGATIVE on the ordering-robust harness (baseline median 109,529 vs armR 0.20: 12,643,
armR 0.30: 37,464, monotonic). Why: ~90% of the net comes from ~4 home-run trades out of 94, and at
0.2-0.3R a fader and a runner are indistinguishable, so any trail that banks the faders decapitates
the runners on a normal mid-flight pullback.

Two things follow, and this module addresses both:

1. THE MISSING SIGNAL. A P&L trail fires on premium noise (IV, theta, spread wobble). A runner is
   defined by the UNDERLYING continuing to make new favourable extremes; a fader is defined by spot
   retracing. So we trail SPOT, not P&L — `SPOT_REVERSAL_*` below. That is a real-time
   fader/runner discriminator, which a blanket P&L trail is not.

2. THE MEASUREMENT. That harness charges SYNTHETIC bid/ask at ~4.0% of mid where the live chain is
   ~0.8%, so it over-penalises every rule that exits mid-day by ~5x the real friction — precisely
   the rules under test. Per the project's evidence hierarchy (live/forward > backtest), the honest
   court for an exit rule is live marks. So this module DECIDES NOTHING: it records what each
   candidate rule WOULD have returned, marked at the same real live prices the agent itself used,
   and writes one row per closed trade to state/exit_shadow.jsonl.

Nothing here can change live behaviour. It is a recorder. A rule graduates only on the forward
record, through the existing promotion machinery — never off a backtest and never off a handful of
recent trades.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_STATE_DIR = Path(__file__).resolve().parent.parent / "state"
_PATH_FILE = _STATE_DIR / "exit_shadow_path.json"
_LEDGER = _STATE_DIR / "exit_shadow.jsonl"

# Spot-reversal thresholds to evaluate, in index points retraced from the favourable extreme.
_REVERSAL_PTS = (15.0, 25.0, 40.0)
# Classic P&L trail, kept as a reference arm so the ledger shows the thing already rejected
# on the harness measured against live friction instead of synthetic 4% spreads.
_PNL_TRAIL_ARM_R = 0.30
_PNL_TRAIL_GIVEBACK = 0.35


def _favourable_direction(strategy_name: str) -> int:
    """-1 = position profits when spot FALLS, +1 = profits when spot RISES, 0 = non-directional."""
    s = (strategy_name or "").upper()
    if "PUT_DEBIT" in s or "BEAR_CALL" in s:
        return -1
    if "CALL_DEBIT" in s or "BULL_PUT" in s:
        return 1
    return 0


# ── recording ────────────────────────────────────────────────────────────────────────────────
def record_mark(position, snapshot, open_pnl: float) -> None:
    """Append one (time, spot, pnl) sample for the open position. Never raises."""
    try:
        entry_key = position.entry_time.isoformat()
        try:
            buf = json.loads(_PATH_FILE.read_text())
        except (OSError, ValueError):
            buf = {}
        if buf.get("entry_timestamp") != entry_key:      # new trade -> fresh buffer
            buf = {"entry_timestamp": entry_key,
                   "strategy": position.structure.strategy.value,
                   "entry_debit_points": abs(float(position.entry_credit_points)),
                   "marks": []}
        buf["marks"].append({
            "t": snapshot.timestamp.isoformat(timespec="seconds"),
            "spot": round(float(snapshot.option_chain.spot), 2),
            "pnl": round(float(open_pnl), 2),
        })
        _PATH_FILE.write_text(json.dumps(buf))
    except Exception:
        pass


# ── candidate rules: each returns (exit_time, pnl) given the recorded path ────────────────────
def _ride_to_close(marks, **_):
    return marks[-1]["t"], marks[-1]["pnl"]


def _spot_reversal(marks, *, direction: int, pts: float, **_):
    """THE FADER/RUNNER SIGNAL. Track spot's favourable extreme; exit once spot has retraced
    `pts` from it while the position is green. A runner keeps extending the extreme and is never
    touched; a fader trips the moment the underlying turns."""
    if direction == 0:
        return _ride_to_close(marks)
    extreme = marks[0]["spot"]
    for m in marks:
        spot = m["spot"]
        if (direction < 0 and spot < extreme) or (direction > 0 and spot > extreme):
            extreme = spot
        retrace = (spot - extreme) if direction < 0 else (extreme - spot)
        if retrace >= pts and m["pnl"] > 0:
            return m["t"], m["pnl"]
    return _ride_to_close(marks)


def _pnl_trail(marks, *, debit_points: float, lot_value: float, **_):
    """Reference arm: the already-rejected P&L trail, armed on R = debit paid."""
    one_r = debit_points * lot_value
    if one_r <= 0:
        return _ride_to_close(marks)
    peak = 0.0
    armed = False
    for m in marks:
        peak = max(peak, m["pnl"])
        if peak >= _PNL_TRAIL_ARM_R * one_r:
            armed = True
        if armed and m["pnl"] <= peak * (1.0 - _PNL_TRAIL_GIVEBACK):
            return m["t"], m["pnl"]
    return _ride_to_close(marks)


def _afternoon_lock(marks, *, hhmm: str = "14:00", **_):
    for m in marks:
        if m["t"][11:16] >= hhmm and m["pnl"] > 0:
            return m["t"], m["pnl"]
    return _ride_to_close(marks)


def _evaluate(buf: dict, lot_value: float) -> dict[str, Any]:
    marks = buf.get("marks") or []
    if not marks:
        return {}
    direction = _favourable_direction(buf.get("strategy", ""))
    debit = float(buf.get("entry_debit_points") or 0.0)
    kw = {"direction": direction, "debit_points": debit, "lot_value": lot_value}
    rules: dict[str, Any] = {"RIDE_TO_CLOSE": _ride_to_close(marks, **kw)}
    for pts in _REVERSAL_PTS:
        rules[f"SPOT_REVERSAL_{int(pts)}"] = _spot_reversal(marks, pts=pts, **kw)
    rules["PNL_TRAIL_R30_GB35"] = _pnl_trail(marks, **kw)
    rules["AFTERNOON_LOCK_1400"] = _afternoon_lock(marks, **kw)
    return {name: {"exit_at": t, "pnl_rupees": round(p, 2)} for name, (t, p) in rules.items()}


def finalize(position, exit_event: dict) -> None:
    """Called once at exit: score every candidate rule over the recorded path, append one ledger
    row, clear the buffer. Never raises — this must not be able to break the trading loop."""
    try:
        try:
            buf = json.loads(_PATH_FILE.read_text())
        except (OSError, ValueError):
            return
        if buf.get("entry_timestamp") != position.entry_time.isoformat():
            return
        marks = buf.get("marks") or []
        lot_value = float(position.lot_size) * float(position.lots)
        row = {
            "session_date": exit_event.get("session_date"),
            "strategy": buf.get("strategy"),
            "entry_timestamp": buf.get("entry_timestamp"),
            "entry_debit_points": buf.get("entry_debit_points"),
            "lot_value": lot_value,
            "n_marks": len(marks),
            "spot_at_entry": marks[0]["spot"] if marks else None,
            "spot_extreme": (min(m["spot"] for m in marks) if _favourable_direction(buf.get("strategy", "")) < 0
                             else max(m["spot"] for m in marks)) if marks else None,
            "actual": {"exit_at": exit_event.get("exit_timestamp"),
                       "exit_reason": exit_event.get("exit_reason"),
                       "pnl_rupees": exit_event.get("realized_paper_pnl")},
            "mfe_rupees": exit_event.get("mfe_rupees"),
            "candidates": _evaluate(buf, lot_value),
        }
        with _LEDGER.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass
    finally:
        try:
            _PATH_FILE.unlink()
        except OSError:
            pass
