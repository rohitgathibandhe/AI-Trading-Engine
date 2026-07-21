#!/usr/bin/env python
"""Daily SHADOW BOOK — the honest evidence engine for the seller/all-weather build.

Every day, from the live-captured option chain (rolling_option_live/<date>/chain_snapshots.jsonl),
compute what the REGIME-MATCHED credit trades WOULD have made, bypassing the agent's construction
filters entirely. This accumulates the real, forward, out-of-sample record we need to decide
whether bull-put / bear-call / iron-condor are genuine edges — BEFORE any of them drives live money.

For each day it logs, for bull-put (up/sideways bet), bear-call (down/sideways bet), and
iron-condor (range bet): entry credit, final mark, P&L/lot, and the day's actual open->close
move (so we can later bucket by realized regime). Appends one JSON line to
state/shadow_book.jsonl. Scheduled at 15:40 IST after the capture stops.

NOT a trade and NOT the agent's P&L — it's a what-if ledger for validation. Mid-priced
(real fills cost the spread); treat magnitudes as indicative, direction/consistency as signal.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
IST = timezone(timedelta(hours=5, minutes=30))
LOT = 75
ENTRY_AFTER = "09:30"   # let the opening range form
MARK_BY = "15:20"

# Transaction cost per FILL, in index points, on top of the bid/ask spread (which is already
# modelled via sell@bid / buy@ask). The bid/ask captures slippage; this captures the fee stack
# the exchange/broker take, which the spread does NOT include and which was silently zero before
# (2026-07-21). For ONE lot (75) of Nifty weekly options a round-trip fill is dominated by flat
# brokerage (~Rs20/order), plus STT (0.0625% sell premium), exchange txn (~0.035%), GST (18% on
# brokerage+txn), SEBI + stamp. All-in that lands near Rs20-27 per fill ≈ 0.30-0.36 pts at LOT=75;
# we take a deliberately conservative 0.35. It is per-fill, so a 2-leg vertical marked out intraday
# pays it 4x (enter short+long, exit short+long) and a condor pays it 8x. NOTE: this is scale-
# sensitive — flat brokerage means more lots would lower the per-lot points cost; the shadow book
# is 1 lot, so 0.35 is right for THIS ledger and would be pessimistic for a larger book.
FEE_PTS_PER_FILL = 0.35
VERTICAL_FILLS = 4   # short+long entered, short+long exited (intraday mark-out, not expiry)


def _parse_snapshot(rec):
    oc = (((rec.get("response") or {}).get("data") or {}).get("data") or {})
    spot = float(oc.get("last_price") or 0)
    rows = {}
    for k, node in (oc.get("oc") or {}).items():
        strike = float(k)
        rows[strike] = {}
        for side in ("ce", "pe"):
            leg = node.get(side) or {}
            g = leg.get("greeks") or {}
            b = float(leg.get("top_bid_price") or 0)
            a = float(leg.get("top_ask_price") or 0)
            ltp = float(leg.get("last_price") or 0)
            mid = (b + a) / 2 if b > 0 and a > 0 else ltp
            # bid/ask retained for REALISTIC fills — mid-price P&L is a mirage for multi-leg
            # selling (2026-07-16: mid condor +6k flipped to -124k with real fills). A credit
            # spread is entered SELL short @ bid, BUY long @ ask; the P&L must use those.
            rows[strike][side] = {"delta": g.get("delta"), "mid": mid, "bid": b, "ask": a}
    t = rec.get("captured_at", "")[11:19]
    return t, spot, rows


def _pick(rows, side, dtarget):
    best = None; bd = 9
    for strike, sides in rows.items():
        leg = sides.get(side) or {}
        d = leg.get("delta")
        if d is None or leg.get("mid", 0) <= 0:
            continue
        e = abs(abs(d) - dtarget)
        if e < bd:
            bd = e; best = (strike, leg)
    return best if bd <= 0.08 else None


def _credit_spread(entry_rows, exit_rows, side):
    """Short ~0.30 delta, long ~0.15 delta, marked with REALISTIC fills (sell@bid, buy@ask).
    Mid-price P&L is a mirage for multi-leg selling (mid condor +6k → -124k on real fills)."""
    sp = _pick(entry_rows, side, 0.30)
    lp = _pick(entry_rows, side, 0.15)
    if not sp or not lp:
        return None
    sk, sv = sp; lk, lv = lp
    # entry: SELL short @ bid, BUY long @ ask
    if not (sv.get("bid") and lv.get("ask")):
        return None
    credit = sv["bid"] - lv["ask"]
    if credit <= 0:
        return None
    ex_s = (exit_rows.get(sk) or {}).get(side) or {}
    ex_l = (exit_rows.get(lk) or {}).get(side) or {}
    # close: BUY BACK short @ ask, SELL long @ bid
    if not ex_s.get("ask") or ex_l.get("bid") is None:
        return None
    close_cost = ex_s["ask"] - ex_l["bid"]
    fees = VERTICAL_FILLS * FEE_PTS_PER_FILL   # brokerage/STT/exch/GST the bid/ask does NOT include
    pnl = (credit - close_cost - fees) * LOT
    return {"short": sk, "long": lk, "credit_pts": round(credit, 2),
            "close_cost_pts": round(close_cost, 2), "fees_pts": round(fees, 2),
            "pnl_rupees": round(pnl), }


def _debit_spread(entry_rows, exit_rows, side):
    """Directional DEBIT: BUY near-ATM (~0.45 delta) + SELL further OTM (~0.25 delta). side 'pe' =
    bearish put-debit, 'ce' = bullish call-debit. Real fills: buy@ask / sell@bid, reverse on exit."""
    lp = _pick(entry_rows, side, 0.45)   # long leg (buy), near ATM
    sp = _pick(entry_rows, side, 0.25)   # short leg (sell), further OTM
    if not lp or not sp:
        return None
    lk, lv = lp; sk, sv = sp
    if not (lv.get("ask") and sv.get("bid")):
        return None
    debit = lv["ask"] - sv["bid"]
    if debit <= 0:
        return None
    ex_l = (exit_rows.get(lk) or {}).get(side) or {}
    ex_s = (exit_rows.get(sk) or {}).get(side) or {}
    if ex_l.get("bid") is None or ex_s.get("ask") is None:
        return None
    exit_val = ex_l["bid"] - ex_s["ask"]                    # close: sell long @ bid, buy short @ ask
    fees = VERTICAL_FILLS * FEE_PTS_PER_FILL
    pnl = (exit_val - debit - fees) * LOT
    return {"long": lk, "short": sk, "debit_pts": round(debit, 2),
            "exit_val_pts": round(exit_val, 2), "fees_pts": round(fees, 2), "pnl_rupees": round(pnl)}


def _iron_fly(entry_rows, exit_rows, spot_e, wing=200.0):
    """SELL ATM straddle + BUY ~wing-pt OTM wings (defined risk). Real fills."""
    if not entry_rows:
        return None
    atm = min(entry_rows.keys(), key=lambda k: abs(k - spot_e))
    lc_k = min(entry_rows.keys(), key=lambda k: abs(k - (atm + wing)))
    lp_k = min(entry_rows.keys(), key=lambda k: abs(k - (atm - wing)))
    sc = entry_rows[atm].get("ce") or {}; sp = entry_rows[atm].get("pe") or {}
    wc = entry_rows[lc_k].get("ce") or {}; wp = entry_rows[lp_k].get("pe") or {}
    if not (sc.get("bid") and sp.get("bid") and wc.get("ask") and wp.get("ask")):
        return None
    credit = (sc["bid"] + sp["bid"]) - (wc["ask"] + wp["ask"])   # sell straddle@bid, buy wings@ask
    if credit <= 0 or not (lp_k < atm < lc_k):
        return None
    xc = (exit_rows.get(atm) or {}).get("ce") or {}; xp = (exit_rows.get(atm) or {}).get("pe") or {}
    xwc = (exit_rows.get(lc_k) or {}).get("ce") or {}; xwp = (exit_rows.get(lp_k) or {}).get("pe") or {}
    if not (xc.get("ask") and xp.get("ask")) or xwc.get("bid") is None or xwp.get("bid") is None:
        return None
    close_cost = (xc["ask"] + xp["ask"]) - (xwc["bid"] + xwp["bid"])
    fees = 8 * FEE_PTS_PER_FILL                             # 4 legs round-trip
    pnl = (credit - close_cost - fees) * LOT
    return {"short_atm": atm, "wings": [lp_k, lc_k], "credit_pts": round(credit, 2),
            "close_cost_pts": round(close_cost, 2), "fees_pts": round(fees, 2), "pnl_rupees": round(pnl)}


def _short_2leg(entry_rows, exit_rows, *, atm_spot=None, delta=0.15):
    """Naked short: SELL a call + a put. atm_spot set => SHORT STRADDLE (ATM); else SHORT STRANGLE
    (~delta OTM each side). Real fills: sell@bid, buy back@ask. NOTE: naked (undefined risk)."""
    if atm_spot is not None:
        if not entry_rows:
            return None
        ck = pk = min(entry_rows.keys(), key=lambda k: abs(k - atm_spot))
        cv = entry_rows[ck].get("ce") or {}; pv = entry_rows[pk].get("pe") or {}
    else:
        sc = _pick(entry_rows, "ce", delta); spu = _pick(entry_rows, "pe", delta)
        if not sc or not spu:
            return None
        ck, cv = sc; pk, pv = spu
    if not (cv.get("bid") and pv.get("bid")):
        return None
    credit = cv["bid"] + pv["bid"]
    xc = (exit_rows.get(ck) or {}).get("ce") or {}; xp = (exit_rows.get(pk) or {}).get("pe") or {}
    if not (xc.get("ask") and xp.get("ask")):
        return None
    close_cost = xc["ask"] + xp["ask"]
    fees = 4 * FEE_PTS_PER_FILL                             # 2 legs round-trip
    pnl = (credit - close_cost - fees) * LOT
    return {"short_call": ck, "short_put": pk, "credit_pts": round(credit, 2),
            "close_cost_pts": round(close_cost, 2), "fees_pts": round(fees, 2), "pnl_rupees": round(pnl)}


# Precondition fields to stamp on each row — the FEATURES the matrix selects on. Joining these to the
# per-structure P&L is the labelled dataset for learning `preconditions -> which strategy wins`.
_PRECOND_KEYS = [
    "gex_regime", "pin_risk_active", "spot_to_pin_pts", "gamma_concentration", "trend_efficiency_ratio",
    "vol_regime", "option_chain_pressure_state", "range_balance_score", "range_penalty_score",
    "bullish_trend_quality_score", "bearish_trend_quality_score", "put_support_strike",
    "call_resistance_strike", "opening_range_break_state", "accepted_breakout", "inside_opening_range",
    "smart_money_bias", "oi_pressure_bias", "overhead_call_pressure_score", "india_vix",
    "expected_move_pts", "daily_trend", "opening_gap_pct",
]


def _preconditions_for(day: str, entry_time: str) -> dict:
    """The agent's precondition vector nearest the shadow ENTRY time — pulled from the decision log,
    which stamps full metadata every cycle. This is what the day 'looked like' when the trade opened."""
    log = STATE / "intraday_v83_runner.log"
    if not log.exists():
        return {}
    try:
        et = datetime.strptime(entry_time, "%H:%M:%S")
    except ValueError:
        return {}
    best, best_gap = {}, 1e9
    for line in log.read_text(errors="ignore").splitlines():
        if day not in line or '"day_archetype"' not in line:
            continue
        try:
            d = json.loads(line[line.index("{"):])
        except (ValueError, json.JSONDecodeError):
            continue
        t = str(d.get("emitted_at") or "")[11:19]
        try:
            gap = abs((datetime.strptime(t, "%H:%M:%S") - et).total_seconds())
        except ValueError:
            continue
        if gap < best_gap:
            md = d.get("metadata") or {}
            best_gap = gap
            best = {k: md.get(k) for k in _PRECOND_KEYS if k in md}
            best["_decision_time"] = t
    return best


def main() -> int:
    now = datetime.now(IST)
    day = now.strftime("%Y-%m-%d")
    snap_file = STATE / "rolling_option_live" / day / "chain_snapshots.jsonl"
    if not snap_file.exists():
        print(f"[shadow] no capture for {day}; skip")
        return 0
    snaps = []
    for line in snap_file.read_text().splitlines():
        try:
            t, spot, rows = _parse_snapshot(json.loads(line))
        except Exception:
            continue
        if t >= "09:15" and spot > 0 and any((s.get("pe") or {}).get("bid", 0) > 0 for s in rows.values()):
            snaps.append((t, spot, rows))
    if len(snaps) < 2:
        print(f"[shadow] <2 live snapshots for {day}; skip")
        return 0

    entry = next((s for s in snaps if s[0] >= ENTRY_AFTER), snaps[0])
    exit_ = next((s for s in reversed(snaps) if s[0] <= MARK_BY), snaps[-1])
    (te, spot_e, rows_e), (tx, spot_x, rows_x) = entry, exit_
    move_pct = (spot_x - spot_e) / spot_e * 100 if spot_e else 0.0
    realized = "UP" if move_pct >= 0.3 else "DOWN" if move_pct <= -0.3 else "RANGE"

    # ── The full structure menu, each at REAL fills + fees (see helpers) ──────────────────────
    bull_put = _credit_spread(rows_e, rows_x, "pe")   # sell puts — profits up/sideways
    bear_call = _credit_spread(rows_e, rows_x, "ce")  # sell calls — profits down/sideways
    condor = None
    if bull_put and bear_call:
        condor = {"pnl_rupees": bull_put["pnl_rupees"] + bear_call["pnl_rupees"],
                  "credit_pts": round(bull_put["credit_pts"] + bear_call["credit_pts"], 2),
                  "close_cost_pts": round(bull_put["close_cost_pts"] + bear_call["close_cost_pts"], 2),
                  "fees_pts": round(bull_put["fees_pts"] + bear_call["fees_pts"], 2)}
    structures = {
        "bull_put": bull_put,
        "bear_call": bear_call,
        "iron_condor": condor,
        "put_debit": _debit_spread(rows_e, rows_x, "pe"),    # buy puts — bearish
        "call_debit": _debit_spread(rows_e, rows_x, "ce"),   # buy calls — bullish
        "iron_fly": _iron_fly(rows_e, rows_x, spot_e),       # sell ATM straddle + wings (defined)
        "short_strangle": _short_2leg(rows_e, rows_x, delta=0.15),          # naked
        "short_straddle": _short_2leg(rows_e, rows_x, atm_spot=spot_e),     # naked
    }

    # ── The FEATURES the day looked like at entry — the label side is the per-structure P&L above ─
    preconditions = _preconditions_for(day, te)

    record = {
        "date": day, "entry_time": te, "exit_time": tx,
        "spot_entry": round(spot_e, 1), "spot_exit": round(spot_x, 1),
        "day_move_pct": round(move_pct, 2), "realized_regime": realized,
        "snapshots": len(snaps),
        "preconditions": preconditions,          # FEATURES (the matrix selects on these)
        "structures": structures,                # LABELS (what each structure actually made, real fills)
        # legacy top-level keys kept so the existing retrospective still reads them
        "bull_put": bull_put, "bear_call": bear_call, "iron_condor": condor,
    }
    out = STATE / "shadow_book.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(record) + "\n")

    def _p(x): return f"{x['pnl_rupees']:+,}" if x else "n/a"
    parts = " | ".join(f"{k} {_p(v)}" for k, v in structures.items())
    print(f"[shadow] {day} {realized} ({move_pct:+.2f}%) | {parts}")
    print(f"[shadow]   preconditions: gex={preconditions.get('gex_regime')} "
          f"pin={preconditions.get('pin_risk_active')} vol={preconditions.get('vol_regime')} "
          f"eff={preconditions.get('trend_efficiency_ratio')} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
