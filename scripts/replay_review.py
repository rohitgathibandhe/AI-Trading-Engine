#!/usr/bin/env python3
"""Standard post-replay review — runs after EVERY replay (owner, 2026-10-19). For a live_path_replay artifact
(summary.json trades + the agent's decision log) on REAL recorded-chain days it reports:

  1. WINNERS  — entry quality (how close to the best entry), exit quality (% of the available move captured,
                points left on the table), the exit rule that fired.
  2. LOSERS   — root cause per trade + the best trade actually available that day (real option prices).
  3. MISSED   — days with no trade (or only losses) where >= MIN_OPP points were available: what blocked it
                (the agent's own block reasons, by time) and what could have been traded.
  4. SLICES   — net P&L by DTE, day type, time of day, structure/setup; option-vs-spot responsiveness; with and
                without replay data-gap days.
  5. RANKED   — recurring causes by rupees.

  python scripts/replay_review.py --artifact <livereplay_dir> [--cost 150] [--out report.txt]
All 'available'/'best' numbers are hindsight on real bid/ask (buy at ask, sell at bid) — that is the point of the
review; the 'context at entry' columns use only data known at that moment.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import agent_skill_report as asr  # noqa: E402

RO = ROOT / "data_engine" / "market_ai" / "state" / "rolling_option_live"
DIR = {"CALL_DEBIT_SPREAD": 1, "BULL_PUT_CREDIT_SPREAD": 1, "PUT_DEBIT_SPREAD": -1, "BEAR_CALL_CREDIT_SPREAD": -1}
GAP_EXITS = {"STALE_OVERNIGHT_EXIT", "EOD_DATA_LOSS_FORCE_EXIT"}
MIN_OPP = 40.0


def _q(j, k, side, f):
    x = ((j["response"]["data"]["data"]["oc"].get(f"{k:.6f}") or {}).get(side) or {})
    return x.get(f) or x.get("last_price") or 0


def day_map(day: str) -> dict | None:
    """Real-chain hindsight map for a day: best ATM call / put buy and best straddle-sell window, spot path."""
    f = RO / day / "chain_snapshots.jsonl"
    if not f.exists():
        return None
    snaps = []
    for ln in open(f):
        try:
            j = json.loads(ln)
            t = datetime.fromisoformat(j["captured_at"]).replace(tzinfo=None)
            sp = j["response"]["data"]["data"]["last_price"]
            if "09:30" <= t.strftime("%H:%M") <= "15:15" and sp > 1000:
                snaps.append((t, j, sp))
        except Exception:
            continue
    if len(snaps) < 15:
        return None
    exp = snaps[0][1].get("expiry")
    snaps = [s for s in snaps if s[1].get("expiry") == exp]
    med = sorted(s[2] for s in snaps)[len(snaps) // 2]
    snaps = [s for s in snaps if abs(s[2] - med) < 800]           # drop corrupt snapshots
    best = {"CE": (0.0, None, None), "PE": (0.0, None, None)}
    for i, (ti, ji, si) in enumerate(snaps):
        k = round(si / 50) * 50
        for side in ("ce", "pe"):
            buy = _q(ji, k, side, "top_ask_price")
            if buy <= 0:
                continue
            for tk, jk, _ in snaps[i + 1:]:
                g = _q(jk, k, side, "top_bid_price") - buy
                if g > best[side.upper()][0]:
                    best[side.upper()] = (g, ti, tk)
    spots = [s[2] for s in snaps]
    return {"dte": (datetime.fromisoformat(exp).date() - snaps[0][0].date()).days, "best": best,
            "range": max(spots) - min(spots), "move": spots[-1] - spots[0], "snaps": snaps}


def day_type(day: str) -> str:
    bars = [b for b in asr._candles(day) if b["t"].time() < time(15, 15)]
    if len(bars) < 40:
        return "?"
    orb = [b for b in bars if b["t"].time() < time(9, 30)]
    oh, ol = max(b["h"] for b in orb), min(b["l"] for b in orb)
    post = [b for b in bars if b["t"].time() >= time(9, 30)]
    up, dn = any(b["c"] > oh for b in post), any(b["c"] < ol for b in post)
    mv = bars[-1]["c"] - bars[0]["o"]
    rng = max(b["h"] for b in bars) - min(b["l"] for b in bars)
    if up and dn:
        return "RANGE (two-sided)"
    if abs(mv) >= 120 and abs(mv) / max(rng, 1) >= 0.6:
        return "TREND " + ("UP" if mv > 0 else "DOWN")
    return "MIXED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--cost", type=float, default=150.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    art = Path(a.artifact)
    trades = json.load(open(art / "summary.json"))["trades"]
    dec_path = art / "repo" / "data_engine" / "market_ai" / "state" / "intraday_v83_paper_live_validation_decisions.jsonl"
    decisions = [json.loads(x) for x in open(dec_path) if x.strip().startswith("{")] if dec_path.exists() else []
    out: list[str] = []
    P = out.append
    ent = {str(t.get("entry_timestamp") or t["timestamp"]): t for t in trades if t["event"] == "PAPER_ENTRY"}
    by_day = defaultdict(list)
    for t in trades:
        if t["event"] != "PAPER_EXIT":
            continue
        e = ent.get(str(t.get("entry_timestamp"))) or {}
        d = str(e.get("timestamp") or t["timestamp"])[:10]
        by_day[d].append((e, t))
    replay_days = sorted({str(x.get("timestamp", ""))[:10] for x in decisions} | set(by_day))
    maps = {d: day_map(d) for d in replay_days}
    types = {d: day_type(d) for d in replay_days}

    # ---------- 1. winners ----------
    P("1. WINNING TRADES — entry & exit quality (real option prices, hindsight)")
    cap_rows = []
    for d, lst in sorted(by_day.items()):
        m = maps.get(d)
        for e, t in lst:
            pnl = float(t.get("realized_paper_pnl") or 0)
            if pnl <= 0 or t.get("exit_reason") in GAP_EXITS:
                continue
            dr = DIR.get(e.get("strategy"), 0)
            bars = [b for b in asr._candles(d) if b["t"].time() < time(15, 15)]
            tin = datetime.fromisoformat(str(e["timestamp"])[:19])
            tout = datetime.fromisoformat(str(t.get("exit_timestamp") or t["timestamp"])[:19])
            after = [b for b in bars if b["t"] >= tin]
            held = [b for b in after if b["t"] < tout]
            if not after or not dr:
                continue
            spot = float(e.get("spot_at_entry") or after[0]["o"])
            avail = max(((b["h"] - spot) if dr > 0 else (spot - b["l"])) for b in after)
            took = ((held[-1]["c"] if held else spot) - spot) * dr
            before = [b for b in bars if b["t"] < tin]
            best_entry = (min(b["l"] for b in before) if dr > 0 else max(b["h"] for b in before)) if before else spot
            late = (spot - best_entry) * dr
            cap_rows.append((d, avail, took, late, pnl))
            P(f"  {d} {tin:%H:%M}->{tout:%H:%M} {e.get('strategy'):24s} Rs {pnl:>7,.0f} | move available {avail:4.0f} pts, captured "
              f"{took:4.0f} ({100*took/max(avail,1):3.0f}%), entered {late:+4.0f} pts after the day's best entry | exit {t.get('exit_reason')}")
    if cap_rows:
        A = sum(r[1] for r in cap_rows); T = sum(max(r[2], 0) for r in cap_rows)
        P(f"  => captured {T:.0f} of {A:.0f} available pts ({100*T/max(A,1):.0f}%); median late-entry "
          f"{sorted(r[3] for r in cap_rows)[len(cap_rows)//2]:.0f} pts")

    # ---------- 2. losers ----------
    P("\n2. LOSING TRADES — root cause + what the day actually offered")
    causes = defaultdict(list)
    for d, lst in sorted(by_day.items()):
        m = maps.get(d)
        for e, t in lst:
            pnl = float(t.get("realized_paper_pnl") or 0)
            if pnl >= 0:
                continue
            why = t.get("exit_reason")
            pb = str(e.get("playbook") or "")
            st = str(e.get("strategy") or "")
            dr = DIR.get(st, 0)
            if why in GAP_EXITS:
                cause = "REPLAY_DATA_GAP (not an agent decision)"
            elif m and dr and ((m["move"] > 30 and dr < 0) or (m["move"] < -30 and dr > 0)):
                cause = f"WRONG SIDE ({'short' if dr < 0 else 'long'} on a {m['move']:+.0f} day) via {pb}"
            elif "CHOP" in pb or "ZONE" in pb:
                cause = f"NO-EDGE SETUP ({pb} {st})"
            elif "DEBIT" in st and why == "DEBIT_STOP":
                cause = "DEBIT STOPPED (setup failed / oversized)"
            elif why == "SELLER_STRIKE_BUFFER":
                cause = "CREDIT TESTED AT THE SHORT STRIKE"
            else:
                cause = f"OTHER ({why})"
            causes[cause].append(pnl)
            alt = ""
            if m:
                bc, bp = m["best"]["CE"], m["best"]["PE"]
                pick = ("CALL", bc) if bc[0] >= bp[0] else ("PUT", bp)
                if pick[1][1] is not None:
                    alt = f" | day offered: {pick[0]} buy {pick[1][1]:%H:%M}->{pick[1][2]:%H:%M} +{pick[1][0]:.0f} pts"
            P(f"  {d} [{types.get(d)} DTE {m['dte'] if m else '?'}] {str(e.get('timestamp',''))[11:16]} {st:24s} {pb:22s} "
              f"Rs {pnl:>7,.0f} exit {why} -> {cause}{alt}")

    # ---------- 3. missed days ----------
    P(f"\n3. MISSED / FLAT DAYS — >= {MIN_OPP:.0f} pts available, agent made nothing")
    dec_by_day = defaultdict(list)
    for x in decisions:
        dec_by_day[str(x.get("timestamp", ""))[:10]].append(x)
    for d in replay_days:
        m = maps.get(d)
        made = sum(float(t.get("realized_paper_pnl") or 0) for _, t in by_day.get(d, []))
        if not m or made > 0:
            continue
        bc, bp = m["best"]["CE"], m["best"]["PE"]
        best = max(bc[0], bp[0])
        if best < MIN_OPP:
            continue
        side, b = ("CALL", bc) if bc[0] >= bp[0] else ("PUT", bp)
        wanted = [x for x in dec_by_day.get(d, []) if x.get("raw_action") == "TRADE"]
        blocks = Counter(str(x.get("block_reason"))[:40] for x in wanted if x.get("block_reason") not in (None, "NONE"))
        stand = Counter(str(x.get("block_reason"))[:30] for x in dec_by_day.get(d, []) if x.get("raw_action") != "TRADE")
        P(f"  {d} [{types.get(d)} DTE {m['dte']}] made Rs {made:,.0f} | best {side} buy {b[1]:%H:%M}->{b[2]:%H:%M} +{b[0]:.0f} pts | "
          f"agent wanted to trade {len(wanted)}x, blocked by {dict(blocks.most_common(3)) or '-'}; otherwise {dict(stand.most_common(2))}")

    # ---------- 4. slices ----------
    P("\n4. SLICES (net of Rs %.0f per trade; data-gap exits excluded)" % a.cost)
    sl = defaultdict(lambda: [0, 0.0])
    for d, lst in by_day.items():
        m = maps.get(d)
        for e, t in lst:
            if t.get("exit_reason") in GAP_EXITS:
                continue
            net = float(t.get("realized_paper_pnl") or 0) - a.cost
            hhmm = str(e.get("timestamp", ""))[11:16]
            keys = [("DTE", str(m["dte"]) if m else "?"), ("day type", types.get(d, "?")),
                    ("entry time", "<10:30" if hhmm < "10:30" else "10:30-12:30" if hhmm < "12:30" else ">=12:30"),
                    ("structure", str(e.get("strategy"))), ("setup", str(e.get("playbook")))]
            for k in keys:
                sl[k][0] += 1
                sl[k][1] += net
    for dim in ("DTE", "day type", "entry time", "structure", "setup"):
        P(f"  by {dim}:")
        for (k, v), (n, s) in sorted(((k, v) for k, v in sl.items() if k[0] == dim), key=lambda kv: kv[1][1]):
            P(f"     {v:28s} n={n:3d} net Rs {s:>9,.0f}")
    tot = sum(float(t.get("realized_paper_pnl") or 0) for lst in by_day.values() for _, t in lst)
    gap = sum(float(t.get("realized_paper_pnl") or 0) for lst in by_day.values() for _, t in lst if t.get("exit_reason") in GAP_EXITS)
    n = sum(len(v) for v in by_day.values())
    P(f"  TOTAL gross Rs {tot:,.0f} | net of costs Rs {tot - n*a.cost:,.0f} | of which replay data-gap exits Rs {gap:,.0f}")

    # ---------- 5. ranked ----------
    P("\n5. RANKED LOSS CAUSES (Rs)")
    for k, v in sorted(causes.items(), key=lambda kv: sum(kv[1])):
        P(f"  {k:60s} {len(v):3d} trades  Rs {sum(v):>9,.0f}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
