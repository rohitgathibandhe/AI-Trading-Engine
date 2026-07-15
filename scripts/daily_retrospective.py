#!/usr/bin/env python
"""Daily trade-quality RETROSPECTIVE — "every day is learning; how to be better tomorrow."

Automates the manual audit: for each of today's paper trades it checks entry TIMING (did it chase
an extended move?), STRIKE quality (directional delta punch), P&L and exit; cross-references the
shadow book (what the matched seller trades would have made); tallies errors (rate-limit cooldowns,
NO_DEBIT/DELTA blocks); and emits actionable "tomorrow's focus" flags.

Writes one JSON line to state/daily_retrospective.jsonl and prints a human summary. Scheduled 15:45
IST (after close, shadow book, and eod_learning). Read-only — it never trades or changes config.
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
LOGS = Path(__file__).resolve().parents[1] / "logs"
IST = timezone(timedelta(hours=5, minutes=30))


def _load_price_path(day):
    f = STATE / "rolling_option_live" / day / "chain_snapshots.jsonl"
    path = []
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
                sp = r["response"]["data"]["data"].get("last_price")
                t = r["captured_at"][11:16]
                if sp and t >= "09:15":
                    path.append((t, float(sp)))
            except Exception:
                continue
    return path


def _spot_at(path, hhmm):
    best = None
    for t, s in path:
        if t <= hhmm:
            best = s
    return best


def _today_trades(day):
    out = []
    f = STATE / "intraday_v83_paper_live_trades.jsonl"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if str(d.get("entry_timestamp", "")).startswith(day):
            out.append(d)
    return out


def main() -> int:
    now = datetime.now(IST)
    day = now.strftime("%Y-%m-%d")
    path = _load_price_path(day)
    trades = _today_trades(day)

    # day regime from captured path
    day_ctx = {}
    if path:
        o, last = path[0][1], path[-1][1]
        hi = max(s for _, s in path); lo = min(s for _, s in path)
        mv = (last - o) / o * 100
        day_ctx = {"open": round(o), "close": round(last), "high": round(hi), "low": round(lo),
                   "range_pts": round(hi - lo), "move_pct": round(mv, 2),
                   "regime": "UP" if mv >= 0.3 else "DOWN" if mv <= -0.3 else "RANGE"}

    audits = []
    for t in trades:
        legs = t.get("legs") or []
        et = str(t.get("entry_timestamp", ""))[11:16]
        strat = t.get("strategy", "?")
        net_delta = round(sum((1 if l.get("action") == "BUY" else -1) * float(l.get("delta") or 0) for l in legs), 3)
        flags = []
        # CHASE: a down-trade entered after a big prior-hour drop
        if path and "PUT_DEBIT" in strat and et:
            eh, em = int(et[:2]), int(et[3:5])
            prev = f"{eh-1:02d}:{em:02d}" if eh > 9 else "09:15"
            s_entry = _spot_at(path, et); s_prev = _spot_at(path, prev)
            if s_entry and s_prev:
                hour_move = (s_entry - s_prev) / s_prev * 100
                if hour_move <= -0.45:
                    flags.append(f"CHASE(last-hr {hour_move:+.2f}% — bought an extended drop)")
        # NOTE: net-delta "punch" is NOT a quality signal — measured 2026-07-15 on the dense 7mo:
        # LOW-punch (<0.13) put-debits WIN 73% vs HIGH-punch (>0.18) 41%, corr(netΔ,pnl)=-0.086.
        # The R:R-optimizing selector picking cheaper low-punch spreads is CORRECT. No flag.
        audits.append({"entry": et, "strategy": strat, "net_delta": net_delta,
                       "pnl": t.get("pnl_rupees"), "exit_reason": t.get("exit_reason"), "flags": flags})

    # shadow book today
    shadow = None
    sf = STATE / "shadow_book.jsonl"
    if sf.exists():
        for line in sf.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("date") == day:
                    shadow = r
            except Exception:
                continue

    # errors
    def _count(path_, pat):
        try:
            return sum(1 for l in Path(path_).read_text(errors="ignore").splitlines()[-2000:] if pat in l.lower())
        except Exception:
            return 0
    errors = {"agent_429_cooldown": _count(STATE / "intraday_v83_runner.log", "429") +
                                     _count(STATE / "intraday_v83_runner.log", "cooldown"),
              "no_debit_structure": _count(STATE / "intraday_v83_runner.log", "no_debit_structure"),
              "delta_too_high": _count(STATE / "intraday_v83_runner.log", "delta_too_high")}

    # tomorrow's focus = distinct flags + standing gaps
    focus = sorted({f.split("(")[0] for a in audits for f in a["flags"]})
    if errors["agent_429_cooldown"] > 30:
        focus.append("API_CONTENTION(agent rate-limited — capture competes for 1/3s chain limit)")

    record = {"date": day, "day": day_ctx, "trades": len(trades), "trade_audits": audits,
              "shadow_book": {k: shadow.get(k) for k in ("realized_regime", "bull_put", "bear_call", "iron_condor")} if shadow else None,
              "errors": errors, "tomorrow_focus": focus}
    (STATE / "daily_retrospective.jsonl").open("a").write(json.dumps(record) + "\n")

    # human summary
    print(f"\n===== DAILY RETROSPECTIVE {day} =====")
    if day_ctx:
        print(f"Market: {day_ctx['regime']} {day_ctx['move_pct']:+.2f}% (O{day_ctx['open']} H{day_ctx['high']} L{day_ctx['low']} C{day_ctx['close']}, range {day_ctx['range_pts']}pt)")
    print(f"Trades taken: {len(trades)}")
    for a in audits:
        pnl = a["pnl"]; pstr = f"Rs{pnl:+,.0f}" if isinstance(pnl, (int, float)) else "open"
        print(f"  {a['entry']} {a['strategy']} netΔ{a['net_delta']:+.2f} {pstr} {a['exit_reason'] or ''}")
        for f in a["flags"]:
            print(f"      ⚠ {f}")
    if shadow:
        def p(x): return f"Rs{x['pnl_rupees']:+,}" if x else "n/a"
        print(f"Shadow book ({shadow.get('realized_regime')}): bull_put {p(shadow.get('bull_put'))} | "
              f"bear_call {p(shadow.get('bear_call'))} | condor {p(shadow.get('iron_condor'))}")
    print(f"Errors: 429/cooldown={errors['agent_429_cooldown']} no_debit={errors['no_debit_structure']} delta_high={errors['delta_too_high']}")
    print(f"TOMORROW'S FOCUS: {focus if focus else 'none flagged'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
