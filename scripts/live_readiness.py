#!/usr/bin/env python3
"""LIVE READINESS — one place that answers "how far are we from a 1-lot live start, and why?"

Combines the objective bars the go-live plan (2026-10-09) set:
  1. EVIDENCE  — a strategy clears the promotion gate on the agent's own forward paper trades
                 (scripts/promotion_gate.py: >=20 trades, avg>0, structure-aware win floor, PF>=1.3,
                 total >= 3x worst trade, no drift).
  2. STABILITY — OPS_STREAK_TARGET consecutive clean trading days: the watchdog was awake for most of
                 the session (Mac not asleep) and logged no sleep gap / stall-restart / dead agent /
                 expired token. Live money needs a boring machine, not just a good strategy.
  3. LOCKS     — reported, never changed: paper executor, mode, live_arm, empty whitelist. Flipping
                 them is the owner's deliberate action (project memory project_go_live_gate).
Also reports maturity score + token/power as context, and an ETA from the recent trade rate.

Decides nothing. Writes state/live_readiness.json (served by the dashboard at /api/live_readiness).
  python scripts/live_readiness.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
STATE = ROOT / "data_engine" / "market_ai" / "state"
WATCHDOG_LOG = ROOT / "logs" / "v83_watchdog.log"
OUT = STATE / "live_readiness.json"

OPS_STREAK_TARGET = 10          # two trading weeks
EXPECTED_TICKS = 185            # watchdog ticks every 2 min, 09:20-15:29
MIN_TICK_SHARE = 0.75           # below this the Mac was asleep/offline for a big part of the session
INCIDENT_RE = re.compile(r"SLEEP GAP|Log stale|Agent dead|TOKEN EXPIRED")
LEAD_STRATEGY = "PUT_DEBIT_SPREAD"   # the validated intraday engine — first candidate for 1-lot live


def _trading_days(lookback: int = 40) -> set[str] | None:
    """NSE trading days from Dhan daily Nifty bars (cached per day in state/). None if unavailable —
    then every weekday counts (a holiday with the Mac off would read as a not-clean day)."""
    cache = STATE / "trading_days_cache.json"
    today = date.today().isoformat()
    try:
        c = json.loads(cache.read_text())
        if c.get("fetched") == today:
            return set(c["days"])
    except Exception:
        pass
    try:
        sys.path.insert(0, str(ROOT / "data_engine"))
        from market_ai.dhan_wrapper import DhanWrapper
        cr = json.loads((STATE / "creds.json").read_text())
        dw = DhanWrapper(dhan_client_id=cr["client_id"], access_token=cr["access_token"])
        bars = dw.get_daily_candles(13, "IDX_I", from_date=(date.today() - timedelta(days=lookback + 10)).isoformat(),
                                    to_date=today)
        days = sorted({str(b["timestamp"])[:10] for b in bars or []})
        if not days:
            return None
        cache.write_text(json.dumps({"fetched": today, "days": days}))
        return set(days)
    except Exception:
        return None


def _ops_days(lookback: int = 40) -> list[dict]:
    """Per weekday: watchdog market-hours ticks + incidents, oldest first. Today counts only after close."""
    ticks: dict[str, int] = {}
    incidents: dict[str, list[str]] = {}
    try:
        lines = WATCHDOG_LOG.read_text(errors="ignore").splitlines()
    except OSError:
        lines = []
    since = (date.today() - timedelta(days=lookback)).isoformat()
    for ln in lines:
        if not ln.startswith("[") or ln[1:11] < since:
            continue
        d, hm = ln[1:11], ln[12:17]
        if not ("09:20" <= hm <= "15:29"):
            continue
        if "Watchdog tick" in ln:
            ticks[d] = ticks.get(d, 0) + 1
        m = INCIDENT_RE.search(ln)
        if m:
            incidents.setdefault(d, []).append(f"{hm} {m.group(0)}")
    now = datetime.now()
    tdays = _trading_days(lookback)
    out = []
    for d in sorted(ticks):
        dd = date.fromisoformat(d)
        if dd.weekday() >= 5 or (dd == now.date() and now.strftime("%H:%M") < "15:30"):
            continue
        if tdays is not None and d not in tdays and d < max(tdays):
            continue                    # exchange holiday (e.g. 10-02) — not a session to be awake for
        share = ticks[d] / EXPECTED_TICKS
        inc = incidents.get(d, [])
        problems = list(dict.fromkeys(x.split(" ", 1)[1] for x in inc))
        if share < MIN_TICK_SHARE:
            problems.insert(0, f"awake only {share:.0%} of session")
        out.append({"date": d, "ticks": ticks[d], "awake_share": round(share, 2),
                    "clean": not problems, "problems": problems})
    return out


def _streak(days: list[dict]) -> int:
    n = 0
    for d in reversed(days):
        if not d["clean"]:
            break
        n += 1
    return n


def _locks() -> dict:
    def _rd(p):
        try:
            return json.loads((STATE / p).read_text())
        except Exception:
            return {}
    rl, rt = _rd("intraday_v83_run_live_config.json"), _rd("intraday_v83_runtime_config.json")
    # NOTE: live_enabled_strategies is NOT a real-money lock — evaluate_entry_gate applies it in PAPER too
    # (emptying it would stop paper trading). The per-strategy real-money governor is the promotion
    # gate's eligible list, read by promotion.real_money_eligible() before any MICRO_LIVE order.
    promo = _rd("promotion_state.json")
    locks = {
        "paper_executor": "PaperOnly" in str(rl.get("executor_class", "")),
        "paper_mode": str(rl.get("trade_mode", "")).lower() == "paper" and rt.get("mode") != "MICRO_LIVE",
        "live_arm_off": rt.get("live_arm") is False,
        "no_strategy_promoted": not (promo.get("eligible_for_live") or []),
    }
    locks["all_engaged"] = all(locks.values())
    return locks


def _maturity() -> str | None:
    try:
        txt = (STATE / "maturity_scorecard.log").read_text(errors="ignore")
        m = re.findall(r"SCORE: (\d+/\d+)", txt)
        return m[-1] if m else None
    except OSError:
        return None


def _context() -> dict:
    ctx: dict = {}
    try:
        import base64
        tok = json.loads((STATE / "creds.json").read_text()).get("access_token", "")
        exp = json.loads(base64.urlsafe_b64decode(tok.split(".")[1] + "=="))["exp"]
        ctx["token_hours_left"] = round((exp - datetime.now().timestamp()) / 3600, 1)
    except Exception:
        ctx["token_hours_left"] = None
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        ctx["on_ac_power"] = "AC Power" in out
    except Exception:
        ctx["on_ac_power"] = None
    return ctx


def evaluate() -> dict:
    import promotion_gate as pg
    report = {r["strategy"]: r for r in pg.evaluate()}
    lead = report.get(LEAD_STRATEGY, {})
    eligible = [s for s, r in report.items() if r.get("status") == "ELIGIBLE_FOR_LIVE"]
    n = int(lead.get("trades") or 0)
    trades_needed = max(0, pg.MIN_DAYS - n)
    worst = lead.get("worst") or 0
    profit_needed = max(0.0, pg.MIN_RET_TAIL * abs(min(worst, 0)) - float(lead.get("total") or 0))
    # trade rate over the last 21 days -> ETA for the missing trades
    recent = [t for t, _ in pg._paper_pnls().get(LEAD_STRATEGY, [])
              if t[:10] >= (date.today() - timedelta(days=21)).isoformat()]
    per_week = len(recent) / 3.0
    weeks_trades = (trades_needed / per_week) if per_week > 0 else None

    days = _ops_days()
    streak = _streak(days)
    ops_needed = max(0, OPS_STREAK_TARGET - streak)
    weeks_ops = ops_needed / 5.0
    weeks = max(weeks_trades if weeks_trades is not None else 99, weeks_ops)
    # the skill/overall bar needs MIN_DAYS trades across ALL structures — usually sooner than one structure
    _all_recent = [t for v in pg._paper_pnls().values() for t, _ in v
                   if t[:10] >= (date.today() - timedelta(days=21)).isoformat()]
    _need_all = max(0, pg.MIN_DAYS - sum(len(v) for v in pg._paper_pnls().values()))
    _wk_all = (_need_all / (len(_all_recent) / 3.0)) if _all_recent else 99
    weeks = max(_wk_all, weeks_ops)
    eta = (date.today() + timedelta(weeks=weeks)).isoformat() if weeks < 99 else None

    # SKILL is the deciding evidence (user, 2026-10-10): judge the TRADER across all trades and day
    # types — read accuracy, entry, exit — plus an overall profitable forward record, not one structure.
    import agent_skill_report as sk
    skill = sk.evaluate(pg.FORWARD_EPOCH)
    allp = [p for v in pg._paper_pnls().values() for _, p in v]
    gw, gl = sum(x for x in allp if x > 0), -sum(x for x in allp if x < 0)
    overall = {"trades": len(allp), "net": round(sum(allp)), "profit_factor": round(gw / gl, 2) if gl > 0 else None,
               "ok": len(allp) >= pg.MIN_DAYS and sum(allp) > 0 and (gl == 0 or gw / gl >= pg.MIN_PROFIT_FACTOR)}
    evidence_ok = bool(skill["skill_ok"] and overall["ok"])
    stability_ok = streak >= OPS_STREAK_TARGET
    locks = _locks()
    if evidence_ok and stability_ok:
        verdict = "READY_FOR_OWNER_DECISION"   # bars cleared — flipping the locks is the owner's call
    else:
        verdict = "NOT_READY"
    res = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "verdict": verdict,
        "eta_date": eta,
        "skill": {"ok": skill["skill_ok"], "checks": skill["checks"], "overall": skill["overall"],
                  "by_day_type": skill["by_day_type"], "reads": skill["reads"]},
        "forward_overall": overall,
        "evidence": {
            "ok": evidence_ok, "eligible": eligible, "lead_strategy": LEAD_STRATEGY,
            "lead_status": lead.get("status"), "trades": n, "trades_target": pg.MIN_DAYS,
            "trades_needed": trades_needed, "total": lead.get("total"), "win_rate": lead.get("win_rate"),
            "profit_factor": lead.get("profit_factor"), "worst": worst,
            "profit_needed_for_tail_bar": round(profit_needed), "checks": lead.get("checks"),
            "trades_per_week_recent": round(per_week, 1),
            "since": pg.FORWARD_EPOCH,
        },
        "stability": {"ok": stability_ok, "clean_streak": streak, "target": OPS_STREAK_TARGET,
                      "recent_days": days[-10:]},
        "locks": locks,
        "maturity_score": _maturity(),
        "context": _context(),
    }
    OUT.write_text(json.dumps(res, indent=2))
    return res


def main() -> int:
    r = evaluate()
    e, s = r["evidence"], r["stability"]
    print(f"LIVE READINESS — {r['verdict']}   (ETA ~{r['eta_date'] or 'unknown — no recent trade rate'})")
    sk_, fo = r["skill"], r["forward_overall"]
    print(f"\n1. SKILL     {'PASS' if sk_['ok'] else 'not yet'} — all trades since {e['since']}: "
          f"{fo['trades']}/20 trades, net Rs {fo['net']:,}, PF {fo['profit_factor']}")
    for k, c in sk_["checks"].items():
        print(f"   [{'PASS' if c['ok'] else 'FAIL'}] {k:13} {c['value']} vs bar {c['bar']}"
              + (f"  weak on: {', '.join(c['weak_day_types'])}" if c['weak_day_types'] else ""))
    print(f"\n   per-strategy money switch (promotion gate) — {e['lead_strategy']} {e['trades']}/{e['trades_target']} "
          f"trades since {e['since']}, total Rs {e['total']}, win {e['win_rate']}%, PF {e['profit_factor']}, "
          f"worst Rs {e['worst']}")
    print(f"   needs {e['trades_needed']} more trades (~{e['trades_per_week_recent']}/week recently) and "
          f"Rs {e['profit_needed_for_tail_bar']:,} more profit to clear 'total >= 3x worst trade'")
    print(f"\n2. STABILITY {'PASS' if s['ok'] else 'not yet'} — clean trading-day streak {s['clean_streak']}/{s['target']}")
    for d in s["recent_days"]:
        print(f"   {d['date']}  {'clean' if d['clean'] else 'X ' + '; '.join(d['problems'])}")
    lk = r["locks"]
    print(f"\n3. LOCKS     {'all engaged (paper only)' if lk['all_engaged'] else 'NOT all engaged: ' + str(lk)}"
          f" — flipping them is your decision")
    c = r["context"]
    print(f"\n   maturity {r['maturity_score']} | token {c['token_hours_left']}h left | "
          f"AC power {'yes' if c['on_ac_power'] else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
