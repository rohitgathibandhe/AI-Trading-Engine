#!/usr/bin/env python
"""SELF-HEAL — proactively detect and FIX operational faults that stop the agent from trading, without
waiting to be asked. Runs every ~5 min during market hours (launchd).

Scope is deliberately drawn at what is SAFE to auto-fix:

  AUTO-FIX (this script):  operational / infrastructure faults — the class that actually caused the
                           0-trade days: agent hung (not emitting decisions), option chain
                           unavailable well after the open, quote count stuck at 0. These are
                           deterministic and reversible, so the loop restarts the agent itself
                           (rate-limited) and escalates to a phone alert if a restart can't fix it
                           (an expired DHAN token needs a human to refresh — the one thing this
                           cannot do).

  NOT auto-fixed:          decision-logic defects (a too-strict gate, a whipsaw exit). Auto-rewriting
                           the trading brain unattended is how one bad day blows an account. Those are
                           detected + alerted by daily_self_diagnosis.py and fixed through the
                           shadow->promotion validation path with a human gate — by design.

Health is read from the latest runner-log decision. Remediation ladder (state in self_heal_state.json):
  unhealthy x2 in a row + cooldown ok  -> kickstart-restart the agent, alert
  still unhealthy after 2 restarts     -> escalate: alert "manual token refresh likely needed", stop
  healthy                              -> reset counters

Run:  python scripts/self_heal.py            (one-shot; safe to run any time — no-op outside market hours)
      python scripts/self_heal.py --dry-run  (report only, never restart/alert)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
CREDS = STATE / "creds.json"
RUNNER_LOG = STATE / "intraday_v83_runner.log"
HEAL_STATE = STATE / "self_heal_state.json"
HEAL_LOG = STATE / "self_heal.jsonl"
LABEL = "com.algoagent.intraday_v83"

# Market session (IST; host clock is IST). Grace to 09:35 so the normal MIN_5M_CANDLES_NOT_READY
# warm-up right after the 09:15 open is never mistaken for a fault.
OPEN_HHMM = (9, 15)
CHAIN_GRACE_HHMM = (9, 35)
CLOSE_HHMM = (15, 30)
STALE_DECISION_SECS = 240          # no fresh decision in 4 min during market hours = hung
RESTART_COOLDOWN_SECS = 12 * 60    # don't restart more than once per 12 min
MAX_RESTARTS_BEFORE_ESCALATE = 2   # after this many auto-restarts still unhealthy -> human alert


def _now() -> datetime:
    return datetime.now()


def _hhmm(dt: datetime) -> tuple[int, int]:
    return (dt.hour, dt.minute)


def _is_market_hours(dt: datetime) -> bool:
    if dt.weekday() >= 5:          # Sat/Sun
        return False
    return OPEN_HHMM <= _hhmm(dt) <= CLOSE_HHMM


def _tail_lines(path: Path, n_bytes: int = 200_000) -> list[str]:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            data = f.read().decode("utf-8", errors="ignore")
        return data.splitlines()
    except OSError:
        return []


# A decision is a DATA-FAILURE only if it says so explicitly. A NORMAL strategic decision
# (BREAKOUT_DOWN, ZONE_DEMAND, RANGE_WIDE, ...) carries NO data_readiness block at all — the agent
# only emits data_readiness on the failure path. So "no option_chain_available field" means the chain
# WAS available, not that it was missing. Reading a healthy decision as unhealthy is exactly the
# false-positive that would restart a working agent, so classification is failure-explicit.
_DATA_FAIL_MARKERS = ("unavailable", "cannot be empty", "min_5m", "insufficient_data",
                      "snapshot is empty", "no option chain")


def _recent_decisions(n: int = 14) -> list[dict]:
    out: list[dict] = []
    for line in reversed(_tail_lines(RUNNER_LOG)):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "emitted_at" in j or (j.get("metadata") or {}).get("data_readiness"):
            out.append(j)
            if len(out) >= n:
                break
    out.reverse()
    return out


def _is_data_failure(j: dict) -> bool:
    dr = (j.get("metadata") or {}).get("data_readiness") or {}
    rats = j.get("rationale") or []
    rat0 = (rats[0] if rats else "") or ""
    return (any(m in rat0.lower() for m in _DATA_FAIL_MARKERS)
            or dr.get("snapshot_status") in ("INSUFFICIENT_DATA", "UNAVAILABLE"))


def _emitted_of(j: dict):
    dr = (j.get("metadata") or {}).get("data_readiness") or {}
    return j.get("emitted_at") or dr.get("timestamp")


def _is_rate_limited() -> bool:
    """A CHAIN_BLACKOUT caused by a 429 rate-limit must NOT be 'healed' by restarting — each restart
    fires a burst of chain requests and DEEPENS the throttle, turning a transient 429 into a long
    outage (2026-09-22: 2 restarts spiralled a rate-limit into a 15-min blackout during an open,
    winning trade). On a 429 the right action is to BACK OFF and let it cool. Detect it from the
    recent runner log."""
    tail = "\n".join(_tail_lines(RUNNER_LOG, 40_000)).lower()
    return ("429" in tail or "too many requests" in tail or '"805"' in tail
            or "may result in the user being blocked" in tail)


def _data_fault_hint() -> str:
    """Scan the recent runner log for the ACTUAL cause of a data blackout so the alert names the right
    fix. A lapsed Dhan DATA API subscription (DH-902 / 'not subscribed to Data APIs') is NOT a token
    problem — the trading token still authenticates, so 'refresh the token' sends the user down the
    wrong path (this cost 2026-08-13 and 08-14)."""
    tail = "\n".join(_tail_lines(RUNNER_LOG, 120_000)).lower()
    if "429" in tail or "too many requests" in tail or "may result in the user being blocked" in tail:
        return ("Dhan is RATE-LIMITING the chain (HTTP 429). The token is fine — do NOT restart "
                "(restarts burst more requests and worsen it). Let polling idle for a few minutes to cool.")
    if "dh-902" in tail or "not subscribed to data" in tail or "subscribe to data api" in tail:
        return ("Dhan DATA API SUBSCRIPTION has lapsed (DH-902 / HTTP 451). The trading token is fine — "
                "refreshing it will NOT help. Renew the Dhan 'Data APIs' plan to restore live data.")
    if "dh-901" in tail or "invalid token" in tail or "token expired" in tail or "unauthorized" in tail:
        return "DHAN access token looks expired/invalid — refresh the token in state/creds.json."
    return "No live data (candles/chain). Check the Dhan data feed: token validity AND Data API subscription."


def _process_alive() -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", "intraday_defined_risk.cli run_live"],
                             capture_output=True, text=True, timeout=10)
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def assess(now: datetime) -> dict:
    """Return {healthy: bool, reason: str, detail: {...}}. Only meaningful during market hours.

    Health is judged over a WINDOW of recent decisions, not one line: the live chain is intermittent
    (the odd 'snapshot cannot be empty' cycle is normal), so a single failed cycle must NOT be read as
    a fault. The agent is healthy as long as it is (a) emitting fresh decisions and (b) producing at
    least one real strategic analysis in the recent window. It is unhealthy only on a TRUE blackout —
    every recent cycle is a data-failure — or when it stops emitting at all (hung)."""
    alive = _process_alive()
    detail: dict = {"process_alive": alive}
    if not alive:
        return {"healthy": False, "reason": "PROCESS_DEAD", "detail": detail}

    decs = _recent_decisions(14)
    if not decs:
        return {"healthy": False, "reason": "NO_DECISIONS_IN_LOG", "detail": detail}

    real = sum(1 for d in decs if not _is_data_failure(d))
    fail = len(decs) - real
    newest_em = _emitted_of(decs[-1])
    detail.update({"emitted_at": newest_em, "real_analyses": real, "data_fails": fail,
                   "window": len(decs)})

    # Freshness — a live agent emits a decision every ~30s. Prolonged silence = hung.
    if newest_em:
        try:
            age = (now - datetime.fromisoformat(str(newest_em)).replace(tzinfo=None)).total_seconds()
            detail["decision_age_secs"] = round(age)
            if age > STALE_DECISION_SECS:
                return {"healthy": False, "reason": "STALE_DECISIONS", "detail": detail}
        except Exception:  # noqa: BLE001
            pass

    # Data health (past warm-up grace): a TRUE blackout = zero real analyses in the whole window.
    if _hhmm(now) >= CHAIN_GRACE_HHMM and real == 0:
        return {"healthy": False, "reason": "CHAIN_BLACKOUT", "detail": detail}

    return {"healthy": True, "reason": "OK", "detail": detail}


def _load_state() -> dict:
    try:
        return json.loads(HEAL_STATE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_state(s: dict) -> None:
    try:
        HEAL_STATE.write_text(json.dumps(s, indent=2))
    except OSError:
        pass


def _log(record: dict) -> None:
    try:
        with HEAL_LOG.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass
    print(json.dumps(record))


def _alert(text: str) -> bool:
    try:
        creds = json.loads(CREDS.read_text()) if CREDS.exists() else {}
    except Exception:  # noqa: BLE001
        creds = {}
    bot = str(creds.get("telegram_bot_token") or "").strip()
    chat = str(creds.get("telegram_chat_id") or "").strip()
    if not (bot and chat):
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{bot}/sendMessage",
                          json={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=15)
        return r.status_code == 200
    except Exception as e:  # noqa: BLE001
        print(f"[self-heal] telegram failed: {str(e)[:160]}", file=sys.stderr)
        return False


def _restart_agent() -> bool:
    try:
        uid = os.getuid()
        subprocess.run(["launchctl", "kickstart", "-k", f"gui/{uid}/{LABEL}"],
                       capture_output=True, text=True, timeout=30)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[self-heal] restart failed: {str(e)[:160]}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only; never restart or alert")
    args = ap.parse_args()

    now = _now()
    if not _is_market_hours(now):
        print(f"self-heal: outside market hours ({now:%Y-%m-%d %H:%M}); no-op.")
        return 0

    result = assess(now)
    state = _load_state()
    today = now.date().isoformat()
    if state.get("day") != today:                      # new day -> reset daily counters
        state = {"day": today}

    base = {"ts": now.isoformat(timespec="seconds"), "healthy": result["healthy"],
            "reason": result["reason"], **result["detail"]}

    if result["healthy"]:
        if state.get("consecutive_unhealthy"):
            _log({**base, "action": "RECOVERED"})
        state["consecutive_unhealthy"] = 0
        state["escalated"] = False
        state["rl_alerted"] = False          # recovered — re-arm the rate-limit alert for next time
        _save_state(state)
        d = result["detail"]
        print(f"self-heal: HEALTHY ({d.get('real_analyses')}/{d.get('window')} recent cycles are real "
              f"analyses, newest {d.get('decision_age_secs','?')}s ago)")
        return 0

    # Unhealthy path
    consec = int(state.get("consecutive_unhealthy") or 0) + 1
    state["consecutive_unhealthy"] = consec
    restarts = int(state.get("restarts_today") or 0)

    if args.dry_run:
        _save_state(state)
        _log({**base, "action": "DRY_RUN_WOULD_REMEDIATE", "consecutive_unhealthy": consec})
        return 0

    # Escalate rather than restart-loop forever: a persistent fault after N restarts is a broker-side
    # issue a human must clear. Name the ACTUAL cause (data-API subscription vs token) — see _data_fault_hint.
    if restarts >= MAX_RESTARTS_BEFORE_ESCALATE:
        if not state.get("escalated"):
            _alert(f"🚑 SELF-HEAL: agent still NOT trading-capable after {restarts} auto-restarts "
                   f"today ({result['reason']}). {_data_fault_hint()} Auto-restarts paused to avoid a loop.")
            state["escalated"] = True
        _save_state(state)
        _log({**base, "action": "ESCALATED_MANUAL", "restarts_today": restarts})
        return 0

    # Restart only on the 2nd consecutive unhealthy read (avoid acting on a single transient blip),
    # and respect the cooldown.
    last_restart = state.get("last_restart_ts")
    cooldown_ok = True
    if last_restart:
        try:
            cooldown_ok = (now - datetime.fromisoformat(last_restart)).total_seconds() >= RESTART_COOLDOWN_SECS
        except Exception:  # noqa: BLE001
            cooldown_ok = True

    # RATE-LIMIT BACK-OFF: never restart into a 429 — it bursts more requests and deepens the throttle.
    # Back off (idle poll) and let it cool; the running agent recovers on its own once the limit lifts.
    if result["reason"] == "CHAIN_BLACKOUT" and _is_rate_limited():
        state["consecutive_unhealthy"] = 0    # don't let a rate-limit accrue toward the restart trigger
        _save_state(state)
        if not state.get("rl_alerted"):
            _alert("⏳ SELF-HEAL: chain blackout is a Dhan RATE-LIMIT (429), not a fault — backing off "
                   "(no restart; restarts make it worse). Agent keeps polling; will recover when it cools.")
            state["rl_alerted"] = True
            _save_state(state)
        _log({**base, "action": "RATE_LIMIT_BACKOFF"})
        return 0

    if consec >= 2 and cooldown_ok:
        ok = _restart_agent()
        state["last_restart_ts"] = now.isoformat()
        state["restarts_today"] = restarts + 1
        state["consecutive_unhealthy"] = 0            # give the restart a cycle to take effect
        _save_state(state)
        _alert(f"🔧 SELF-HEAL: {result['reason']} detected — auto-restarted the agent "
               f"(restart #{restarts + 1} today). Will re-check next cycle.")
        _log({**base, "action": "AUTO_RESTART", "restart_ok": ok, "restart_number": restarts + 1})
    else:
        _save_state(state)
        _log({**base, "action": "WATCH", "consecutive_unhealthy": consec, "cooldown_ok": cooldown_ok})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
