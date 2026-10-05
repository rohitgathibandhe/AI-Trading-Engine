#!/usr/bin/env python
"""DAILY SELF-DIAGNOSIS — the closing-loop that makes a defect impossible to ignore.

The gap this closes: audits (missed_trade_audit, maturity_scorecard) already DETECT problems, but they
only print to a log. Nothing reads them, so a real defect — e.g. 2026-08-04, where a guard blocked a
+2,985 put_debit and the agent stood aside — sat silent for a day until a human asked. That is the
opposite of self-healing.

This job runs AFTER THE CLOSE (launchd, ~16:20 IST, after the shadow book is written). For the latest
completed session it asks three questions and, if any answers badly, PUSHES a phone alert (Telegram)
with the diagnosis AND the specific guard to challenge — so the fix happens the next morning, not the
next time someone happens to ask.

  1. MISS       agent stood aside while a defined-risk structure cleared the "worth taking" bar
                -> alert names the dominant stand-aside reason = the guard to re-examine
  2. BAD LOSS   agent traded and lost more than the soft daily pain threshold
  3. SUBOPTIMAL agent traded but left a large gap vs the best-available defined structure

It does NOT rewrite selector logic (auto-editing decision code unattended is unsafe). It guarantees the
one thing that was missing: a detected defect SURFACES, loudly, the same evening it happens.

Run:  python scripts/daily_self_diagnosis.py         (add --quiet to skip the phone push; always logs)
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
CREDS = STATE / "creds.json"
DIAG_LOG = STATE / "self_diagnosis.jsonl"
LATEST = STATE / "LATEST_SELF_DIAGNOSIS.txt"

# Naked structures are banned, so a day only a naked short would have won is NOT a miss.
DEFINED = ("put_debit", "call_debit", "bull_put", "bear_call", "iron_fly", "iron_condor")
GOOD_TRADE_RUPEES = 500.0      # a defined-risk structure clearing this = a trade worth having taken
BAD_LOSS_RUPEES = -4000.0      # traded and lost worse than this -> flag for retro
SUBOPTIMAL_GAP_RUPEES = 2500.0 # traded, but best-available beat what we took by more than this
EXIT_GIVEBACK_RUPEES = -1200.0 # total handed back by exiting early vs holding to close -> flag


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _agent_day(day: str) -> tuple[bool, float, str | None]:
    """(did_trade, realized_pnl, first_strategy) for a given session date."""
    traded = False
    pnl = 0.0
    first_strat = None
    for r in _jsonl(STATE / "intraday_v83_paper_live_trades.jsonl"):
        if str(r.get("session_date")) != day:
            continue
        if r.get("event") == "PAPER_ENTRY":
            traded = True
            if first_strat is None:
                first_strat = r.get("strategy")
        if r.get("event") == "PAPER_EXIT" and r.get("realized_paper_pnl") is not None:
            pnl += float(r["realized_paper_pnl"])
    return traded, pnl, first_strat


_SMAP = {"BULL_PUT_CREDIT_SPREAD": "bull_put", "BEAR_CALL_CREDIT_SPREAD": "bear_call",
         "PUT_DEBIT_SPREAD": "put_debit", "CALL_DEBIT_SPREAD": "call_debit",
         "IRON_FLY": "iron_fly", "IRON_CONDOR": "iron_condor"}


def _standaside_reason(day: str) -> str:
    """Dominant reason the agent stood aside that day (from the runner-log rationale) = the guard."""
    c: collections.Counter = collections.Counter()
    f = STATE / "intraday_v83_runner.log"
    if not f.exists():
        return "reason not in log"
    for line in f.read_text().splitlines():
        if f"{day}T" not in line or not line.strip().startswith("{"):
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (j.get("metadata") or {}).get("selector_family") == "STAND_ASIDE" or j.get("action") == "NO_TRADE":
            rat = (j.get("rationale") or [""])[0]
            for key in ("ORB", "cheap", "before", "blackout", "OVERHEAD", "VETO", "CHOP",
                        "RANGE_TIGHT", "INSUFFICIENT", "MIN_5M", "efficiency", "pin"):
                if key.lower() in rat.lower():
                    c[key] += 1
                    break
    return c.most_common(1)[0][0] if c else "no clear reason in log"


def _diagnose(day: str, structures: dict) -> list[dict]:
    """Return a list of red-flag findings for the given day (empty = clean)."""
    avail = {k: (structures.get(k) or {}).get("pnl_rupees")
             for k in DEFINED if isinstance(structures.get(k), dict)}
    avail = {k: v for k, v in avail.items() if v is not None}
    if not avail:
        return []
    best_k = max(avail, key=avail.get)
    best_v = avail[best_k]

    did_trade, agent_pnl, first_strat = _agent_day(day)
    findings: list[dict] = []

    # 1. MISS — stood aside while a defined-risk winner was available
    if not did_trade and best_v >= GOOD_TRADE_RUPEES:
        findings.append({
            "kind": "MISS",
            "detail": f"stood aside; {best_k} would have made {best_v:+,.0f}",
            "guard": _standaside_reason(day),
            "cost_rupees": round(best_v),
        })

    # 2. BAD LOSS — traded and lost hard
    if did_trade and agent_pnl <= BAD_LOSS_RUPEES:
        findings.append({
            "kind": "BAD_LOSS",
            "detail": f"traded {first_strat} for {agent_pnl:+,.0f} (worse than {BAD_LOSS_RUPEES:+,.0f})",
            "guard": "entry/exit — retro this loss",
            "cost_rupees": round(agent_pnl),
        })

    # 3. SUBOPTIMAL — traded, but a much better defined structure was on the table
    if did_trade:
        took_k = _SMAP.get(str(first_strat))
        took_v = avail.get(took_k)
        if took_v is not None and (best_v - took_v) >= SUBOPTIMAL_GAP_RUPEES:
            findings.append({
                "kind": "SUBOPTIMAL",
                "detail": f"took {took_k} ({took_v:+,.0f}); best was {best_k} ({best_v:+,.0f}), "
                          f"gap {best_v - took_v:+,.0f}",
                "guard": "structure selection — wrong pick, not wrong direction",
                "cost_rupees": round(best_v - took_v),
            })
    return findings


def _exit_giveback(day: str) -> list[dict]:
    """From exit_shadow_toclose.jsonl: did the agent's exits hand back money vs holding to close?
    exit_cost_rupees = actual - hold_to_close (< 0 = gave up money). This is the give-back the raw
    P&L hides — 2026-08-06 netted -62 but handed back ~1,644 by exiting two flies in <1.2 min."""
    rows = [r for r in _jsonl(STATE / "exit_shadow_toclose.jsonl") if str(r.get("session_date")) == day]
    costs = [float(r.get("exit_cost_rupees")) for r in rows if r.get("exit_cost_rupees") is not None]
    given_back = sum(c for c in costs if c < 0)
    if given_back <= EXIT_GIVEBACK_RUPEES:
        worst = min(rows, key=lambda r: (r.get("exit_cost_rupees") or 0), default={})
        wr = worst.get("exit_cost_rupees")
        detail = f"exits handed back {given_back:+,.0f} vs holding to close across {len(costs)} trade(s)"
        if wr is not None:
            detail += (f"; worst: {worst.get('strategy')} exited on "
                       f"{(worst.get('actual') or {}).get('exit_reason')} gave up {wr:+,.0f}")
        return [{
            "kind": "EXIT_GIVEBACK",
            "detail": detail,
            "guard": "exit timing — a stop fired too early on a trade that would have won held",
            "cost_rupees": round(given_back),
        }]
    return []


def _load_creds() -> dict:
    try:
        return json.loads(CREDS.read_text()) if CREDS.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def _send_telegram(text: str) -> bool:
    creds = _load_creds()
    bot = str(creds.get("telegram_bot_token") or "").strip()
    chat = str(creds.get("telegram_chat_id") or "").strip()
    if not (bot and chat):
        print("[self-diagnosis] no telegram creds; alert logged only", file=sys.stderr)
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{bot}/sendMessage",
                          json={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=15)
        return r.status_code == 200
    except Exception as e:  # noqa: BLE001
        print(f"[self-diagnosis] telegram failed: {str(e)[:160]}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="log only, no phone push")
    ap.add_argument("--date", help="diagnose a specific session date (default: latest in shadow book)")
    args = ap.parse_args()

    shadow = _jsonl(STATE / "shadow_book.jsonl")
    if not shadow:
        print("No shadow-book history yet — nothing to diagnose.")
        return 0

    by_date = {str(r.get("date")): (r.get("structures") or {}) for r in shadow}
    day = args.date or max(by_date)
    findings = _diagnose(day, by_date.get(day, {}))
    findings += _exit_giveback(day)   # independent of the shadow book (reads the to-close ledger)

    stamp = datetime.now(timezone.utc).isoformat()
    record = {"ts": stamp, "date": day, "clean": not findings, "findings": findings}
    with DIAG_LOG.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    if not findings:
        msg = f"SELF-DIAGNOSIS {day}: clean — no miss, no bad loss, no wrong-structure pick."
        LATEST.write_text(msg + "\n")
        print(msg)
        return 0

    total = sum(f["cost_rupees"] for f in findings)
    lines = [f"⚠️ SELF-DIAGNOSIS {day} — {len(findings)} red flag(s), ~Rs {total:+,.0f} impact\n"]
    for f in findings:
        lines.append(f"• <b>{f['kind']}</b>: {f['detail']}")
        lines.append(f"  guard to fix: <i>{f['guard']}</i>")
    lines.append("\nThe guard named above is the defect to re-examine tomorrow — a stand-aside is not")
    lines.append("automatically discipline. Fix it the next session, don't let it repeat.")
    text = "\n".join(lines)

    LATEST.write_text(text.replace("<b>", "").replace("</b>", "")
                          .replace("<i>", "").replace("</i>", "") + "\n")
    print(text.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))

    if not args.quiet:
        ok = _send_telegram(text)
        print(f"\n[self-diagnosis] phone alert {'sent' if ok else 'FAILED (logged to LATEST_SELF_DIAGNOSIS.txt)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
