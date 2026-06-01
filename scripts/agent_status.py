#!/usr/bin/env python3
"""Quick agent status — one command, one screen of output.

Usage:
  python scripts/agent_status.py              # today
  python scripts/agent_status.py 2026-05-27   # specific date
"""
from __future__ import annotations
import json, sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data_engine" / "market_ai" / "state"
IST = timezone(timedelta(hours=5, minutes=30))


def _jwt_expiry(token: str) -> str:
    try:
        import base64
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        d = json.loads(base64.urlsafe_b64decode(payload))
        exp = d.get("exp")
        if exp:
            dt = datetime.fromtimestamp(exp, tz=IST)
            h = (dt - datetime.now(IST)).total_seconds() / 3600
            return f"{dt.strftime('%Y-%m-%d %H:%M IST')} ({h:.1f}h remaining)"
    except Exception:
        pass
    return "unknown"


def _runtime_summary() -> None:
    try:
        rt = json.loads((STATE / "intraday_v83_runtime_state.json").read_text())
        print(f"  Mode:          {rt.get('mode')}")
        print(f"  Block reason:  {rt.get('primary_block_reason')}")
        print(f"  Session date:  {rt.get('session_date')}")
        print(f"  Trades today:  {rt.get('session_trade_count', 0)}")
        print(f"  Realized PnL:  ₹{rt.get('realized_pnl_rupees', 0):,.0f}")
        print(f"  Updated:       {rt.get('updated_at', '')}")
    except Exception as e:
        print(f"  [runtime state unreadable: {e}]")


def _token_status() -> None:
    try:
        creds = json.loads((STATE / "creds.json").read_text())
        token = creds.get("access_token", "")
        print(f"  Token expires: {_jwt_expiry(token)}")
    except Exception:
        print("  Token: unreadable")


def _process_status() -> None:
    import subprocess
    try:
        out = subprocess.check_output(
            ["pgrep", "-a", "python"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.strip().splitlines():
            if "intraday_defined_risk" in line or "paper_pnl_server" in line:
                parts = line.split(None, 1)
                pid = parts[0]
                cmd = parts[1] if len(parts) > 1 else ""
                label = "v83 agent" if "intraday" in cmd else "UI server"
                print(f"  {label}: PID {pid}")
    except Exception:
        print("  [process check failed]")


def _log_summary(date_str: str) -> None:
    runner = STATE / "intraday_v83_runner.log"
    if not runner.exists():
        print("  [runner log missing]")
        return

    decisions, actions, reasons, ivs, best = 0, Counter(), Counter(), [], (0.0, None)
    try:
        for line in runner.read_text(errors="ignore").splitlines():
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
                meta = d.get("metadata", {})
                dr = meta.get("data_readiness", {})
                ts = dr.get("timestamp", "") or ""
                if date_str and not ts.startswith(date_str):
                    # fallback: check rationale for rough match via action (no timestamp in some entries)
                    # only include if no date filter specified or this is ambiguous
                    if date_str:
                        continue
                decisions += 1
                actions[d.get("action", "?")] += 1
                br = meta.get("primary_block_reason") or (d.get("rationale") or ["?"])[0]
                reasons[str(br)[:60]] += 1
                iv = float(meta.get("avg_chain_iv") or 0)
                if iv > 0:
                    ivs.append(iv)
                bs = float(meta.get("bullish_trade_score") or 0)
                thresh = float(meta.get("bullish_trade_score_threshold") or 0)
                if bs > best[0]:
                    best = (bs, thresh)
            except Exception:
                pass
    except Exception as e:
        print(f"  [log read error: {e}]")
        return

    if not decisions:
        print(f"  No decisions found for {date_str}")
        return

    print(f"  Decisions:     {decisions}")
    for action, cnt in actions.most_common():
        print(f"    {action}: {cnt}")
    if ivs:
        print(f"  IV range:      {min(ivs):.1f}% – {max(ivs):.1f}%  (avg {sum(ivs)/len(ivs):.1f}%)")
    if best[0] > 0:
        print(f"  Best score:    {best[0]:.2f} / {best[1]:.2f} threshold "
              f"({'CROSSED ✅' if best[0] >= best[1] else f'gap={best[1]-best[0]:.2f} ❌'})")
    print(f"  Top block reasons:")
    for r, c in reasons.most_common(3):
        print(f"    {c}x  {r}")


def main() -> None:
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now(IST).strftime("%Y-%m-%d")
    print(f"\n{'─'*50}")
    print(f"  AI Trading Engine — Agent Status  [{date_str}]")
    print(f"{'─'*50}")

    print("\n[Token]")
    _token_status()

    print("\n[Processes]")
    _process_status()

    print("\n[Runtime State]")
    _runtime_summary()

    print(f"\n[Decision Log — {date_str}]")
    _log_summary(date_str)

    # Weekly IC pending plan
    pending_file = ROOT / "data_engine" / "market_ai" / "state" / "weekly_ic_pending.json"
    if pending_file.exists():
        try:
            p = json.loads(pending_file.read_text())
            if p.get("status") in ("PENDING", "ASSESSMENT_HOLD"):
                print(f"\n[Weekly IC Pending]")
                print(f"  Expiry: {p.get('expiry')}  SC={p.get('short_call')}  SP={p.get('short_put')}")
                print(f"  Status: {p.get('status')}  Credit: ₹{p.get('gross_credit', 0):,.0f}")
        except Exception:
            pass

    print(f"\n{'─'*50}\n")


if __name__ == "__main__":
    main()
