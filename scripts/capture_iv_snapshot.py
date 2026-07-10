#!/usr/bin/env python3
"""Capture an ATM premium / IV snapshot at market OPEN and CLOSE each day.

This is the data tap for the vol pillars: IV rank needs a history of daily IV to
rank against, and the premium-crush study needs open→close premium behaviour. The
agent already computes these per-cycle but never persists an open/close pair; this
does, appending one row per (date, phase) to iv_snapshots.csv.

Usage:  python scripts/capture_iv_snapshot.py --phase OPEN|CLOSE
Scheduled via launchd at ~09:20 and ~15:25 IST (see the two .plist files).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    _IST = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_engine"))
STATE = ROOT / "data_engine" / "market_ai" / "state"
SNAPSHOTS = STATE / "iv_snapshots.csv"
IV_HISTORY = STATE / "iv_history.csv"          # legacy day-value file (kept in sync)
MARKET_CONTEXT = STATE / "market_context.json"
CREDS = STATE / "creds.json"

FIELDS = ["date", "phase", "timestamp", "spot", "atm_strike",
          "atm_straddle", "atm_call", "atm_put", "avg_iv", "india_vix"]


def _now_ist() -> datetime:
    return datetime.now(_IST) if _IST else datetime.now()


def _json_read(path: Path) -> dict:
    try:
        import json
        return json.load(open(path))
    except Exception:
        return {}


def _india_vix() -> float | None:
    ctx = _json_read(MARKET_CONTEXT)
    v = ctx.get("india_vix")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def capture_chain_snapshot() -> dict:
    """Fetch the live Nifty chain and distil ATM straddle premium + near-ATM avg IV."""
    from market_ai.dhan_wrapper import DhanWrapper  # type: ignore

    creds = _json_read(CREDS)
    cid = (creds.get("client_id") or "").strip()
    tok = (creds.get("access_token") or "").strip()
    if cid and tok:
        os.environ["DHAN_CLIENT_ID"] = cid
        os.environ["DHAN_ACCESS_TOKEN"] = tok

    dw = DhanWrapper(logger=None)
    expiry = dw.get_optionchain_expirylist("IDX_I", 13)[0]
    raw = dw.get_option_chain(13, "IDX_I", expiry)
    data = (((raw or {}).get("data") or {}).get("data")) or {}
    spot = float(data.get("last_price") or 0.0)
    oc = data.get("oc") or {}

    strikes = []
    for k in oc.keys():
        try:
            strikes.append(float(k))
        except (TypeError, ValueError):
            continue
    if not strikes or spot <= 0:
        raise RuntimeError(f"empty/invalid chain (spot={spot}, strikes={len(strikes)})")

    atm = min(strikes, key=lambda s: abs(s - spot))

    def leg(strike: float, side: str) -> dict:
        node = oc.get(f"{strike:.6f}") or oc.get(str(strike)) or {}
        return node.get(side) or {}

    ce, pe = leg(atm, "ce"), leg(atm, "pe")
    atm_call = float(ce.get("last_price") or 0.0)
    atm_put = float(pe.get("last_price") or 0.0)
    atm_straddle = round(atm_call + atm_put, 2)

    # Average IV over the ATM +/- 2 strikes, ignoring zero/missing quotes.
    strikes.sort()
    idx = strikes.index(atm)
    window = strikes[max(0, idx - 2): idx + 3]
    ivs = []
    for s in window:
        for side in ("ce", "pe"):
            iv = leg(s, side).get("implied_volatility")
            try:
                iv = float(iv)
            except (TypeError, ValueError):
                iv = 0.0
            if iv > 0:
                ivs.append(iv)
    avg_iv = round(sum(ivs) / len(ivs), 4) if ivs else None

    return {
        "spot": round(spot, 2), "atm_strike": atm,
        "atm_straddle": atm_straddle, "atm_call": atm_call, "atm_put": atm_put,
        "avg_iv": avg_iv, "india_vix": _india_vix(),
    }


def _already_captured(date_str: str, phase: str) -> bool:
    if not SNAPSHOTS.exists():
        return False
    try:
        for r in csv.DictReader(open(SNAPSHOTS)):
            if r.get("date") == date_str and (r.get("phase") or "").upper() == phase:
                return True
    except Exception:
        pass
    return False


def _append(row: dict) -> None:
    new = not SNAPSHOTS.exists()
    with open(SNAPSHOTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDS})


def _sync_iv_history(date_str: str, avg_iv) -> None:
    """Keep the legacy date,avg_chain_iv file current (CLOSE value wins)."""
    if avg_iv is None:
        return
    try:
        rows = list(csv.DictReader(open(IV_HISTORY))) if IV_HISTORY.exists() else []
        rows = [r for r in rows if r.get("date") != date_str]
        rows.append({"date": date_str, "avg_chain_iv": avg_iv})
        with open(IV_HISTORY, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "avg_chain_iv"])
            w.writeheader()
            w.writerows(rows)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["OPEN", "CLOSE"])
    ap.add_argument("--force", action="store_true", help="capture even if already recorded")
    args = ap.parse_args()

    now = _now_ist()
    date_str = now.strftime("%Y-%m-%d")
    phase = args.phase.upper()

    if now.weekday() >= 5:
        print(f"[iv-snapshot] {date_str} is a weekend — skipping {phase}")
        return 0
    if _already_captured(date_str, phase) and not args.force:
        print(f"[iv-snapshot] {date_str} {phase} already captured — skipping")
        return 0

    try:
        snap = capture_chain_snapshot()
    except Exception as exc:
        print(f"[iv-snapshot] {date_str} {phase} FAILED: {exc}")
        return 1

    row = {"date": date_str, "phase": phase,
           "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S%z"), **snap}
    _append(row)
    if phase == "CLOSE":
        _sync_iv_history(date_str, snap.get("avg_iv"))
    print(f"[iv-snapshot] {date_str} {phase}: spot {snap['spot']} ATM {snap['atm_strike']:.0f} "
          f"straddle {snap['atm_straddle']} avg_iv {snap['avg_iv']} vix {snap['india_vix']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
