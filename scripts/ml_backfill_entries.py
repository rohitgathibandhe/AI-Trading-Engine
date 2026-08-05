#!/usr/bin/env python3
"""
ML feature backfill for historical PAPER_ENTRY records.

Existing trades (before 2026-07-07 fix) lack ml_* fields so watchdog
_train_win_probability_model() skips them entirely.  This script adds
best-effort approximations from the gate dict and other available fields
so those trades contribute to ML training.

Reconstruction quality per feature:
  ml_bearish_entry_score   ← gate.bearish_trade_score          [HIGH]
  ml_bullish_entry_score   ← gate.bullish_trade_score           [HIGH if present, else 0.0]
  ml_india_vix             ← india_vix_at_entry                 [HIGH if present, else 0.0]
  ml_days_to_monthly_expiry← computed from session_date         [HIGH]
  setup_direction          ← derived from strategy name         [HIGH]
  ml_vwap_reclaim          ← 0.5 (neutral)                      [LOW — can't reconstruct]
  ml_banknifty_divergence  ← 0.0 (was broken before fix)        [MEDIUM — was always 0.0]
  ml_bullish_momentum      ← 0.5 (neutral)                      [LOW]
  ml_bearish_momentum      ← 0.5 (neutral)                      [LOW]
  ml_rv30_pct              ← 0.0                                 [LOW]
  ml_order_flow_imbalance  ← 0.0                                 [LOW — can't reconstruct]

Usage:
  python scripts/ml_backfill_entries.py           # dry run — shows what would change
  python scripts/ml_backfill_entries.py --apply   # rewrite PAPER_ENTRY records
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path

_STATE_ROOT = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
_TRADES_FILE = _STATE_ROOT / "intraday_v83_paper_live_trades.jsonl"

_ML_FEAT_NAMES = [
    "ml_bearish_entry_score",
    "ml_bullish_entry_score",
    "ml_india_vix",
    "ml_vwap_reclaim",
    "ml_banknifty_divergence",
    "ml_days_to_monthly_expiry",
    "ml_bullish_momentum",
    "ml_bearish_momentum",
    "ml_rv30_pct",
    "ml_order_flow_imbalance",
]


def _last_thursday_of_month(yr: int, mo: int) -> date:
    last_day = monthrange(yr, mo)[1]
    d = date(yr, mo, last_day)
    # weekday(): Monday=0 … Thursday=3 … Sunday=6
    offset = (d.weekday() - 3) % 7
    return d - timedelta(days=offset)


def _days_to_monthly_expiry(session_date_str: str) -> int:
    """Days from session_date to the nearest upcoming monthly Nifty expiry (last Thursday)."""
    try:
        sd = date.fromisoformat(session_date_str)
    except Exception:
        return 15  # default: ~2 weeks
    # Check this month and next month
    for delta_months in (0, 1):
        mo = sd.month + delta_months
        yr = sd.year + (mo - 1) // 12
        mo = (mo - 1) % 12 + 1
        exp = _last_thursday_of_month(yr, mo)
        if exp >= sd:
            return max(0, (exp - sd).days)
    return 30


def _setup_direction(strategy: str | None) -> str:
    s = str(strategy or "").upper()
    if "BULL" in s:
        return "BULLISH"
    if "CONDOR" in s or "STRANGLE" in s or "STRADDLE" in s:
        return "RANGE"
    return "BEARISH"


def _reconstruct_ml_features(entry: dict) -> dict:
    """Return a dict of ml_* features reconstructed from available entry data."""
    gate = entry.get("gate") or {}
    strategy = entry.get("strategy") or gate.get("strategy") or ""
    session_date = entry.get("session_date") or ""

    bearish_score = float(gate.get("bearish_trade_score") or 0.0)
    bullish_score = float(gate.get("bullish_trade_score") or 0.0)
    india_vix = float(entry.get("india_vix_at_entry") or 0.0)
    days_to_exp = _days_to_monthly_expiry(session_date)

    return {
        "ml_bearish_entry_score":    bearish_score,
        "ml_bullish_entry_score":    bullish_score,
        "ml_india_vix":              india_vix,
        "ml_vwap_reclaim":           0.5,   # neutral — can't reconstruct
        "ml_banknifty_divergence":   0.0,   # was always 0.0 before fix
        "ml_days_to_monthly_expiry": float(days_to_exp),
        "ml_bullish_momentum":       0.5,   # neutral
        "ml_bearish_momentum":       0.5,   # neutral
        "ml_rv30_pct":               0.0,
        "ml_order_flow_imbalance":   0.0,
        "setup_direction":           _setup_direction(strategy),
        "_backfilled":               True,   # marker so future code can identify these
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill ml_* features into historical PAPER_ENTRY records")
    parser.add_argument("--apply", action="store_true", help="Actually rewrite the JSONL (default is dry-run)")
    args = parser.parse_args()

    if not _TRADES_FILE.exists():
        print(f"ERROR: {_TRADES_FILE} not found")
        sys.exit(1)

    lines = _TRADES_FILE.read_text().splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"WARN: skipping malformed line: {e}")
            records.append(None)

    enriched_count = 0
    already_ok_count = 0
    updated: list[dict | None] = []

    for rec in records:
        if rec is None:
            updated.append(None)
            continue
        if rec.get("event") != "PAPER_ENTRY":
            updated.append(rec)
            continue
        # Already has ml_* fields — leave it alone
        if any(f in rec for f in _ML_FEAT_NAMES):
            already_ok_count += 1
            updated.append(rec)
            continue
        # Needs enrichment
        ml_fields = _reconstruct_ml_features(rec)
        print(
            f"  Enriching {rec.get('session_date')} {rec.get('playbook')}:"
            f" bearish={ml_fields['ml_bearish_entry_score']:.2f}"
            f" vix={ml_fields['ml_india_vix']:.1f}"
            f" days_exp={int(ml_fields['ml_days_to_monthly_expiry'])}"
            f" dir={ml_fields['setup_direction']}"
        )
        updated.append({**rec, **ml_fields})
        enriched_count += 1

    print(f"\nSummary: {enriched_count} entries will be enriched, {already_ok_count} already have ml_* fields")

    if enriched_count == 0:
        print("Nothing to do.")
        return

    if not args.apply:
        print("\nDry run — pass --apply to write changes.")
        return

    # Backup original
    backup = _TRADES_FILE.with_suffix(".jsonl.bak")
    backup.write_text(_TRADES_FILE.read_text())
    print(f"Backup written to {backup}")

    # Rewrite
    out_lines = []
    for rec in updated:
        if rec is not None:
            out_lines.append(json.dumps(rec, default=str))
    _TRADES_FILE.write_text("\n".join(out_lines) + "\n")
    print(f"Rewrote {_TRADES_FILE} with {enriched_count} enriched PAPER_ENTRY records.")
    print("Run the EOD watchdog or restart the agent to trigger ML model training.")


if __name__ == "__main__":
    main()
