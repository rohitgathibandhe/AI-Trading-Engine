"""
Live monitor loop sampling recent spot snapshots and positions, computing S/R zones and a risk view.

Inputs:
- Recent spot snapshots: uses option-chain snapshots under state/option_chain_history/* (written by collect_live_option_chain.py)
- Positions: reads state/dhan_positions.json if present (or empty fallback)

Outputs:
- state/live_monitor/latest.json with zones and position health summary
- Logs to stdout

This avoids scraping charts; it uses structured data we already collect.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from market_ai.modules.analytics.live_trade_monitor import (
    evaluate_position_health,
    find_multi_tf_zones,
    summarize_position_health,
)

LOG = logging.getLogger("live_monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "state"
OC_HISTORY_DIR = STATE_DIR / "option_chain_history"
LIVE_MONITOR_DIR = STATE_DIR / "live_monitor"
LIVE_MONITOR_DIR.mkdir(parents=True, exist_ok=True)


def _load_positions() -> List[Dict]:
    pos_file = STATE_DIR / "dhan_positions.json"
    if pos_file.exists():
        try:
            return json.loads(pos_file.read_text())
        except Exception:
            LOG.warning("Failed to read %s", pos_file)
    return []


def _load_recent_spot_series(limit_minutes: int = 24 * 60 * 7) -> pd.DataFrame:
    """Pull spot history from saved option-chain snapshots (spot columns in JSON).
    Default lookback ~7 days to allow 1D/1W aggregation if snapshots exist.
    """
    rows: List[Dict] = []
    cutoff = datetime.now() - timedelta(minutes=limit_minutes)
    if not OC_HISTORY_DIR.exists():
        return pd.DataFrame()
    for day_dir in sorted(OC_HISTORY_DIR.iterdir(), reverse=True):
        if not day_dir.is_dir():
            continue
        for snap in sorted(day_dir.glob("*.json"), reverse=True):
            try:
                ts_str = snap.stem.split("_")[0]
                ts = datetime.fromisoformat(ts_str.replace("T", " ")) if "T" in ts_str else datetime.fromisoformat(
                    snap.stem.split("_")[0] + " " + snap.stem.split("_")[1][:2] + ":" + snap.stem.split("_")[1][2:]
                )
            except Exception:
                ts = datetime.fromtimestamp(snap.stat().st_mtime)
            if ts < cutoff:
                break
            try:
                js = json.loads(snap.read_text())
                spot = None
                data = js.get("data") or js
                if isinstance(data, dict):
                    # common places spot appears
                    spot = data.get("spot") or data.get("last_price") or data.get("underlyingValue")
                    oc = data.get("oc") or data.get("optionchain")
                    if spot is None and isinstance(oc, dict):
                        # take any leg's spot
                        for leg in oc.values():
                            for side in ("ce", "pe"):
                                leg_obj = (leg or {}).get(side) or {}
                                spot = leg_obj.get("spot") or leg_obj.get("underlyingValue")
                                if spot:
                                    break
                            if spot:
                                break
                if spot is None:
                    continue
                rows.append({"timestamp": ts, "spot": float(spot)})
            except Exception:
                continue
        if rows and rows[-1]["timestamp"] < cutoff:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).dropna()
    df = df.sort_values("timestamp")
    df = df[df["timestamp"] >= cutoff]
    return df


def _make_multi_tf_candles(spot_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if spot_df.empty:
        return {}
    spot_df = spot_df.set_index("timestamp").resample("1T").last().dropna()
    candles = {}
    rules = {
        "1m": "1T",
        "5m": "5T",
        "15m": "15T",
        "1h": "1H",
        "1d": "1D",
        "1w": "1W",
    }
    for tf, rule in rules.items():
        ohlc = spot_df["spot"].resample(rule).ohlc()
        ohlc = ohlc.dropna()
        ohlc = ohlc.rename(columns={"close": "spot"})
        ohlc = ohlc.reset_index().rename(columns={"open": "open", "high": "high", "low": "low"})
        ohlc["timestamp"] = pd.to_datetime(ohlc["timestamp"])
        ohlc["spot"] = ohlc["spot"].fillna(ohlc["close"] if "close" in ohlc else ohlc["open"])
        candles[tf] = ohlc
    return candles


def run_once() -> Dict[str, object]:
    positions = _load_positions()
    spot_df = _load_recent_spot_series()
    candles = _make_multi_tf_candles(spot_df)
    zones = find_multi_tf_zones(candles)
    recent_events: List[Dict[str, Any]] = []
    events_path = STATE_DIR / "events.jsonl"
    if events_path.exists():
        try:
            cutoff = datetime.now() - timedelta(days=2)
            for line in events_path.read_text().splitlines():
                try:
                    ev = json.loads(line)
                    ts = ev.get("timestamp")
                    if ts:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if dt >= cutoff:
                            recent_events.append(ev)
                except Exception:
                    continue
        except Exception:
            pass

    spot_latest = float(spot_df["spot"].iloc[-1]) if not spot_df.empty else None
    prev_close = float(spot_df["spot"].iloc[0]) if len(spot_df.index) else None
    gap_pct = None
    if spot_latest is not None and prev_close:
        gap_pct = (spot_latest - prev_close) / prev_close if prev_close else None
    nearest_sup = nearest_res = None
    if spot_latest is not None and zones:
        supports = [z for z in zones if z.side == "support"]
        resistances = [z for z in zones if z.side == "resistance"]
        nearest_sup = min(supports, key=lambda z: abs(z.price - spot_latest), default=None) if supports else None
        nearest_res = min(resistances, key=lambda z: abs(z.price - spot_latest), default=None) if resistances else None

    health_summaries = []
    for pos in positions:
        pnl = float(pos.get("pnl", pos.get("mtm", 0)) or 0.0)
        pnl_pct = pos.get("pnl_pct") or pos.get("mtm_pct")
        try:
            pnl_pct = float(pnl_pct) if pnl_pct is not None else None
        except Exception:
            pnl_pct = None
        health = evaluate_position_health(
            pnl=pnl,
            pnl_pct=pnl_pct,
            spot=spot_latest or 0.0,
            zones=zones,
            gap_pct=gap_pct,
            support=nearest_sup,
            resistance=nearest_res,
        )
        summary = summarize_position_health(health)
        summary["symbol"] = pos.get("tradingSymbol") or pos.get("symbol") or pos.get("security_id")
        health_summaries.append(summary)

    payload = {
        "as_of": datetime.now().isoformat(),
        "spot": spot_latest,
        "prev_close": prev_close,
        "gap_pct": gap_pct,
        "zones": [z.__dict__ for z in zones],
        "positions": health_summaries,
        "events": recent_events,
    }
    out_file = LIVE_MONITOR_DIR / "latest.json"
    out_file.write_text(json.dumps(payload, indent=2, default=str))
    LOG.info("Wrote live monitor snapshot to %s (positions=%d, zones=%d)", out_file, len(health_summaries), len(zones))
    return payload


def main():
    run_once()


if __name__ == "__main__":
    main()
