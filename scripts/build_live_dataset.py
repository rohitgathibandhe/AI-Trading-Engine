#!/usr/bin/env python
"""Convert the LIVE chain captures (rolling_option_live/<date>/chain_snapshots.jsonl — real bid/ask,
real vol smile, real ΔOI) into a backtest dataset (options_chain.csv + nifty_5m/15m.csv), so the
confluence gate / put-debit engine can be validated on the ONLY trustworthy substrate rather than the
synthetic dense set (5x-too-wide spreads, no smile — 'a policy the live agent never ran').

Snapshots are ~16-min cadence, so we floor each to the 5-min grid and keep the last per bucket.
OHLCV comes from the existing candle_cache_<date>_5m.json files (same live source).

  python scripts/build_live_dataset.py <out_dir>
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data_engine" / "market_ai" / "state"
LIVE = STATE / "rolling_option_live"


def _floor5(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0, minute=(ts.minute // 5) * 5)


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chain_rows = []      # options_chain.csv rows
    spot_by_ts: dict[datetime, float] = {}
    for day_dir in sorted(LIVE.iterdir()):
        f = day_dir / "chain_snapshots.jsonl"
        if not f.exists():
            continue
        # keep the LAST snapshot per 5-min bucket
        by_bucket: dict[datetime, dict] = {}
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            snap = json.loads(line)
            try:
                ts = _floor5(_parse_ts(snap["captured_at"]).replace(tzinfo=None))
            except Exception:
                continue
            by_bucket[ts] = snap
        for ts, snap in by_bucket.items():
            resp = (snap.get("response") or {}).get("data", {}).get("data", {})
            oc = resp.get("oc") or {}
            spot = resp.get("last_price")
            expiry = snap.get("expiry")
            if not oc or spot is None:
                continue
            spot_by_ts[ts] = float(spot)
            for strike_s, node in oc.items():
                strike = float(strike_s)
                for side, opt in (("CALL", node.get("ce")), ("PUT", node.get("pe"))):
                    if not opt:
                        continue
                    g = opt.get("greeks") or {}
                    bid = opt.get("top_bid_price") or 0.0
                    ask = opt.get("top_ask_price") or 0.0
                    ltp = opt.get("last_price") or 0.0
                    # skip strikes with no market (bid=ask=0 and no ltp) — dead wings
                    if not (bid or ask or ltp):
                        continue
                    chain_rows.append({
                        "timestamp": ts.isoformat(),
                        "expiry": expiry,
                        "spot": spot,
                        "strike": strike,
                        "option_type": side,
                        "bid": bid,
                        "ask": ask,
                        "ltp": ltp,
                        "delta": g.get("delta", ""),
                        "iv": opt.get("implied_volatility", ""),
                        "oi": opt.get("oi", ""),
                    })

    # options_chain.csv
    oc_path = out_dir / "options_chain.csv"
    cols = ["timestamp", "expiry", "spot", "strike", "option_type", "bid", "ask", "ltp", "delta", "iv", "oi"]
    with oc_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(chain_rows)

    # OHLCV from candle_cache (fallback: derive a 1-bar-per-5min close series from spot_by_ts)
    bars_5m = _load_candles(spot_by_ts)
    _write_ohlcv(out_dir / "nifty_5m.csv", bars_5m)
    _write_ohlcv(out_dir / "nifty_15m.csv", _resample_15m(bars_5m))

    (out_dir / "dataset_metadata.json").write_text(json.dumps({
        "source": "rolling_option_live (real bid/ask + smile)",
        "chain_rows": len(chain_rows),
        "timestamps": len(spot_by_ts),
        "days": len({ts.date() for ts in spot_by_ts}),
    }, indent=2))
    print(f"chain_rows={len(chain_rows)} timestamps={len(spot_by_ts)} days={len({ts.date() for ts in spot_by_ts})}")
    print(f"5m bars={len(bars_5m)}")
    print(f"written to {out_dir}")


def _load_candles(spot_by_ts: dict[datetime, float]) -> list[dict]:
    """Prefer the candle_cache_*_5m.json files; fall back to spot-derived flat bars."""
    bars: dict[datetime, dict] = {}
    for ts, spot in spot_by_ts.items():
        cc = STATE / f"candle_cache_{ts.date().isoformat()}_5m.json"
        # default: flat bar from spot (used if no candle match)
        bars.setdefault(ts, {"timestamp": ts.isoformat(), "open": spot, "high": spot,
                             "low": spot, "close": spot, "volume": 0})
    # overlay real OHLCV where candle_cache has the bar
    seen_days = {ts.date() for ts in spot_by_ts}
    for day in seen_days:
        cc = STATE / f"candle_cache_{day.isoformat()}_5m.json"
        if not cc.exists():
            continue
        try:
            data = json.loads(cc.read_text())
            cand = data if isinstance(data, list) else (data.get("candles") or data.get("data") or [])
        except Exception:
            continue
        for c in cand:
            t = c.get("timestamp") or c.get("t") or c.get("time")
            if not t:
                continue
            try:
                cts = _floor5(_parse_ts(str(t)).replace(tzinfo=None))
            except Exception:
                continue
            if cts in bars:
                bars[cts] = {"timestamp": cts.isoformat(), "open": c.get("open"), "high": c.get("high"),
                             "low": c.get("low"), "close": c.get("close"), "volume": c.get("volume", 0)}
    return [bars[k] for k in sorted(bars)]


def _resample_15m(bars_5m: list[dict]) -> list[dict]:
    out: dict[datetime, dict] = {}
    for b in bars_5m:
        ts = _parse_ts(b["timestamp"])
        k = ts.replace(minute=(ts.minute // 15) * 15)
        if k not in out:
            out[k] = {"timestamp": k.isoformat(), "open": b["open"], "high": b["high"],
                      "low": b["low"], "close": b["close"], "volume": b.get("volume", 0) or 0}
        else:
            o = out[k]
            o["high"] = max(o["high"], b["high"]); o["low"] = min(o["low"], b["low"])
            o["close"] = b["close"]; o["volume"] = (o["volume"] or 0) + (b.get("volume", 0) or 0)
    return [out[k] for k in sorted(out)]


def _write_ohlcv(path: Path, bars: list[dict]) -> None:
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for b in bars:
            w.writerow({k: b.get(k) for k in cols})


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else STATE / "intraday_live_dataset"
    build(out)
