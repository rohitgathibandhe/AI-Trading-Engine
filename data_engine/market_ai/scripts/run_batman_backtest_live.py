#!/usr/bin/env python3
"""
Run Batman monthly backtest using live Dhan rolling-option API.

Requirements:
- DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN env vars present.
- Uses RollingExpiredOptionsMarket (strikes limited to ATM±10).
- Uses daily close from reports/intraday_from_rolling_latest.csv to anchor dates.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_PATH = Path(__file__).resolve()
# parents: [scripts, market_ai, data_engine, repo_root]
REPO_ROOT = SCRIPT_PATH.parents[3]
for p in (REPO_ROOT, REPO_ROOT / "data_engine"):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from market_ai.modules.strategies.batman_monthly import run_backtest, BatmanConfig  # noqa: E402


STATE_CREDS = REPO_ROOT / "data_engine" / "market_ai" / "state" / "creds.json"
ROLLING_ROOT = REPO_ROOT / "data_engine" / "market_ai" / "state" / "rolling_option"


class LocalRollingMarket:
    """
    Lightweight adapter that reads cached rolling_option parquet files
    from data_engine/market_ai/state/rolling_option/YYYY-MM-DD/rolling_option.parquet
    and serves a dict[strike]->{'ce':{last_price}, 'pe':{last_price}}.
    """

    def __init__(self, root: Path = ROLLING_ROOT):
        self.root = Path(root)

    def get_option_chain(self, underlying_id: int, expiry_or_tag: str, as_of_date: str, underlying_seg: str = "NSE_FNO"):
        day_dir = self.root / as_of_date
        parquet_path = day_dir / "rolling_option.parquet"
        if not parquet_path.exists():
            return {}
        try:
            df = pq.read_table(parquet_path).to_pandas()
        except Exception:
            return {}
        if "expiryDate" not in df.columns or "strikePrice" not in df.columns or "optionType" not in df.columns:
            return {}
        df["expiryDate"] = pd.to_datetime(df["expiryDate"], errors="coerce").dt.date
        try:
            target_expiry = datetime.fromisoformat(expiry_or_tag).date()
        except Exception:
            target_expiry = None
        if target_expiry:
            df = df.loc[df["expiryDate"] == target_expiry]
        if df.empty:
            return {}
        # prefer last quote per strike/optionType
        if "tradeTime" in df.columns and "tradeDate" in df.columns:
            df["ts"] = pd.to_datetime(df["tradeDate"] + " " + df["tradeTime"], errors="coerce")
        elif "timestamp" in df.columns:
            df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            df["ts"] = pd.NaT
        df = df.sort_values("ts")
        latest = df.groupby(["strikePrice", "optionType"]).tail(1)
        chain = {}
        for _, row in latest.iterrows():
            strike = str(float(row["strikePrice"]))
            node = chain.setdefault(strike, {})
            last = row.get("ltp") or row.get("last_price") or row.get("close") or row.get("open") or 0.0
            opt_type = str(row["optionType"]).strip().upper()
            if opt_type == "CALL":
                node["ce"] = {"last_price": float(last)}
            elif opt_type == "PUT":
                node["pe"] = {"last_price": float(last)}
        return chain


def load_daily_spot(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise RuntimeError(f"{csv_path} missing timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    daily = df.sort_values("timestamp").groupby("date").tail(1)
    daily = daily.rename(columns={"spot": "close"})
    return daily[["date", "close"]]


def main() -> None:
    price_csv = REPO_ROOT / "reports" / "intraday_from_rolling_latest.csv"
    if not price_csv.exists():
        raise FileNotFoundError(f"Expected price CSV at {price_csv}")

    daily = load_daily_spot(price_csv)
    # Optional date filter via env vars BACKTEST_START / BACKTEST_END (YYYY-MM-DD)
    start_env = os.getenv("BACKTEST_START")
    end_env = os.getenv("BACKTEST_END")
    if start_env:
        try:
            start_d = datetime.fromisoformat(start_env).date()
            daily = daily[daily["date"] >= start_d]
        except Exception:
            pass
    if end_env:
        try:
            end_d = datetime.fromisoformat(end_env).date()
            daily = daily[daily["date"] <= end_d]
        except Exception:
            pass
    market = LocalRollingMarket()

    cfg = BatmanConfig()
    cfg.market = market
    cfg_dict = cfg.__dict__
    trades, timeline, summary = run_backtest(daily, cfg_dict)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / f"batman_backtest_trades_{ts}.csv"
    summary_path = out_dir / f"batman_backtest_summary_{ts}.json"
    timeline_path = out_dir / f"batman_backtest_playback_{ts}.csv"
    trades.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))
    if timeline is not None and not timeline.empty:
        timeline.to_csv(timeline_path, index=False)
        print(f"Wrote playback -> {timeline_path}")
    print(f"Wrote trades -> {trades_path}")
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
