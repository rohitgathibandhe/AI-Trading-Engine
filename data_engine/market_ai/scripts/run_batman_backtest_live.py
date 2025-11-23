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
from market_ai.modules.data_fetch.rolling_expired_options import RollingExpiredOptionsMarket  # noqa: E402


STATE_CREDS = REPO_ROOT / "data_engine" / "market_ai" / "state" / "creds.json"


def load_creds_from_state() -> bool:
    if os.getenv("DHAN_CLIENT_ID") and os.getenv("DHAN_ACCESS_TOKEN"):
        return True
    if not STATE_CREDS.exists():
        return False
    try:
        import json

        data = json.loads(STATE_CREDS.read_text())
        cid = data.get("client_id")
        tok = data.get("access_token")
        if cid and tok:
            os.environ.setdefault("DHAN_CLIENT_ID", str(cid))
            os.environ.setdefault("DHAN_ACCESS_TOKEN", str(tok))
            return True
    except Exception:
        return False
    return False


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
    creds_ok = load_creds_from_state()
    if not creds_ok:
        raise RuntimeError("Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN or ensure state/creds.json exists.")

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
    market = RollingExpiredOptionsMarket()

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
