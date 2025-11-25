"""
Convenience wrapper: load agent_settings.json and run the weekly backtest with
regime gating (SIDEWAYS -> strangle).
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "data_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from market_ai.strategies.weekly_regime_classifier import RegimeConfig, classify_week  # noqa: E402

SETTINGS = ROOT / "data_engine" / "market_ai" / "state" / "agent_settings.json"
BACKTEST = ROOT / "data_engine" / "market_ai" / "scripts" / "run_weekly_theta_backtest.py"
DEFAULT_INPUT = ROOT / "reports" / "intraday_from_rolling_latest.csv"
OUTPUT_DIR = ROOT / "reports" / "weekly_theta_backtest"


def main() -> None:
    try:
        settings = json.loads(SETTINGS.read_text())
    except Exception:
        settings = {}
    lot_size = int(settings.get("nifty_lot_size", settings.get("lot_size", 75)))
    qty = int(settings.get("max_legs", 1))
    # hedge settings exported for future use; backtest currently ignores hedges
    env = {}
    env.update({
        "WEEKLY_HEDGE_DISTANCE": str(settings.get("weekly_hedge_distance", 300.0)),
        "WEEKLY_HEDGE_PRICE_CAP": str(settings.get("weekly_hedge_price_cap", 6.0)),
    })
    # Classify regime per trade date and filter to SIDEWAYS before backtest
    try:
        df = pd.read_csv(DEFAULT_INPUT)
    except Exception:
        df = None
    if df is not None and not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["trade_date"] = df["timestamp"].dt.date
        decisions = {}
        cfg = RegimeConfig()
        for date, day in df.groupby("trade_date"):
            # basic features for classification
            open_spot = day.iloc[0]["spot"]
            prev_close = day.iloc[0].get("prev_day_close") or open_spot
            gap_pct = (open_spot - prev_close) / prev_close * 100 if prev_close else 0.0
            rng = day["spot"].max() - day["spot"].min()
            rng_pct = (rng / open_spot) * 100 if open_spot else 0.0
            trend_20 = day.iloc[0].get("spot_trend_20") or 0.0
            dec = classify_week(gap_pct, rng_pct, trend_20, cfg)
            decisions[date] = dec
        # keep only SIDEWAYS dates
        keep_dates = [d for d, dec in decisions.items() if dec["regime"] == "SIDEWAYS" and dec["score"] >= cfg.min_score]
        df = df[df["trade_date"].isin(keep_dates)]
        filtered_input = ROOT / "reports" / "intraday_from_rolling_sideways.csv"
        df.to_csv(filtered_input, index=False)
        input_path = filtered_input
    else:
        input_path = DEFAULT_INPUT

    cmd = [
        "python3",
        str(BACKTEST),
        "--input",
        str(input_path),
        "--output-dir",
        str(OUTPUT_DIR),
        "--lot-size",
        str(lot_size),
        "--qty",
        str(qty),
        "--min-premium",
        "0.01",
    ]
    print("Running:", " ".join(cmd))
    out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env={**env, **dict()})
    print(out.stdout)
    if out.stderr:
        print(out.stderr)


if __name__ == "__main__":
    main()
