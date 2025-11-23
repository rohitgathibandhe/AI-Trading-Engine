"""
Convenience wrapper: load agent_settings.json and run the weekly theta backtest
with those settings (lot size, qty, hedge distance/price caps if used later).
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
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
        "WEEKLY_HEDGE_DISTANCE": str(settings.get("weekly_hedge_distance", 200.0)),
        "WEEKLY_HEDGE_PRICE_CAP": str(settings.get("weekly_hedge_price_cap", 3.5)),
    })
    cmd = [
        "python3",
        str(BACKTEST),
        "--input",
        str(DEFAULT_INPUT),
        "--output-dir",
        str(OUTPUT_DIR),
        "--lot-size",
        str(lot_size),
        "--qty",
        str(qty),
    ]
    print("Running:", " ".join(cmd))
    out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env={**env, **dict()})
    print(out.stdout)
    if out.stderr:
        print(out.stderr)


if __name__ == "__main__":
    main()
