# AI-Trading-Engine

## Weekly Theta Automation

We now have a weekly theta engine (see `weekly_theta_strangle.py`) plus two CLIs:

1. `run_weekly_theta_backtest.py` – ad-hoc backtests / tuning.
2. `generate_weekly_plan.py` – generates the next-week plan and appends an order intent.

### Cron / Paper-Trade Workflow

Use `scripts/weekly_plan_cron.sh` to rebuild the latest intraday dataset and emit the weekly plan + intent:

```bash
./scripts/weekly_plan_cron.sh
```

This script:
1. Collects the last ~12 months of trade dates from `data_engine/market_ai/state/rolling_option`.
2. Rebuilds `reports/intraday_from_rolling_latest.csv`.
3. Calls `generate_weekly_plan.py` with the tuned hybrid preset (strangle for trend weeks, iron condor for tight ranges) and appends an intent to `data_engine/market_ai/state/order_intents.jsonl`.

Add the script to crontab (run before market open on Mondays):

```
45 8 * * 1 cd /path/to/AI-Trading-Engine && ./scripts/weekly_plan_cron.sh >> logs/weekly_plan.log 2>&1
```

Latest plan lives in `state/weekly_plan.json`. Review it before market open and, once satisfied, let your paper/live adapter consume the appended intent.

### CLI Tuning Examples

Baseline (hybrid strangle/condor, 12‑month dataset, 2 lots):

```bash
python3 data_engine/market_ai/scripts/run_weekly_theta_backtest.py \
  --input reports/intraday_from_rolling_12m.csv \
  --qty 2 --min-prev-range 0.005 --max-prev-range 0.05 \
  --pnl-target 4000 --pnl-stop 4000 \
  --hybrid --trend-threshold 0.02 --condor-threshold 0.012 --oi-distance 0.012 \
  --wing-offset 4 --emit-plan reports/weekly_theta_plan.json
```

Results (Sep 2024–Nov 2025): 59 weekly trades, total realized ≈ ₹3.2 L (₹5.4 k avg), monthly P&L ranges from ₹12 k to ₹41 k.

Other presets tested (all 12‑month runs, qty = 2):

| Wing Offset | Condor Range Threshold | OI Distance | Total Realized | Avg / trade |
|-------------|-----------------------|-------------|----------------|-------------|
| 3           | 0.010                 | 0.010       | ₹3.28 L        | ₹5.56 k     |
| 4 (baseline)| 0.012                 | 0.012       | ₹3.20 L        | ₹5.43 k     |
| 5           | 0.015                 | 0.015       | ₹3.28 L        | ₹5.56 k     |

Feel free to adjust thresholds/wing offsets and rerun the CLI to compare equity curves. The emitted `reports/weekly_theta_plan*.json` captures each run’s summary for later review.
