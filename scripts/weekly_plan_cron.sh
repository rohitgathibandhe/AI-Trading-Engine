#!/usr/bin/env bash
set -euo pipefail

# Location of the rolling option cache
ROLLING_DIR="data_engine/market_ai/state/rolling_option"
OUTPUT_CSV="reports/intraday_from_rolling_latest.csv"
PLAN_PATH="state/weekly_plan.json"
INTENT_PATH="data_engine/market_ai/state/order_intents.jsonl"

echo "[weekly-plan] collecting last 260 trade dates from ${ROLLING_DIR}"
DATES=$(python3 - <<'PY'
from pathlib import Path
root = Path("data_engine/market_ai/state/rolling_option")
dirs = sorted(d.name for d in root.iterdir() if d.is_dir())
if len(dirs) > 260:
    dirs = dirs[-260:]
print(",".join(dirs))
PY
)

echo "[weekly-plan] building latest intraday dataset -> ${OUTPUT_CSV}"
python3 data_engine/market_ai/scripts/make_intraday_dataset.py \
    --dates "$DATES" \
    --output "${OUTPUT_CSV}"

echo "[weekly-plan] generating weekly plan + intent"
python3 data_engine/market_ai/scripts/generate_weekly_plan.py \
    --input "${OUTPUT_CSV}" \
    --plan-path "${PLAN_PATH}" \
    --intent-path "${INTENT_PATH}" \
    --qty 2 \
    --hybrid \
    --min-prev-range 0.003 \
    --max-prev-range 0.05 \
    --pnl-target 6000 \
    --pnl-stop 4000 \
    --structure STRANGLE \
    --wing-offset 4 \
    --trend-threshold 0.02 \
    --condor-threshold 0.012 \
    --oi-distance 0.012 \
    --expiry-offset 1 \
    --min-days-to-expiry 4 \
    --ml-exit-model "data_engine/market_ai/state/weekly_exit_model.pkl" \
    --ml-exit-threshold 0.9

echo "[weekly-plan] done – plan written to ${PLAN_PATH}"
