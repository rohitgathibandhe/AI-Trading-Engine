#!/usr/bin/env bash
set -euo pipefail

# Usage: weekly_plan_cron.sh [min_prev_range] [pnl_target] [pnl_stop] [directional_enabled(0/1)] [max_gap_pct]
MIN_PREV_RANGE_ARG=${1:-}
PNL_TARGET_ARG=${2:-}
PNL_STOP_ARG=${3:-}
DIRECTIONAL_ENABLED_ARG=${4:-}
MAX_GAP_PCT_ARG=${5:-}

# Location of the rolling option cache
ROLLING_DIR="data_engine/market_ai/state/rolling_option"
OUTPUT_CSV="reports/intraday_from_rolling_latest.csv"
PLAN_PATH="state/weekly_plan.json"
INTENT_PATH="data_engine/market_ai/state/order_intents.jsonl"
SETTINGS_JSON="state/weekly_plan_settings.json"

if [ -f "${SETTINGS_JSON}" ]; then
    eval "$(python3 - <<'PY'
import json
from pathlib import Path
cfg = {}
path = Path("state/weekly_plan_settings.json")
if path.exists():
    try:
        cfg = json.loads(path.read_text()) or {}
    except Exception:
        cfg = {}
print(f"JSON_MIN_PREV_RANGE={cfg.get('min_prev_range','')}")
print(f"JSON_PNL_TARGET={cfg.get('pnl_target','')}")
print(f"JSON_PNL_STOP={cfg.get('pnl_stop','')}")
print(f"JSON_DIRECTIONAL_ENABLED={int(bool(cfg.get('directional_enabled', True)))}")
print(f"JSON_MAX_GAP_PCT={cfg.get('max_gap_pct','')}")
PY
)"
fi

MIN_PREV_RANGE=${MIN_PREV_RANGE_ARG:-${JSON_MIN_PREV_RANGE:-0.003}}
PNL_TARGET=${PNL_TARGET_ARG:-${JSON_PNL_TARGET:-6000}}
PNL_STOP=${PNL_STOP_ARG:-${JSON_PNL_STOP:-4000}}
DIRECTIONAL_ENABLED=${DIRECTIONAL_ENABLED_ARG:-${JSON_DIRECTIONAL_ENABLED:-1}}
MAX_GAP_PCT=${MAX_GAP_PCT_ARG:-${JSON_MAX_GAP_PCT:-0.01}}

collect_option_chain() {
    local cid="$1"
    local token="$2"
    if [ -z "$cid" ] || [ -z "$token" ]; then
        echo "[weekly-plan] skipping option-chain snapshot (missing creds)"
        return
    fi
    echo "[weekly-plan] collecting live option chain snapshot"
    DHAN_CLIENT_ID="$cid" DHAN_ACCESS_TOKEN="$token" PYTHONPATH="data_engine" \
        python3 data_engine/market_ai/scripts/collect_live_option_chain.py >> logs/option_chain.log 2>&1 || \
        echo "[weekly-plan] option-chain collector failed"
}

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
    --min-prev-range "${MIN_PREV_RANGE}" \
    --max-prev-range 0.05 \
    --pnl-target "${PNL_TARGET}" \
    --pnl-stop "${PNL_STOP}" \
    --structure STRANGLE \
    --wing-offset 4 \
    --trend-threshold 0.02 \
    --condor-threshold 0.012 \
    --oi-distance 0.012 \
    --expiry-offset 1 \
    --min-days-to-expiry 4 \
    --max-gap-pct "${MAX_GAP_PCT}" \
    --directional-enabled "${DIRECTIONAL_ENABLED}" \
    --ml-exit-model "data_engine/market_ai/state/weekly_exit_model.pkl" \
    --ml-exit-threshold 0.9

echo "[weekly-plan] done – plan written to ${PLAN_PATH}"
collect_option_chain "${DHAN_CLIENT_ID:-}" "${DHAN_ACCESS_TOKEN:-}"
