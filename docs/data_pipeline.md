# Data Pipeline (Rolling Option ➜ Feature Store)

This job keeps the agent’s historical dataset and selector model fresh by:

1. Pulling the latest rolling-option candles from Dhan (chunked to respect their 30‑day window limit) across a default ladder of `ATM`, `ATM±2`, `ATM±4`, `ATM±6`.
2. Regenerating `state/feature_history.csv`, engineered feature parquet, targets, and dataset summaries.
3. Retraining the selector (ridge regression per strategy) and writing metrics consumed by the dashboard.
4. Surfacing coverage + training health in the Observability tab.

## Prerequisites

```bash
export DHAN_CLIENT_ID=xxxxxxxxxx
export DHAN_ACCESS_TOKEN=your_live_token
```

## Ad-hoc run

```bash
python3 data_engine/market_ai/scripts/run_data_pipeline.py \
  NIFTY \
  --security-id 13 \
  --seg NSE_FNO \
  --lookback-days 30 \
  --option-type CALL \
  --option-type PUT
```

Artifacts written under `data_engine/market_ai/state/`:

| File | Purpose |
| --- | --- |
| `rolling_option/` | Day partitions with `rolling_option.parquet` + raw JSONL |
| `rolling_option_summary.json` | Coverage + IV/OI completeness (rendered in dashboard) |
| `feature_history.csv` | Combined live + historical samples (now ~258k rows) |
| `feature_history_engineered.parquet` | Ready-to-train feature matrix |
| `feature_targets.parquet` | Provisional targets (net credit shifted 5 steps) |
| `feature_history_summary.json` | Training stats, counts per strategy, date span |
| `strategy_selector_model.json` | Latest selector weights/intercepts |
| `strategy_selector_training_summary.json` | RMSE/R² per strategy (rendered in dashboard) |

## Cron / launchd example

Run every weekday at 7:30 AM IST (convert to your server’s timezone):

```cron
30 2 * * 1-5 cd /Users/Rohit/AI-Trading-Engine && \
  /usr/bin/env DHAN_CLIENT_ID=xxx DHAN_ACCESS_TOKEN=yyy \
  /usr/bin/python3 data_engine/market_ai/scripts/run_data_pipeline.py \
    NIFTY --security-id 13 --seg NSE_FNO --lookback-days 30 \
    --option-type CALL --option-type PUT >> logs/data_pipeline.log 2>&1
```

Use `launchctl` on macOS if you prefer GUI scheduling; point it to the same command.

## Troubleshooting

- `DH-905` errors mean the requested date range exceeds the API window; the pipeline already chunks to 30 days, so verify system clocks and provided start/end overrides.
- `DH-901` means the access token expired; generate a fresh one and re-export the env vars.
- If `feature_history_summary.json` shows `days_missing` growing, re-run the pipeline with a larger `--lookback-days` to backfill older sessions.
- Use `--skip-training` / `--skip-refresh` when debugging individual stages or if you only need a data refresh without retraining.
