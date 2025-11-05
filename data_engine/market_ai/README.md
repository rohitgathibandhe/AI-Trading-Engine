# market_ai / data_engine

Backtesting & research framework for NIFTY options strategies (option-selling / hedged sells).
Modular design so production and research components are clearly separated.

## Quick overview

**Main entry (CLI)**:
- `ai_strategy_backtest.py` — orchestrates backtests (reads predictions CSV or uses option-chain).
- `modules/` — small modules:
  - `dhan_api.py` — Dhan API helpers (option chain, expiries, account/margin).
  - `vix_utils.py` — VIX estimation / fallback.
  - `sizing.py` — position sizing / lot-size lookup.
  - `trade_logic.py` — strategy rules (sell/hedge logic).
  - `backtest_engine.py` — engine functions used by CLI.

**Data / outputs**
- `market_data.db` — local sqlite for lot sizes & cached metadata (DO NOT version control — add to .gitignore).
- `predictions_latest.csv` — model output (must contain `pred` or `prob_pos`).
- outputs: `backtest_trades.csv`, `backtest_returns.csv`, `backtest_equity.png`.

## Getting started (local)

1. Create venv and install:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
