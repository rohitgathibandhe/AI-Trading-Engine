# AI Trading Engine — Claude Instructions

## Git Workflow (REQUIRED)

**Never commit directly to `main`.** Always follow this flow:

1. **Create a feature branch** before making any changes:
   ```
   git checkout -b fix/short-description   # for bug fixes
   git checkout -b feat/short-description  # for new features
   git checkout -b chore/short-description # for config/tooling changes
   ```
2. **Commit changes** to the feature branch.
3. **Push the branch** to remote: `git push -u origin <branch-name>`
4. **Open a PR** into `main` using `gh pr create`.
5. Report the PR URL wrapped in `<pr-created>...</pr-created>`.

If already on `main` with uncommitted changes, stash them, create a branch, then apply the stash before committing.

## Project Context

- **Language:** Python 3.10 (venv: `/Users/Rohit/.venvs/data_engine_py310`)
- **Working dir:** `/Users/Rohit/AI-Trading-Engine`
- **Agent module:** `data_engine/market_ai/`
- **State files:** `data_engine/market_ai/state/`
- **Run agent:** `TRADE_MODE=paper python -m market_ai.intraday_defined_risk.cli run_live --config data_engine/market_ai/state/intraday_v83_run_live_config.json`
- **Agent log:** `data_engine/market_ai/state/intraday_v83_runner.log`
- **Settings:** `data_engine/market_ai/state/agent_settings.json`
- **Runtime config:** `data_engine/market_ai/state/intraday_v83_runtime_config.json`
