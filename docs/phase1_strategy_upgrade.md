# Phase 1 – Environment-aware Strategy Selection

## Goals
- Derive richer market context (trend, volatility regime, breadth proxies) from index price history and option chain snapshots.
- Feed the context into a new StrategyRecommender that picks the best strategy configuration for the current regime.
- Surface regime diagnostics in logs and the Strategy Monitor.

## Components

### MarketRegimeAnalyzer (`market_ai/modules/analytics/market_state.py`)
- Inputs: intraday index price series (OHLC), rolling realized volatility, VIX/IV metrics, option-chain aggregates.
- Outputs: `MarketState` dataclass containing trend label, volatility regime, IV rank, breadth score, and supporting metrics.
- Implementation (Phase 1): heuristic-based using EMAs, ATR/price ratio, IV percentile, and option-chain open-interest skew. Future phases can swap in ML models.

### StrategyRecommender (`market_ai/modules/agents/strategy_recommender.py`)
- Combines MarketState with (optional) selector model scores (`modules/strategies/strategy_selector.py`).
- Produces ranked `StrategyCandidate` objects with rationale and risk hints (delta targets, position sizing adjustments).
- Provides fallbacks so the agent continues trading even if model scores are stale.

### Agent Integration
- `start_agent.py` instantiates MarketRegimeAnalyzer and StrategyRecommender.
- On each loop (or at a configurable cadence), the agent:
  1. Updates market state (fetch or compute OHLC snapshot + VIX/chain data).
  2. Requests strategy recommendation.
  3. Stores the decision context in feature history + logs.
  4. Chooses entry logic based on recommendation (Phase 1: just log recommendation, continue running default strategy unless selector flips).

### UI Enhancements (Phase 1 scope)
- Strategy Monitor shows:
  - Current trend & volatility regime.
  - IV rank / realized volatility metrics.
  - Top 3 strategy recommendations with confidence and rationale.

## Future Hooks (Phase 2+)
- Portfolio risk ledger integration (exposure-based throttling).
- ML classifier for regime detection (trained via stored history).
- Scenario testing / Monte Carlo before entries.

## Data Contracts
- MarketState serialized as JSON when appended to feature history: `context.market_state = {...}`.
- Strategy recommendations included in feature context and logs for auditability.

## Risks & Mitigations
- **Data availability**: if OHLC or VIX data missing, fallback to last known state and log warning.
- **Performance**: heuristics designed to run in <10ms. Long-term we may batch compute features and cache.
- **Backwards compatibility**: default strategy continues to run if recommender fails, preventing trading downtime.

