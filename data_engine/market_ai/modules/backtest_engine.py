"""
Fallback backtest engine.

Provides a single entrypoint:

    run_backtest(pred_df: pd.DataFrame, cfg: Dict[str, Any]) -> (trades_df, returns_df, summary)

The function attempts to call a strategy's run_backtest() in a tolerant way, while providing
standardised configuration defaults and input normalization.

Design goals:
- Robust import of strategy modules.
- Normalise predictions CSV → `pred_df` (date column -> datetime, pred/prob column names).
- Keep defaults configurable and avoid hard-coded capital/lot counts inside this file.
- Support multiple strategy.run_backtest signatures.
- If strategy missing or fails, return a clean placeholder result.
"""
from __future__ import annotations

import importlib
import logging
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def _import_strategy(strategy_name: str):
    """Try a set of module import paths for strategy_name"""
    attempts = [
        f"market_ai.modules.strategies.{strategy_name}",
        strategy_name,
    ]
    for nm in attempts:
        try:
            return importlib.import_module(nm)
        except Exception:
            log.debug("Strategy import attempt failed for %s", nm, exc_info=True)
            continue
    raise ModuleNotFoundError(f"No strategy module found for {strategy_name}. Tried: {attempts}")


def _normalise_pred_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure pred_df has sensible columns:
      - 'date' as pd.Timestamp (if present and parseable).
      - 'pred' column (from 'pred_signal' or similar).
      - 'prob' column (if any of prob/prob_pos/prob_score).
    """
    df = df.copy()
    # normalize date
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            # try infer
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # try index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        else:
            raise ValueError("pred_df must contain a 'date' column or datetime index")

    # normalize predictions column names
    if "pred" not in df.columns:
        if "pred_signal" in df.columns:
            df = df.rename(columns={"pred_signal": "pred"})
        else:
            # try other names
            for candidate in ("prediction", "signal", "pred_signal_raw"):
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "pred"})
                    break

    # force pred to numeric/binary where possible (but keep strings if strategy expects them)
    # if pred looks like 'BUY'/'SELL' etc., keep as-is
    # else convert to int
    if "pred" in df.columns:
        try:
            if df["pred"].dtype.kind in ("i", "f"):
                df["pred"] = df["pred"].astype(int)
        except Exception:
            # leave as-is (string labels)
            pass

    # find probability column
    prob_candidates = ["prob", "prob_pos", "prob_score", "probability"]
    prob_col = None
    for c in prob_candidates:
        if c in df.columns:
            prob_col = c
            break
    if prob_col and prob_col != "prob":
        df = df.rename(columns={prob_col: "prob"})
        prob_col = "prob"

    # ensure ordering by date ascending - many strategies expect chronological order
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def _default_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return a config copied and merged with sane defaults. Do not mutate caller dict.
    Keys added/ensured:
      - capital: float
      - lot_size: int
      - lots: int      # configurable number of lots
      - lot_cost_per_unit: float
      - hold_days: int
      - skip_vix: bool
      - skip_sells: bool (compat)
      - mode: 'sell'|'buy'
      - strategy: str (must be supplied by caller)
    """
    base = dict(cfg or {})
    defaults = {
        "capital": float(base.get("capital", 0.0)),        # user must provide or strategy should read from broker
        "lot_size": int(base.get("lot_size", base.get("market_lot", 75))),
        "lots": int(base.get("lots", 1)),
        "lot_cost_per_unit": float(base.get("lot_cost_per_unit", base.get("lot_cost", 0.0))),
        "hold_days": int(base.get("hold_days", 1)),
        # Skip VIX gate/logic centrally (user override)
        "skip_vix": bool(base.get("skip_vix", base.get("skip_vix_due_to_data", True))),
        # compatibility with older keys
        "skip_sells": bool(base.get("skip_sells", False)),
        "mode": base.get("mode", "sell"),
        "prob_threshold": float(base.get("prob_threshold", 0.6)),
        "use_optionchain": bool(base.get("use_optionchain", False)),
        "underlying_id": base.get("underlying_id", None),
        # leave any other values intact
    }
    # merge defaults back into base without overwriting user keys
    for k, v in defaults.items():
        base.setdefault(k, v)
    return base


def run_backtest(pred_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run a strategy backtest tolerant to various strategy signatures.

    Returns:
      (trades_df, returns_df, summary_dict)
    """
    # Validate strategy
    strategy_name = cfg.get("strategy") or cfg.get("strategy_name")
    if not strategy_name:
        raise ValueError("cfg must include a 'strategy' key (strategy module name)")

    # Normalize CFG & Data
    cfg_clean = _default_cfg(cfg)
    try:
        df = _normalise_pred_df(pred_df)
    except Exception as e:
        log.error("Failed to normalise preds dataframe: %s", e)
        raise

    log.debug("Backtest engine running strategy=%s with cfg=%s", strategy_name, {k: cfg_clean.get(k) for k in ("capital", "lot_size", "lots", "hold_days", "skip_vix")})

    # Import strategy module (attempt multiple import paths)
    try:
        strat_mod = _import_strategy(strategy_name)
    except ModuleNotFoundError as e:
        log.error("Could not import strategy %s: %s", strategy_name, e)
        # fallback empty results
        trades_df = pd.DataFrame(columns=["entry_date", "exit_date", "pnl"])
        returns_df = pd.DataFrame([{"date": d, "equity": cfg_clean["capital"]} for d in pd.to_datetime(df["date"].unique())])
        summary = {"initial_capital": cfg_clean["capital"], "final_equity": cfg_clean["capital"], "n_trades": 0, "total_net_profit": 0.0}
        return trades_df, returns_df, summary

    # Instantiate Strategy object if present
    strat_obj = None
    if hasattr(strat_mod, "Strategy"):
        try:
            # try named argument style
            strat_obj = strat_mod.Strategy(config=cfg_clean, capital=cfg_clean["capital"])
        except Exception:
            try:
                strat_obj = strat_mod.Strategy(cfg_clean, cfg_clean["capital"])
            except Exception:
                log.debug("Strategy class exists but could not be instantiated with common signatures; falling back to module-level functions", exc_info=True)
                strat_obj = strat_mod
    else:
        strat_obj = strat_mod

    # Determine skip flags to pass in
    skip_sells_flag = bool(cfg_clean.get("skip_sells", False) or cfg_clean.get("skip_sells_due_to_vix", False))
    # If user requested to completely skip VIX checks centrally, set vix_used to None
    vix_used = None if bool(cfg_clean.get("skip_vix", True)) else cfg_clean.get("vix_used", None)

    # Try calling strategy.run_backtest with various signatures
    run_fn = getattr(strat_obj, "run_backtest", None)
    if callable(run_fn):
        # try most feature rich call first
        try:
            log.debug("Attempting run_backtest(pred_df, cfg, skip_sells=..., vix_used=...)")
            res = run_fn(df, cfg_clean, skip_sells=skip_sells_flag, vix_used=vix_used)
            log.debug("Strategy.run_backtest returned (signature 1)")
        except TypeError:
            try:
                log.debug("Attempting run_backtest(pred_df, cfg)")
                res = run_fn(df, cfg_clean)
                log.debug("Strategy.run_backtest returned (signature 2)")
            except TypeError:
                try:
                    log.debug("Attempting run_backtest(cfg)")
                    res = run_fn(cfg_clean)
                    log.debug("Strategy.run_backtest returned (signature 3)")
                except Exception as e:
                    log.exception("strategy.run_backtest could not be invoked with known signatures")
                    raise RuntimeError("strategy.run_backtest could not be called with expected signatures") from e

        # validate result
        if isinstance(res, tuple) and len(res) == 3:
            trades_df, returns_df, summary = res
            # ensure types (best-effort)
            if trades_df is None:
                trades_df = pd.DataFrame(columns=["entry_date", "exit_date", "pnl"])
            if returns_df is None:
                returns_df = pd.DataFrame([{"date": d, "equity": cfg_clean["capital"]} for d in pd.to_datetime(df["date"].unique())])
            if summary is None:
                summary = {"initial_capital": cfg_clean["capital"], "final_equity": cfg_clean["capital"], "n_trades": 0, "total_net_profit": 0.0}
            # Ensure dataframes are real pd.DataFrame
            trades_df = pd.DataFrame(trades_df) if not isinstance(trades_df, pd.DataFrame) else trades_df
            returns_df = pd.DataFrame(returns_df) if not isinstance(returns_df, pd.DataFrame) else returns_df
            # Done
            log.info("Strategy backtest finished: trades=%s rows, returns=%s rows", len(trades_df), len(returns_df))
            return trades_df, returns_df, summary
        else:
            log.warning("strategy.run_backtest returned unexpected shape/type. Expected (trades_df, returns_df, summary). Got: %s", type(res))
            # Build fallback
            trades_df = pd.DataFrame(columns=["entry_date", "exit_date", "pnl"])
            returns_df = pd.DataFrame([{"date": d, "equity": cfg_clean["capital"]} for d in pd.to_datetime(df["date"].unique())])
            summary = {"initial_capital": cfg_clean["capital"], "final_equity": cfg_clean["capital"], "n_trades": 0, "total_net_profit": 0.0}
            return trades_df, returns_df, summary

    # If the module/class has no run_backtest, return a placeholder result
    log.warning("Strategy %s has no run_backtest; returning placeholder empty result", strategy_name)
    trades_df = pd.DataFrame(columns=["entry_date", "exit_date", "pnl"])
    returns_df = pd.DataFrame([{"date": d, "equity": cfg_clean["capital"]} for d in pd.to_datetime(df["date"].unique())])
    summary = {"initial_capital": cfg_clean["capital"], "final_equity": cfg_clean["capital"], "n_trades": 0, "total_net_profit": 0.0}
    return trades_df, returns_df, summary
