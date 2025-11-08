# market_ai/modules/backtest_engine_adapter.py
"""
Robust adapter between CLI (ai_strategy_backtest) and strategy modules.

This adapter:
 - Imports strategy module by trying a few candidate package paths
 - Calls strategy.run_backtest(df, cfg) in a tolerant way
 - Centralizes the skip_sells toggle (cfg may include '_skip_sells_due_to_vix' or 'skip_sells')
 - IMPORTANT: VIX handling removed (Option A). Adapter does not fetch or gate by VIX.
"""
from __future__ import annotations
import importlib
import inspect
import logging
from typing import Any, Dict, Tuple

LOG = logging.getLogger("market_ai.backtest_adapter")
LOG.addHandler(logging.NullHandler())


class AdapterError(RuntimeError):
    pass


def import_strategy(strategy_name: str):
    """
    Try several import paths and return the module. Raise AdapterError on failure
    with helpful diagnostics.
    """
    candidates = [
        f"market_ai.modules.strategies.{strategy_name}",
        f"market_ai.modules.{strategy_name}",
        strategy_name,
    ]
    errs = []
    for c in candidates:
        try:
            mod = importlib.import_module(c)
            LOG.debug("Imported strategy module %s -> %s", strategy_name, c)
            return mod
        except Exception as e:
            errs.append((c, type(e).__name__, str(e)))
    raise AdapterError(f"Could not import strategy '{strategy_name}'. Tried: {errs}")


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs that the callable accepts (positional/keyword).
    """
    sig = inspect.signature(fn)
    out = {}
    for name, p in sig.parameters.items():
        if name in kwargs and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            out[name] = kwargs[name]
    return out


def _call_runbacktest(fn, df, cfg: Dict[str, Any], extra_kwargs: Dict[str, Any]) -> Tuple:
    """
    Try to call provided function in a tolerant way:
      preferred: fn(df, cfg, **extra_kwargs)
      fallback: fn(df, cfg), fn(df), fn(cfg), fn(**filtered_kwargs)
    Returns whatever the strategy returns (expected: trades_df, returns_df, summary)
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    LOG.debug("Calling strategy function %s signature=%s", getattr(fn, "__name__", str(fn)), sig)

    # Attempt (df, cfg, **extra)
    try:
        # If function accepts at least two args, call with df and cfg (and filtered extra kwargs)
        if len(params) >= 2:
            filtered = _filter_kwargs_for_callable(fn, extra_kwargs)
            try:
                return fn(df, cfg, **filtered)
            except TypeError:
                # maybe the strategy doesn't accept the extra kwargs -> call without them
                return fn(df, cfg)
        # If function expects one parameter, heuristically decide which
        if len(params) == 1:
            p0 = params[0].lower()
            if p0 in ("df", "data", "predictions", "prices"):
                filtered = _filter_kwargs_for_callable(fn, extra_kwargs)
                if filtered:
                    return fn(df, **filtered)
                return fn(df)
            if p0 in ("cfg", "config", "options"):
                return fn(cfg)
        # No params or unknown signature: try calling with filtered kwargs
        filtered = _filter_kwargs_for_callable(fn, extra_kwargs)
        if filtered:
            return fn(**filtered)
        # final fallback attempts
        try:
            return fn(df, cfg)
        except TypeError:
            return fn(df)
    except TypeError as te:
        raise AdapterError(f"Strategy run_backtest TypeError: {te}") from te
    except Exception:
        LOG.exception("Strategy function raised exception")
        raise


def run_backtest(df, cfg: Dict[str, Any]):
    """
    Public adapter entrypoint.

    df: DataFrame (predictions + price/time columns)
    cfg: dict containing at least:
      - 'strategy' (name)
      - optional strategy-specific keys (lot_size, capital, etc.)
      - '_skip_sells_due_to_vix' or 'skip_sells' (adapter will forward to strategy)
    Returns whatever strategy returns (expected tuple of (trades_df, returns_df, summary))
    """
    LOG.info("Adapter run_backtest called for strategy=%s", cfg.get("strategy"))
    strategy_name = cfg.get("strategy") or cfg.get("strategy_name")
    if not strategy_name:
        raise AdapterError("cfg must include 'strategy' key")

    # import strategy module
    strategy_mod = import_strategy(strategy_name)

    # find run callable (run_backtest preferred)
    fn = None
    for cand in ("run_backtest", "backtest", "execute", "run"):
        if hasattr(strategy_mod, cand):
            fn = getattr(strategy_mod, cand)
            LOG.debug("Using strategy callable '%s' from module %s", cand, strategy_mod.__name__)
            break
    if fn is None:
        raise AdapterError(f"Strategy module {strategy_mod.__name__} has no run_backtest/backtest/execute/run callable")

    # centralize skip_sells toggle (user may set _skip_sells_due_to_vix or skip_sells)
    skip_sells = bool(cfg.get("_skip_sells_due_to_vix", cfg.get("skip_sells", False)))
    # Prepare extra kwargs to pass to strategy in a filtered way
    extra_kwargs = dict(cfg)  # copy - strategy adapter will filter what it needs
    extra_kwargs["skip_sells"] = skip_sells

    LOG.debug("Adapter forwarding extra_kwargs keys: %s", list(sorted(extra_kwargs.keys())))

    try:
        result = _call_runbacktest(fn, df, cfg, extra_kwargs)
        LOG.info("Strategy returned result type %s", type(result))
        return result
    except AdapterError:
        LOG.exception("Adapter error while running strategy")
        raise
    except Exception as e:
        LOG.exception("Unhandled exception running strategy")
        raise AdapterError("Unhandled exception in run_backtest") from e
