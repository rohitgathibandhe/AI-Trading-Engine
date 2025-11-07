"""
Tiny helper to exercise strategy backtests inside tests without standing up the
full live engine.  The harness intentionally stays minimal: feed synthetic spot
series + option-chain snapshots into any `run_backtest`-style function and
collect the summary dict it returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

import pandas as pd


@dataclass
class _SequenceMarket:
    """Cycles through a list of option-chain snapshots."""

    chain_snapshots: List[dict]

    def get_option_chain(
        self,
        underlying_id: int,
        expiry: str,
        as_of_date: Optional[str] = None,
    ) -> dict:
        if not self.chain_snapshots:
            return {}
        snapshot = self.chain_snapshots.pop(0)
        self.chain_snapshots.append(snapshot)
        return snapshot


def run_daily_backtest(
    strategy_fn: Callable[[pd.DataFrame, dict], Any],
    date: str,
    spot_series: Iterable[float] | Iterable[dict],
    option_chain_series: Iterable[dict],
    *,
    lot_size: int = 50,
) -> dict:
    """
    Execute ``strategy_fn`` (e.g. ``run_backtest``) for a single day using the
    provided synthetic timeseries, and return the resulting summary dict.

    Parameters
    ----------
    strategy_fn:
        Callable accepting ``(prices_df, cfg)`` and returning the standard
        ``(trades_df, _, summary_dict)`` tuple.
    date:
        ISO date string used for all rows in ``spot_series``.
    spot_series:
        Iterable of floats or dicts; floats map to close prices. Dicts should
        supply ``{"close": value}``.
    option_chain_series:
        Iterable of option-chain payloads consumed sequentially by the stub
        market adapter.
    lot_size:
        Optional override passed to the strategy config.
    """

    rows: List[dict] = []
    for idx, item in enumerate(spot_series):
        if isinstance(item, dict):
            close = float(item.get("close", 0.0))
        else:
            close = float(item)
        rows.append({"date": pd.to_datetime(date).date(), "close": close, "idx": idx})
    prices_df = pd.DataFrame(rows)[["date", "close"]]

    market = _SequenceMarket(list(option_chain_series))
    cfg = {"lot_size": lot_size, "market": market}

    trades_df, _, summary = strategy_fn(prices_df, cfg)
    summary = dict(summary)
    summary["n_trades"] = len(trades_df)
    return summary
