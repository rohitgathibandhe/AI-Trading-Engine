from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

DEFAULT_TIME_COL = "timestamp"


@dataclass
class ReplayConfig:
    """
    Basic knobs for intraday data replay.

    Attributes
    ----------
    flatten_time: time
        Hard cut-off; any open exposure should be closed at or before this
        timestamp. The simulator merely records the intent; strategy must
        honour it until execution modeling is implemented.
    time_col: str
        Column name containing timezone-aware timestamps.
    include_columns: Optional[List[str]]
        Extra columns to carry into each bar dict (e.g., iv_rank, chain stats).
    """

    flatten_time: time = time(15, 10)
    time_col: str = DEFAULT_TIME_COL
    include_columns: Optional[List[str]] = None
    warn_on_gaps: bool = True
    dropna: bool = True
    sort: bool = True


class IntradayReplaySession:
    """
    Lightweight iterator that feeds intraday bars to a strategy instance.

    This is intentionally minimal right now: it does **not** simulate fills or
    slippage. The goal is to provide a consistent data contract so we can hook
    in richer execution logic later.
    """

    def __init__(self, df: pd.DataFrame, cfg: Optional[ReplayConfig] = None):
        if df is None or df.empty:
            raise ValueError("Intraday replay requires a non-empty DataFrame.")
        self.cfg = cfg or ReplayConfig()
        data = df.copy()
        if self.cfg.dropna:
            data = data.dropna(subset=[self.cfg.time_col])
        if self.cfg.sort:
            data = data.sort_values(self.cfg.time_col)
        # ensure datetime objects
        data[self.cfg.time_col] = pd.to_datetime(data[self.cfg.time_col], utc=True)
        self.data = data.reset_index(drop=True)

    def iterate(self) -> Iterable[Dict[str, Any]]:
        cols = list(self.data.columns)
        for row in self.data.itertuples(index=False):
            payload: Dict[str, Any] = {}
            for col, value in zip(cols, row):
                payload[col] = value
            if self.cfg.include_columns:
                payload = {k: payload.get(k) for k in self.cfg.include_columns if k in payload}
            yield payload

    def run(self, strategy) -> Dict[str, Any]:
        """
        Pushes every bar to strategy.on_new_bar(...).
        Collects emitted intents/fills for future analysis.
        Returns a dict with 'events' and placeholder 'summary'.
        """
        events: List[Dict[str, Any]] = []
        for bar in self.iterate():
            ts = bar.get(self.cfg.time_col)
            if ts is None:
                continue
            intents = strategy.on_new_bar(bar) or []
            for intent in intents:
                events.append({"timestamp": ts, **intent})
        summary = {
            "events": len(events),
            "flatten_time": self.cfg.flatten_time.isoformat(),
        }
        return {
            "events": events,
            "summary": summary,
        }

