"""
Intraday Theta Harvesting (WIP)
==============================

Goal
----
Daily, intraday-only options strategy that:
1. Targets at least ₹1,000 realized profit per session with a trailing stop to lock gains.
2. Never carries exposure past 15:15 IST (hard flatten).
3. Chooses entries using intraday option-chain signals + underlying chart context.

Why this lives in its own module:
- The monthly strangle engine is large and tuned for swing trades; intraday needs different
  data cadence, risk controls, and order flow.
- We can iterate faster by keeping configs/backtests separate, then later wire the agent UI.

MVP Requirements (tracked as TODOs below):
- Intraday data replay (1–5 minute candles + option chain snapshots/greeks or
  OHLC + derived features) for backtests.
- Strategy config with:
    * allowed trade window (e.g., 09:20–15:15)
    * instrument universe (ATM straddles, OTMs, hedges)
    * entry filters (IV change, delta band, trend filter)
    * daily target, trailing stop %, hard SL, per-trade max loss
- Execution simulator that:
    * Applies brokerage, slippage
    * Tracks cumulative MTM to enforce daily target/trail
    * Forces flatten orders at cut-off time
- `run_backtest` / `run_live` adapters consistent with the rest of market_ai.

Implementation Status
---------------------
- Skeleton dataclasses + adapters below
- No actual signal logic yet; this file is the home for future implementation.

Recommended next steps
----------------------
1. finalize data contract (which columns the replay engine provides).
2. build the intraday `IntradaySimSession` that feeds ticks to the strategy.
3. implement base playbook (e.g., ATM short straddle with hedge + MTM trail).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from market_ai.modules.sim import IntradayReplaySession, ReplayConfig

__all__ = [
    "IntradayThetaScalp",
    "IntradayConfig",
    "DailyTargetConfig",
    "EntryFilterConfig",
    "run_backtest",
    "run_live",
]


@dataclass
class DailyTargetConfig:
    min_profit: float = 1_000.0          # INR target before trail starts
    trail_pct: float = 0.25              # give-back allowed once target hit (25% of peak)
    hard_loss_per_day: float = 5_000.0   # flatten + disable after this loss
    flatten_time: time = time(15, 10)    # final exit command, done by 15:15


@dataclass
class EntryFilterConfig:
    min_iv_rank: float = 0.20
    max_iv_rank: float = 0.80
    min_premium_pct: float = 0.018       # combined call+put / spot
    trend_lookback_min: int = 15         # minutes for VWAP/EMA slope checks (reserved)
    spot_atm_band: float = 0.003         # +/- band to define “ATM” for straddle entries
    max_abs_delta: float = 0.80
    max_trend_pct: float = 0.002         # reject entries if |spot - MA| / MA exceeds
    max_trend_slope: float = 0.0005      # reject entries if MA slope (per bar) exceeds
    enable_directional_overrides: bool = True


@dataclass
class IntradayConfig:
    underlying_id: int = 13                      # NIFTY default
    underlying_seg: str = "NSE_FNO"
    lot_size: int = 50
    capital: float = 500_000.0
    trade_window: Tuple[time, time] = (time(9, 20), time(15, 10))
    rebalance_interval_min: int = 1
    entry_filter: EntryFilterConfig = field(default_factory=EntryFilterConfig)
    daily_target: DailyTargetConfig = field(default_factory=DailyTargetConfig)
    max_concurrent_positions: int = 1
    use_weekly_hedges: bool = True
    slippage_bps: float = 5.0
    fees_per_leg: float = 20.0
    warn_only: bool = False
    notes: Optional[str] = None
    timezone: str = "Asia/Kolkata"
    max_entries_per_day: int = 2
    risk_stop_pct: float = 0.35          # exit if debit >= entry_credit * (1 + risk_stop_pct)
    per_leg_stop_pct: float = 0.40       # exit if any leg loses 40% vs entry
    max_hold_minutes: int = 90
    hedge_spot_pct: float = 0.004        # add hedge when |spot-entry| >= this %
    hedge_cost_fixed: float = 250.0
    max_hedges_per_trade: int = 1


@dataclass
class PositionState:
    entry_time: datetime
    entry_call: float
    entry_put: float
    qty: int
    lot_size: int
    entry_spot: Optional[float]
    hedges: int = 0

    def entry_credit(self) -> float:
        return (self.entry_call + self.entry_put) * self.qty * self.lot_size

    def current_debit(self, call_ltp: float, put_ltp: float) -> float:
        return (call_ltp + put_ltp) * self.qty * self.lot_size

    def unrealized(self, call_ltp: float, put_ltp: float) -> float:
        return self.entry_credit() - self.current_debit(call_ltp, put_ltp)


class IntradayThetaScalp:
    """
    Core intraday strategy (logic WIP).

    Expected integration:
    - `step(session_state)` gets invoked on every new tick/candle from the simulator or
       live market adapter.
    - The strategy decides to open/adjust/close positions while respecting the config.
    """

    REQUIRED_BAR_FIELDS = ("timestamp", "atm_call_ltp", "atm_put_ltp")

    def __init__(self, cfg: IntradayConfig):
        self.cfg = cfg
        self._tz = ZoneInfo(cfg.timezone)
        self.state: Dict[str, Any] = {}
        self._reset_session(None)

    # ------------------------------------------------------------------ helpers
    def _reset_session(self, session_date: Optional[datetime.date]) -> None:
        self.state = {
            "session_date": session_date,
            "position": None,
            "daily_realized": 0.0,
            "daily_peak_pnl": 0.0,
            "target_triggered": False,
            "disabled": False,
            "events": [],
            "entries": 0,
            "spot_window": [],
            "equity_curve": [],
        }

    @staticmethod
    def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
        if val in (None, "", [], {}, "nan"):
            return default
        try:
            out = float(val)
            if out != out:  # NaN check
                return default
            return out
        except Exception:
            return default

    def _to_local(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=self._tz)
        return ts.astimezone(self._tz)

    def _within_trade_window(self, ts: datetime) -> bool:
        local_ts = self._to_local(ts)
        start, end = self.cfg.trade_window
        if local_ts.time() < start:
            return False
        if local_ts.time() > end:
            return False
        return True

    def _bar_missing_required_fields(self, bar: Dict[str, Any]) -> bool:
        for field in self.REQUIRED_BAR_FIELDS:
            if bar.get(field) is None:
                return True
        return False

    def _open_position(self, ts: datetime, bar: Dict[str, Any]) -> Dict[str, Any]:
        qty = 1  # TODO: sizing logic
        pos = PositionState(
            entry_time=ts,
            entry_call=float(bar["atm_call_ltp"]),
            entry_put=float(bar["atm_put_ltp"]),
            qty=qty,
            lot_size=self.cfg.lot_size,
            entry_spot=self._safe_float(bar.get("spot")),
        )
        self.state["position"] = pos
        self.state["entries"] = self.state.get("entries", 0) + 1
        event = {
            "action": "OPEN",
            "side": "SHORT_STRADDLE",
            "timestamp": ts,
            "call_strike": bar.get("atm_call_strike"),
            "put_strike": bar.get("atm_put_strike"),
            "call_ltp": pos.entry_call,
            "put_ltp": pos.entry_put,
            "notes": "auto-entry",
        }
        return event

    def _close_position(self, ts: datetime, bar: Dict[str, Any], reason: str) -> Dict[str, Any]:
        pos: PositionState = self.state["position"]
        call = float(bar["atm_call_ltp"])
        put = float(bar["atm_put_ltp"])
        realized = pos.unrealized(call, put)
        self.state["daily_realized"] += realized
        self.state["position"] = None
        self.state["daily_peak_pnl"] = max(self.state["daily_peak_pnl"], self.state["daily_realized"])
        event = {
            "action": "CLOSE",
            "reason": reason,
            "timestamp": ts,
            "call_ltp": call,
            "put_ltp": put,
            "realized": realized,
        }
        return event

    def _current_equity(self, unrealized: float) -> float:
        return self.state["daily_realized"] + unrealized

    # ------------------------------------------------------------------ hooks
    def on_new_bar(self, bar: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        ts: Optional[datetime] = bar.get("timestamp")
        if ts is None:
            return events
        local_ts = self._to_local(ts)

        session_date = local_ts.date()
        if self.state.get("session_date") != session_date:
            self._reset_session(session_date)

        if self._bar_missing_required_fields(bar):
            return events

        if self.state["disabled"]:
            return events

        if not self._within_trade_window(ts):
            return events

        position: Optional[PositionState] = self.state["position"]
        call_ltp = float(bar["atm_call_ltp"])
        put_ltp = float(bar["atm_put_ltp"])
        spot_val = self._safe_float(bar.get("spot"))
        self._append_spot_window(spot_val)

        if position is None:
            if self.state["entries"] >= self.cfg.max_entries_per_day:
                return events
            if not self._entry_allowed(bar):
                return events
            event = self._open_position(local_ts, bar)
            self.state["events"].append(event)
            events.append(event)
            return events

        unrealized = position.unrealized(call_ltp, put_ltp)
        equity = self._current_equity(unrealized)
        self.state["daily_peak_pnl"] = max(self.state["daily_peak_pnl"], equity)
        self._record_equity(local_ts, equity)

        hedge_event = self._maybe_add_hedge(local_ts, position, spot_val)
        if hedge_event:
            events.append(hedge_event)

        daily_cfg = self.cfg.daily_target
        trail_floor = self.state["daily_peak_pnl"] * (1 - daily_cfg.trail_pct)
        flatten_time = daily_cfg.flatten_time
        reasons: List[str] = []

        if equity >= daily_cfg.min_profit:
            self.state["target_triggered"] = True
        if self.state["target_triggered"] and equity <= trail_floor:
            reasons.append("trail")
        if equity <= -daily_cfg.hard_loss_per_day:
            reasons.append("daily_loss")
        if local_ts.time() >= flatten_time:
            reasons.append("flatten_time")

        stop_pct = max(0.0, float(self.cfg.risk_stop_pct))
        if stop_pct > 0 and position.current_debit(call_ltp, put_ltp) >= position.entry_credit() * (1 + stop_pct):
            reasons.append("risk_stop")
        per_leg_stop = max(0.0, float(self.cfg.per_leg_stop_pct))
        if per_leg_stop > 0 and (
            call_ltp >= position.entry_call * (1 + per_leg_stop)
            or put_ltp >= position.entry_put * (1 + per_leg_stop)
        ):
            reasons.append("leg_stop")
        if self.cfg.max_hold_minutes > 0:
            minutes_open = (local_ts - position.entry_time).total_seconds() / 60.0
            if minutes_open >= self.cfg.max_hold_minutes:
                reasons.append("max_hold")

        if reasons:
            event = self._close_position(local_ts, bar, reason="|".join(reasons))
            self.state["events"].append(event)
            events.append(event)
            self.state["disabled"] = True

        return events

    def _entry_allowed(self, bar: Dict[str, Any]) -> bool:
        filt = self.cfg.entry_filter
        premium_pct = self._safe_float(bar.get("combined_premium_pct"))
        if premium_pct is None or premium_pct < filt.min_premium_pct:
            return False
        iv_rank = self._safe_float(bar.get("iv_rank"))
        if iv_rank is not None and not (filt.min_iv_rank <= iv_rank <= filt.max_iv_rank):
            return False
        call_delta = abs(self._safe_float(bar.get("atm_call_delta"), 0.0) or 0.0)
        put_delta = abs(self._safe_float(bar.get("atm_put_delta"), 0.0) or 0.0)
        if call_delta > filt.max_abs_delta or put_delta > filt.max_abs_delta:
            return False
        if filt.max_trend_pct > 0:
            spot = self._safe_float(bar.get("spot"))
            trend = self._spot_trend()
            if spot is not None and trend is not None and trend:
                drift = abs((spot - trend) / trend)
                if drift > filt.max_trend_pct:
                    return False
        slope = self._spot_slope()
        if slope is not None and abs(slope) > filt.max_trend_slope:
            return False
        return True

    def _append_spot_window(self, spot: Optional[float]) -> None:
        if spot is None:
            return
        window = self.state.get("spot_window", [])
        window.append(spot)
        max_len = max(1, int(self.cfg.entry_filter.trend_lookback_min))
        if len(window) > max_len:
            window = window[-max_len:]
        self.state["spot_window"] = window

    def _spot_trend(self) -> Optional[float]:
        window = self.state.get("spot_window") or []
        if not window:
            return None
        return sum(window) / len(window)

    def _spot_slope(self) -> Optional[float]:
        window = self.state.get("spot_window") or []
        if len(window) < 2:
            return None
        first, last = window[0], window[-1]
        if not first:
            return None
        steps = max(len(window) - 1, 1)
        return (last - first) / abs(first) / steps

    def _maybe_add_hedge(
        self,
        ts: datetime,
        position: Optional[PositionState],
        spot_val: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        if position is None or spot_val is None:
            return None
        cfg = self.cfg
        if cfg.max_hedges_per_trade <= 0 or position.hedges >= cfg.max_hedges_per_trade:
            return None
        entry_spot = position.entry_spot or spot_val
        drift = abs(spot_val - entry_spot) / max(entry_spot, 1.0)
        if drift < cfg.hedge_spot_pct:
            return None
        cost = cfg.hedge_cost_fixed * position.qty
        position.hedges += 1
        position.entry_spot = spot_val
        self.state["daily_realized"] -= cost
        hedge_event = {
            "timestamp": ts,
            "action": "HEDGE",
            "cost": cost,
            "spot": spot_val,
            "reason": "spot_drift",
        }
        self.state["events"].append(hedge_event)
        return hedge_event

    def _record_equity(self, ts: datetime, equity: float) -> None:
        self.state.setdefault("equity_curve", []).append({"timestamp": ts, "equity": equity})

    def on_finish(self, final_bar: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if self.state.get("position") is None or final_bar is None:
            return events
        ts = self._to_local(final_bar["timestamp"]) if final_bar.get("timestamp") is not None else datetime.now(self._tz)
        event = self._close_position(ts, final_bar, reason="eod_flatten")
        self.state["events"].append(event)
        events.append(event)
        return events

    def on_fill(self, fill: Dict[str, Any]) -> None:
        """
        Hook for executed orders so we can update running P&L/positions.
        """
        # TODO: update self.state with fills.
        pass

    def enforce_daily_rules(self, now: datetime) -> Optional[Dict[str, Any]]:
        """
        Example helper that would emit a flatten intent if the daily
        profit target is hit (with trailing) or if we're near cut-off time.
        """
        # TODO: implement trailing logic
        return None


# --- adapters -----------------------------------------------------------------

def _coerce_config(cfg: Dict[str, Any]) -> IntradayConfig:
    data = dict(cfg)
    if isinstance(data.get("daily_target"), dict):
        data["daily_target"] = DailyTargetConfig(**data["daily_target"])
    if isinstance(data.get("entry_filter"), dict):
        data["entry_filter"] = EntryFilterConfig(**data["entry_filter"])
    return IntradayConfig(**data)


def _summarize_events(events_df: pd.DataFrame, strat) -> Dict[str, Any]:
    entries = (events_df["action"] == "OPEN").sum() if "action" in events_df else 0
    closes = (events_df["action"] == "CLOSE").sum() if "action" in events_df else 0
    total_realized = float(events_df.loc[events_df["action"] == "CLOSE", "realized"].sum()) if "realized" in events_df else 0.0
    equity_curve = strat.state.get("equity_curve", []) if hasattr(strat, "state") else []

    def _drawdown(curve: List[Dict[str, Any]]) -> float:
        peak = float("-inf")
        max_dd = 0.0
        for point in curve:
            equity = float(point.get("equity", 0.0))
            peak = max(peak, equity)
            if peak > 0:
                max_dd = min(max_dd, equity - peak)
        return max_dd

    summary = {
        "entries": int(entries),
        "closes": int(closes),
        "total_realized": total_realized,
        "max_drawdown": _drawdown(equity_curve),
        "note": "Intraday simulator prototype – rules are placeholder sizing/entries.",
    }
    return summary


def run_backtest(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Backtest harness placeholder. Once the intraday simulator exists, this
    will replay `df` and return (trades_df, metrics_df, summary).
    """
    config_obj = _coerce_config(cfg)
    strat = IntradayThetaScalp(config_obj)
    replay = IntradayReplaySession(df, ReplayConfig(include_columns=list(df.columns)))
    result = replay.run(strat)
    trades_df = pd.DataFrame(result["events"])
    metrics_df = pd.DataFrame([result["summary"]])
    summary = _summarize_events(trades_df, strat)
    return trades_df, metrics_df, summary


def run_live(*args, **kwargs):
    """
    Live adapter placeholder to keep API parity with other strategies.
    """
    raise NotImplementedError("Intraday live orchestration not implemented yet.")
