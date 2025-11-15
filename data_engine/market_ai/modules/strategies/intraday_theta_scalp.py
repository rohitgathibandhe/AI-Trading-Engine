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
    second_target_mult: float = 1.5      # second target = min_profit * mult
    lock_in_pct: float = 0.15            # once second target hit, keep at least 15%


@dataclass
class EntryFilterConfig:
    min_iv_rank: float = 0.20
    max_iv_rank: float = 0.80
    min_premium_pct: float = 0.016       # combined call+put / spot
    trend_lookback_min: int = 15         # minutes for VWAP/EMA slope checks
    trend_long_lookback_min: int = 45
    spot_atm_band: float = 0.003
    max_abs_delta: float = 0.80
    max_trend_pct: float = 0.002
    max_trend_slope: float = 0.0005
    directional_slope_trigger: float = 0.0007
    range_slope_threshold: float = 0.00015
    atr_vol_threshold: float = 0.0025
    iv_skew_trend: float = 0.08
    oi_skew_trend: float = 0.08
    range_skew_max: float = 0.04
    support_res_pct: float = 0.001
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
    max_entries_per_day: int = 3
    risk_stop_pct: float = 0.35          # exit if debit >= entry_credit * (1 + risk_stop_pct)
    per_leg_stop_pct: float = 0.30       # exit if any leg loses 30% vs entry
    max_hold_minutes: int = 90
    hedge_spot_pct: float = 0.0035        # add hedge when |spot-entry| >= this %
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
    call_active: bool = True
    put_active: bool = True
    structure: str = "STRANGLE"
    best_equity: float = 0.0
    hit_first_target: bool = False
    hit_second_target: bool = False

    def entry_credit(self) -> float:
        credit = 0.0
        if self.call_active:
            credit += self.entry_call
        if self.put_active:
            credit += self.entry_put
        return credit * self.qty * self.lot_size

    def current_debit(self, call_ltp: float, put_ltp: float) -> float:
        debit = 0.0
        if self.call_active:
            debit += call_ltp
        if self.put_active:
            debit += put_ltp
        return debit * self.qty * self.lot_size

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
            "spot_window_long": [],
            "atr_window": [],
            "intraday_high": None,
            "intraday_low": None,
            "equity_curve": [],
            "_prev_spot": None,
            "last_regime": "unknown",
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

    def _open_position(self, ts: datetime, bar: Dict[str, Any], structure: str) -> Dict[str, Any]:
        qty = 1  # TODO: sizing logic
        call_active = structure != "SHORT_PUT"
        put_active = structure != "SHORT_CALL"
        call_entry = float(bar["atm_call_ltp"]) if call_active else 0.0
        put_entry = float(bar["atm_put_ltp"]) if put_active else 0.0
        pos = PositionState(
            entry_time=ts,
            entry_call=call_entry,
            entry_put=put_entry,
            qty=qty,
            lot_size=self.cfg.lot_size,
            entry_spot=self._safe_float(bar.get("spot")),
            call_active=call_active,
            put_active=put_active,
            structure=structure,
        )
        self.state["position"] = pos
        self.state["entries"] = self.state.get("entries", 0) + 1
        event = {
            "action": "OPEN",
            "side": structure,
            "timestamp": ts,
            "call_strike": bar.get("atm_call_strike"),
            "put_strike": bar.get("atm_put_strike"),
            "call_ltp": pos.entry_call,
            "put_ltp": pos.entry_put,
            "notes": f"auto-entry|regime={self.state.get('last_regime')}",
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
            "structure": pos.structure,
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
        self._update_spot_stats(spot_val)
        self.state["_last_iv_skew"] = self._safe_float(bar.get("iv_skew"))
        self.state["_last_oi_skew"] = self._safe_float(bar.get("oi_skew"))

        if position is None:
            if self.state["entries"] >= self.cfg.max_entries_per_day:
                return events
            if not self._entry_allowed(bar):
                return events
            structure = self._determine_structure()
            if not structure:
                return events
            event = self._open_position(local_ts, bar, structure)
            self.state["events"].append(event)
            events.append(event)
            return events

        unrealized = position.unrealized(call_ltp, put_ltp)
        equity = self._current_equity(unrealized)
        self.state["daily_peak_pnl"] = max(self.state["daily_peak_pnl"], equity)
        self._record_equity(local_ts, equity)

        position.best_equity = max(position.best_equity, equity)
        daily_cfg = self.cfg.daily_target
        first_target = daily_cfg.min_profit
        second_target = first_target * max(1.0, daily_cfg.second_target_mult)
        if equity >= first_target:
            position.hit_first_target = True
        if equity >= second_target:
            position.hit_second_target = True

        hedge_event = self._maybe_add_hedge(local_ts, position, spot_val)
        if hedge_event:
            events.append(hedge_event)

        daily_cfg = self.cfg.daily_target
        flatten_time = daily_cfg.flatten_time
        reasons: List[str] = []

        self.state["target_triggered"] = position.hit_first_target
        dynamic_floor: Optional[float] = None
        if position.hit_first_target:
            dynamic_floor = position.best_equity * (1 - daily_cfg.trail_pct)
        if position.hit_second_target:
            lock_floor = position.best_equity * (1 - daily_cfg.lock_in_pct)
            dynamic_floor = max(dynamic_floor or lock_floor, lock_floor)
        if dynamic_floor is not None and equity <= dynamic_floor:
            reasons.append("trail")
        if equity <= -daily_cfg.hard_loss_per_day:
            reasons.append("daily_loss")
        if local_ts.time() >= flatten_time:
            reasons.append("flatten_time")

        stop_pct = max(0.0, float(self.cfg.risk_stop_pct))
        if stop_pct > 0 and position.current_debit(call_ltp, put_ltp) >= position.entry_credit() * (1 + stop_pct):
            reasons.append("risk_stop")
        per_leg_stop = max(0.0, float(self.cfg.per_leg_stop_pct))
        leg_stop_hit = False
        if per_leg_stop > 0:
            if position.call_active and position.entry_call > 0:
                if call_ltp >= position.entry_call * (1 + per_leg_stop):
                    leg_stop_hit = True
            if position.put_active and position.entry_put > 0:
                if put_ltp >= position.entry_put * (1 + per_leg_stop):
                    leg_stop_hit = True
        if leg_stop_hit:
            reasons.append("leg_stop")
        if self.cfg.max_hold_minutes > 0:
            minutes_open = (local_ts - position.entry_time).total_seconds() / 60.0
            if minutes_open >= self.cfg.max_hold_minutes:
                reasons.append("max_hold")

        if reasons:
            reason_str = "|".join(reasons)
            event = self._close_position(local_ts, bar, reason=reason_str)
            self.state["events"].append(event)
            events.append(event)
            if "daily_loss" in reasons or local_ts.time() >= flatten_time:
                self.state["disabled"] = True
            elif self.state.get("entries", 0) >= self.cfg.max_entries_per_day:
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
        atr_norm = self._spot_atr()
        if atr_norm is not None and atr_norm > filt.atr_vol_threshold:
            return False
        hi = self.state.get("intraday_high")
        lo = self.state.get("intraday_low")
        spot = self._safe_float(bar.get("spot"))
        if spot is not None and hi is not None and lo is not None and hi != lo:
            buffer = filt.support_res_pct * spot
            if abs(spot - hi) <= buffer or abs(spot - lo) <= buffer:
                return False
        iv_skew_val = self._safe_float(bar.get("iv_skew"))
        if iv_skew_val is not None and abs(iv_skew_val) > max(0.5, filt.range_skew_max * 5):
            return False
        return True

    def _update_spot_stats(self, spot: Optional[float]) -> None:
        if spot is None:
            return
        short_len = max(1, int(self.cfg.entry_filter.trend_lookback_min))
        long_len = max(short_len, int(self.cfg.entry_filter.trend_long_lookback_min))

        def _append(name: str, length: int) -> None:
            buf = self.state.get(name, [])
            buf.append(spot)
            if len(buf) > length:
                buf = buf[-length:]
            self.state[name] = buf

        _append("spot_window", short_len)
        _append("spot_window_long", long_len)

        prev = self.state.get("_prev_spot")
        if prev is not None:
            tr = abs(spot - prev)
            atr_window = self.state.get("atr_window", [])
            atr_window.append(tr)
            if len(atr_window) > short_len:
                atr_window = atr_window[-short_len:]
            self.state["atr_window"] = atr_window
        self.state["_prev_spot"] = spot

        hi = self.state.get("intraday_high")
        lo = self.state.get("intraday_low")
        self.state["intraday_high"] = max(spot, hi or spot)
        self.state["intraday_low"] = min(spot, lo or spot)

    def _spot_trend(self, window_name: str = "spot_window") -> Optional[float]:
        window = self.state.get(window_name) or []
        if not window:
            return None
        return sum(window) / len(window)

    def _spot_slope(self, window_name: str = "spot_window") -> Optional[float]:
        window = self.state.get(window_name) or []
        if len(window) < 2:
            return None
        first, last = window[0], window[-1]
        if not first:
            return None
        steps = max(len(window) - 1, 1)
        return (last - first) / abs(first) / steps

    def _spot_atr(self) -> Optional[float]:
        atr_window = self.state.get("atr_window") or []
        if not atr_window:
            return None
        spot_window = self.state.get("spot_window") or []
        spot = spot_window[-1] if spot_window else None
        if spot in (None, 0):
            return None
        return (sum(atr_window) / len(atr_window)) / abs(spot)

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

    def _infer_regime(self) -> str:
        filt = self.cfg.entry_filter
        slope_short = self._spot_slope("spot_window") or 0.0
        slope_long = self._spot_slope("spot_window_long") or 0.0
        atr_norm = self._spot_atr() or 0.0
        iv_skew = self._safe_float(self.state.get("_last_iv_skew"), 0.0) or 0.0
        oi_skew = self._safe_float(self.state.get("_last_oi_skew"), 0.0) or 0.0

        if atr_norm > filt.atr_vol_threshold * 1.5:
            regime = "volatile"
            self.state["last_regime"] = regime
            return regime

        trigger = filt.directional_slope_trigger
        skew_trigger_iv = filt.iv_skew_trend
        skew_trigger_oi = filt.oi_skew_trend

        if slope_short >= trigger or slope_long >= trigger or iv_skew >= skew_trigger_iv or oi_skew >= skew_trigger_oi:
            regime = "trend_up"
            self.state["last_regime"] = regime
            return regime
        if slope_short <= -trigger or slope_long <= -trigger or iv_skew <= -skew_trigger_iv or oi_skew <= -skew_trigger_oi:
            regime = "trend_down"
            self.state["last_regime"] = regime
            return regime

        if atr_norm > filt.atr_vol_threshold:
            regime = "volatile"
            self.state["last_regime"] = regime
            return regime

        if abs(slope_short) <= filt.range_slope_threshold and abs(iv_skew) <= filt.range_skew_max:
            regime = "range"
        else:
            regime = "neutral"
        self.state["last_regime"] = regime
        return regime

    def _determine_structure(self) -> Optional[str]:
        regime = self._infer_regime()
        if regime == "trend_up":
            return "SHORT_PUT"
        if regime == "trend_down":
            return "SHORT_CALL"
        if regime == "volatile":
            return None
        return "STRANGLE"

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
