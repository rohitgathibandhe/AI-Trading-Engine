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

import json
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from market_ai.modules.sim import IntradayReplaySession, ReplayConfig

__all__ = [
    "IntradayThetaScalp",
    "IntradayConfig",
    "DailyTargetConfig",
    "EntryFilterConfig",
    "StructureWingConfig",
    "EmaConfig",
    "PositionSizingConfig",
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
    oi_level_buffer_pct: float = 0.0015
    prev_day_buffer_pct: float = 0.001
    oi_level_min_gap: float = 50.0
    block_near_prev_levels: bool = False
    block_near_oi_levels: bool = False
    directional_confirm_mult: float = 1.2
    directional_min_skew: float = 0.02
    atr_vol_hard_cap: float = 0.0035
    max_intraday_range_pct: float = 0.015
    breakout_buffer_pct: float = 0.0005
    breakout_prevday_buffer_pct: float = 0.0008
    inside_day_threshold: float = 0.8
    vwap_band_pct: float = 0.0008


@dataclass
class StructureWingConfig:
    strangle_call_offset: int = 0
    strangle_put_offset: int = 0
    short_call_offset: int = 0
    short_put_offset: int = 0
    bear_call_short_offset: int = 0
    bear_call_long_offset: int = 4
    bull_put_short_offset: int = 0
    bull_put_long_offset: int = -4
    iron_fly_call_short_offset: int = 0
    iron_fly_call_long_offset: int = 4
    iron_fly_put_short_offset: int = 0
    iron_fly_put_long_offset: int = -4
    iron_condor_call_short_offset: int = 2
    iron_condor_call_long_offset: int = 6
    iron_condor_put_short_offset: int = -2
    iron_condor_put_long_offset: int = -6


@dataclass
class PositionSizingConfig:
    strangle_qty: int = 2
    spread_qty: int = 2
    iron_qty: int = 1
    base_qty: int = 1
    max_qty: int = 4
    capital_pct_limit: float = 0.75
    margin_pct: float = 0.09
    aggressive_bonus: int = 1
    drawdown_trigger: float = 0.0
    drawdown_penalty: int = 0
    aggressive_trigger: float = 900.0
    loss_pause_trigger: float = 0.0


@dataclass
class EmaConfig:
    fast_period: int = 20
    mid_period: int = 50
    slow_period: int = 100
    breakout_short_period: int = 15
    breakout_long_period: int = 30


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
    event_calendar_path: Optional[str] = "data_engine/market_ai/state/events.json"
    event_skip_levels: Tuple[str, ...] = ("high",)
    event_light_levels: Tuple[str, ...] = ("medium",)
    structure_offsets: StructureWingConfig = field(default_factory=StructureWingConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)


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
    structure_cost: float = 0.0
    call_field: Optional[str] = None
    put_field: Optional[str] = None
    long_call_entry: float = 0.0
    long_put_entry: float = 0.0
    long_call_field: Optional[str] = None
    long_put_field: Optional[str] = None

    def entry_credit(self) -> float:
        credit = 0.0
        if self.call_active:
            credit += self.entry_call
        if self.put_active:
            credit += self.entry_put
        if self.long_call_field and self.long_call_entry:
            credit -= self.long_call_entry
        if self.long_put_field and self.long_put_entry:
            credit -= self.long_put_entry
        return credit * self.qty * self.lot_size

    def current_debit(
        self,
        call_ltp: float,
        put_ltp: float,
        long_call_ltp: Optional[float] = None,
        long_put_ltp: Optional[float] = None,
    ) -> float:
        debit = 0.0
        if self.call_active:
            debit += call_ltp
        if self.put_active:
            debit += put_ltp
        if self.long_call_field and self.long_call_entry:
            debit -= (long_call_ltp or 0.0)
        if self.long_put_field and self.long_put_entry:
            debit -= (long_put_ltp or 0.0)
        return debit * self.qty * self.lot_size

    def unrealized(
        self,
        call_ltp: float,
        put_ltp: float,
        long_call_ltp: Optional[float] = None,
        long_put_ltp: Optional[float] = None,
    ) -> float:
        return self.entry_credit() - self.current_debit(call_ltp, put_ltp, long_call_ltp, long_put_ltp)


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
        self._event_calendar = self._load_event_calendar(cfg.event_calendar_path)
        self._reset_session(None)

    # ------------------------------------------------------------------ helpers
    def _reset_session(self, session_date: Optional[datetime.date]) -> None:
        max_entries = self.cfg.max_entries_per_day
        directive = None
        if session_date is not None:
            directive = self._event_directive(session_date)
            if directive == "reduce":
                max_entries = min(max_entries, 1)
            elif directive == "skip":
                max_entries = 0
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
            "_prev_call_volume": None,
            "_prev_put_volume": None,
            "vwap_numerator": 0.0,
            "vwap_denominator": 0.0,
            "spot_vwap": None,
            "last_regime": "unknown",
            "event_directive": directive,
            "max_entries_today": max_entries,
            "structure_hint": None,
            "ema_fast": None,
            "ema_mid": None,
            "ema_slow": None,
            "hi_window_short": [],
            "hi_window_long": [],
            "lo_window_short": [],
            "lo_window_long": [],
            "inside_day": False,
            "context_state": "unknown",
            "breakout_flags": {},
        }
        if directive == "skip":
            self.state["disabled"] = True
        if directive:
            self.state.setdefault("events", []).append(
                {
                    "action": "EVENT",
                    "timestamp": datetime.now(self._tz),
                    "directive": directive,
                    "notes": f"calendar={directive}",
                }
            )

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

    @staticmethod
    def _selector_field(option_type: str, offset: int, metric: str) -> str:
        opt = option_type.lower()
        if offset == 0:
            prefix = f"atm_{opt}"
        elif offset > 0:
            prefix = f"{opt}_atm_plus{abs(offset)}"
        else:
            prefix = f"{opt}_atm_minus{abs(offset)}"
        return f"{prefix}_{metric}"

    def _leg_price(self, bar: Dict[str, Any], field: Optional[str]) -> Optional[float]:
        if not field:
            return None
        return self._safe_float(bar.get(field))

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

    def _open_position(self, ts: datetime, bar: Dict[str, Any], structure: str) -> Optional[Dict[str, Any]]:
        legs = self._prepare_structure(structure, bar)
        if not legs:
            return None
        structure = legs.get("structure", structure)
        qty = self._structure_qty(structure, legs, bar)
        call_field = legs.get("call_field")
        put_field = legs.get("put_field")
        long_call_field = legs.get("long_call_field")
        long_put_field = legs.get("long_put_field")
        call_price = legs.get("call_price") or 0.0
        put_price = legs.get("put_price") or 0.0
        long_call_price = legs.get("long_call_price") or 0.0
        long_put_price = legs.get("long_put_price") or 0.0
        call_strike = legs.get("call_strike")
        put_strike = legs.get("put_strike")
        long_call_strike = legs.get("long_call_strike")
        long_put_strike = legs.get("long_put_strike")
        call_active = call_field is not None
        put_active = put_field is not None
        structure_cost = self._structure_cost(structure, qty)
        if structure_cost:
            self.state["daily_realized"] -= structure_cost
        pos = PositionState(
            entry_time=ts,
            entry_call=call_price if call_active else 0.0,
            entry_put=put_price if put_active else 0.0,
            qty=qty,
            lot_size=self.cfg.lot_size,
            entry_spot=self._safe_float(bar.get("spot")),
            call_active=call_active,
            put_active=put_active,
            structure=structure,
            structure_cost=structure_cost,
            call_field=call_field,
            put_field=put_field,
            long_call_entry=long_call_price if long_call_field else 0.0,
            long_put_entry=long_put_price if long_put_field else 0.0,
            long_call_field=long_call_field,
            long_put_field=long_put_field,
        )
        self.state["position"] = pos
        self.state["entries"] = self.state.get("entries", 0) + 1
        event = {
            "action": "OPEN",
            "side": structure,
            "timestamp": ts,
            "call_strike": call_strike,
            "put_strike": put_strike,
            "long_call_strike": long_call_strike,
            "long_put_strike": long_put_strike,
            "call_field": call_field,
            "put_field": put_field,
            "long_call_field": long_call_field,
            "long_put_field": long_put_field,
            "call_ltp": pos.entry_call,
            "put_ltp": pos.entry_put,
            "long_call_ltp": long_call_price if long_call_field else None,
            "long_put_ltp": long_put_price if long_put_field else None,
            "structure_cost": structure_cost,
            "notes": f"auto-entry|regime={self.state.get('last_regime')}",
        }
        return event

    def _close_position(self, ts: datetime, bar: Dict[str, Any], reason: str) -> Dict[str, Any]:
        pos: PositionState = self.state["position"]
        call = self._leg_price(bar, pos.call_field) if pos and pos.call_active else 0.0
        put = self._leg_price(bar, pos.put_field) if pos and pos.put_active else 0.0
        long_call = self._leg_price(bar, pos.long_call_field) if pos and pos.long_call_field else None
        long_put = self._leg_price(bar, pos.long_put_field) if pos and pos.long_put_field else None
        if pos and pos.long_call_field and long_call is None:
            long_call = pos.long_call_entry
        if pos and pos.long_put_field and long_put is None:
            long_put = pos.long_put_entry
        call = call or 0.0
        put = put or 0.0
        realized = pos.unrealized(call, put, long_call, long_put)
        self.state["daily_realized"] += realized
        self.state["position"] = None
        self.state["daily_peak_pnl"] = max(self.state["daily_peak_pnl"], self.state["daily_realized"])
        event = {
            "action": "CLOSE",
            "reason": reason,
            "timestamp": ts,
            "call_ltp": call,
            "put_ltp": put,
            "long_call_ltp": long_call,
            "long_put_ltp": long_put,
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
        spot_val = self._safe_float(bar.get("spot"))
        self._update_spot_stats(bar)
        self.state["_last_iv_skew"] = self._safe_float(bar.get("iv_skew"))
        self.state["_last_oi_skew"] = self._safe_float(bar.get("oi_skew"))

        if position is None:
            if self.state["entries"] >= self._max_entries_today():
                return events
            if not self._entry_allowed(bar):
                return events
            structure = self._determine_structure()
            if not structure:
                return events
            event = self._open_position(local_ts, bar, structure)
            if not event:
                return events
            self.state["events"].append(event)
            events.append(event)
            return events

        call_ltp = self._leg_price(bar, position.call_field) if position.call_active else 0.0
        put_ltp = self._leg_price(bar, position.put_field) if position.put_active else 0.0
        long_call_ltp = self._leg_price(bar, position.long_call_field)
        long_put_ltp = self._leg_price(bar, position.long_put_field)
        if position.long_call_field and long_call_ltp is None:
            long_call_ltp = position.long_call_entry
        if position.long_put_field and long_put_ltp is None:
            long_put_ltp = position.long_put_entry
        call_ltp = call_ltp or 0.0
        put_ltp = put_ltp or 0.0

        unrealized = position.unrealized(call_ltp, put_ltp, long_call_ltp, long_put_ltp)
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
        if stop_pct > 0 and position.current_debit(call_ltp, put_ltp, long_call_ltp, long_put_ltp) >= position.entry_credit() * (1 + stop_pct):
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
            elif self.state.get("entries", 0) >= self._max_entries_today():
                self.state["disabled"] = True

        return events

    def _entry_allowed(self, bar: Dict[str, Any]) -> bool:
        directive = self.state.get("event_directive")
        if directive == "skip":
            return False
        sizing_loss_pause = getattr(self.cfg.position_sizing, "loss_pause_trigger", 0.0)
        if sizing_loss_pause and self.state.get("daily_realized", 0.0) <= min(0.0, sizing_loss_pause):
            return False
        self.state["structure_hint"] = None
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
        if atr_norm is not None:
            if filt.atr_vol_hard_cap > 0 and atr_norm > filt.atr_vol_hard_cap:
                return False
            if atr_norm > filt.atr_vol_threshold:
                return False
        hi = self.state.get("intraday_high")
        lo = self.state.get("intraday_low")
        spot = self._safe_float(bar.get("spot"))
        if spot is not None and hi is not None and lo is not None and hi != lo:
            buffer = filt.support_res_pct * spot
            if abs(spot - hi) <= buffer:
                self.state["structure_hint"] = "resistance"
                if filt.block_near_prev_levels:
                    return False
            elif abs(spot - lo) <= buffer:
                self.state["structure_hint"] = "support"
                if filt.block_near_prev_levels:
                    return False
            range_pct = abs(hi - lo) / max(spot, 1.0)
            if filt.max_intraday_range_pct > 0 and range_pct >= filt.max_intraday_range_pct:
                return False
        oi_buffer_pct = max(0.0, filt.oi_level_buffer_pct)
        if spot is not None and oi_buffer_pct > 0:
            call_oi_strike = self._safe_float(bar.get("call_oi_max_strike"))
            if call_oi_strike is not None and call_oi_strike >= spot:
                diff = abs(call_oi_strike - spot)
                if diff >= filt.oi_level_min_gap and diff / max(call_oi_strike, 1.0) <= oi_buffer_pct:
                    self.state["structure_hint"] = "resistance"
                    if filt.block_near_oi_levels:
                        return False
            put_oi_strike = self._safe_float(bar.get("put_oi_max_strike"))
            if put_oi_strike is not None and put_oi_strike <= spot:
                diff = abs(spot - put_oi_strike)
                if diff >= filt.oi_level_min_gap and diff / max(put_oi_strike, 1.0) <= oi_buffer_pct:
                    self.state["structure_hint"] = "support"
                    if filt.block_near_oi_levels:
                        return False
        prev_buffer_pct = max(0.0, filt.prev_day_buffer_pct)
        if spot is not None and prev_buffer_pct > 0:
            prev_high = self._safe_float(bar.get("prev_day_high"))
            prev_low = self._safe_float(bar.get("prev_day_low"))
            if prev_high is not None and abs(spot - prev_high) / max(prev_high, 1.0) <= prev_buffer_pct:
                self.state["structure_hint"] = "resistance"
                if filt.block_near_prev_levels:
                    return False
            if prev_low is not None and abs(spot - prev_low) / max(prev_low, 1.0) <= prev_buffer_pct:
                self.state["structure_hint"] = "support"
                if filt.block_near_prev_levels:
                    return False
        iv_skew_val = self._safe_float(bar.get("iv_skew"))
        if iv_skew_val is not None and abs(iv_skew_val) > max(0.5, filt.range_skew_max * 5):
            return False
        vwap = self.state.get("spot_vwap")
        context = self.state.get("context_state", "unknown")
        if vwap and spot is not None and filt.vwap_band_pct > 0:
            diff_pct = (spot - vwap) / max(spot, 1.0)
            band = filt.vwap_band_pct
            if context == "trend_up" and diff_pct < band:
                return False
            if context == "trend_down" and diff_pct > -band:
                return False
            if context in {"inside", "range"} and abs(diff_pct) > band * 2:
                return False
        if context == "chop":
            return False
        return True

    def _update_spot_stats(self, bar: Dict[str, Any]) -> None:
        spot = self._safe_float(bar.get("spot"))
        if spot is None:
            return
        short_len = max(1, int(self.cfg.entry_filter.trend_lookback_min))
        long_len = max(short_len, int(self.cfg.entry_filter.trend_long_lookback_min))
        ema_cfg = self.cfg.ema

        def _append(name: str, length: int) -> None:
            buf = self.state.get(name, [])
            buf.append(spot)
            if len(buf) > length:
                buf = buf[-length:]
            self.state[name] = buf

        _append("spot_window", short_len)
        _append("spot_window_long", long_len)
        _append("hi_window_short", max(1, ema_cfg.breakout_short_period))
        _append("hi_window_long", max(1, ema_cfg.breakout_long_period))
        _append("lo_window_short", max(1, ema_cfg.breakout_short_period))
        _append("lo_window_long", max(1, ema_cfg.breakout_long_period))

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
        self._update_ema_values(spot)
        prev_high = self._safe_float(bar.get("prev_day_high"))
        prev_low = self._safe_float(bar.get("prev_day_low"))
        prev_range = self._safe_float(bar.get("prev_day_range"))
        self.state["inside_day"] = self._compute_inside_day(prev_high, prev_low, prev_range)
        self.state["breakout_flags"] = self._compute_breakout_flags(spot, prev_high, prev_low)
        self._update_vwap(bar, spot)
        self.state["context_state"] = self._derive_context()

    def _update_ema_values(self, spot: float) -> None:
        ema_cfg = self.cfg.ema
        self.state["ema_fast"] = self._ema_next(self.state.get("ema_fast"), spot, ema_cfg.fast_period)
        self.state["ema_mid"] = self._ema_next(self.state.get("ema_mid"), spot, ema_cfg.mid_period)
        self.state["ema_slow"] = self._ema_next(self.state.get("ema_slow"), spot, ema_cfg.slow_period)

    def _update_vwap(self, bar: Dict[str, Any], spot: float) -> None:
        call_vol = self._safe_float(bar.get("atm_call_volume"), 0.0) or 0.0
        put_vol = self._safe_float(bar.get("atm_put_volume"), 0.0) or 0.0
        prev_call = self.state.get("_prev_call_volume")
        prev_put = self.state.get("_prev_put_volume")
        delta_call = call_vol if prev_call is None else max(call_vol - prev_call, 0.0)
        delta_put = put_vol if prev_put is None else max(put_vol - prev_put, 0.0)
        synthetic_volume = delta_call + delta_put
        if synthetic_volume > 0:
            self.state["vwap_numerator"] += spot * synthetic_volume
            self.state["vwap_denominator"] += synthetic_volume
            self.state["spot_vwap"] = self.state["vwap_numerator"] / max(self.state["vwap_denominator"], 1e-9)
        self.state["_prev_call_volume"] = call_vol
        self.state["_prev_put_volume"] = put_vol

    @staticmethod
    def _ema_next(prev: Optional[float], spot: float, period: int) -> float:
        if period <= 1 or prev is None:
            return spot
        alpha = 2.0 / (period + 1)
        return prev + alpha * (spot - prev)

    def _recent_high(self, name: str) -> Optional[float]:
        window = self.state.get(name) or []
        if not window:
            return None
        return max(window)

    def _recent_low(self, name: str) -> Optional[float]:
        window = self.state.get(name) or []
        if not window:
            return None
        return min(window)

    def _compute_inside_day(self, prev_high: Optional[float], prev_low: Optional[float], prev_range: Optional[float]) -> bool:
        hi = self.state.get("intraday_high")
        lo = self.state.get("intraday_low")
        if hi is None or lo is None or prev_high is None or prev_low is None:
            return False
        if not (lo >= prev_low and hi <= prev_high):
            return False
        if not prev_range or prev_range <= 0:
            return True
        current_range = hi - lo
        threshold = max(0.0, self.cfg.entry_filter.inside_day_threshold)
        return current_range <= prev_range * threshold

    def _compute_breakout_flags(self, spot: float, prev_high: Optional[float], prev_low: Optional[float]) -> Dict[str, bool]:
        filt = self.cfg.entry_filter
        buffer = spot * max(0.0, filt.breakout_buffer_pct)
        prev_buffer = spot * max(0.0, filt.breakout_prevday_buffer_pct)
        hi_short = self._recent_high("hi_window_short") or spot
        hi_long = self._recent_high("hi_window_long") or spot
        lo_short = self._recent_low("lo_window_short") or spot
        lo_long = self._recent_low("lo_window_long") or spot
        breakout_up = spot >= (hi_long + buffer)
        breakout_down = spot <= (lo_long - buffer)
        prev_breakout_up = prev_high is not None and spot >= (prev_high + prev_buffer)
        prev_breakout_down = prev_low is not None and spot <= (prev_low - prev_buffer)
        return {
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "prev_breakout_up": prev_breakout_up,
            "prev_breakout_down": prev_breakout_down,
        }

    def _ema_alignment(self) -> str:
        fast = self.state.get("ema_fast")
        mid = self.state.get("ema_mid")
        slow = self.state.get("ema_slow")
        if fast is None or mid is None or slow is None:
            return "unknown"
        if fast > mid > slow:
            return "up"
        if fast < mid < slow:
            return "down"
        return "mixed"

    def _derive_context(self) -> str:
        flags = self.state.get("breakout_flags", {})
        inside = self.state.get("inside_day")
        ema_alignment = self._ema_alignment()
        atr_norm = self._spot_atr() or 0.0
        filt = self.cfg.entry_filter
        if inside:
            return "inside"
        if atr_norm >= filt.atr_vol_hard_cap:
            return "chop"
        if ema_alignment == "up" and (flags.get("breakout_up") or flags.get("prev_breakout_up")):
            return "trend_up"
        if ema_alignment == "down" and (flags.get("breakout_down") or flags.get("prev_breakout_down")):
            return "trend_down"
        if atr_norm <= filt.atr_vol_threshold * 0.7:
            return "range"
        return "chop"

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

    def _directional_confirm(self, regime: str) -> bool:
        if regime not in {"trend_up", "trend_down"}:
            return False
        if self.state.get("inside_day"):
            return False
        filt = self.cfg.entry_filter
        slope_short = self._spot_slope("spot_window") or 0.0
        slope_long = self._spot_slope("spot_window_long") or 0.0
        atr_norm = self._spot_atr() or 0.0
        iv_skew = self._safe_float(self.state.get("_last_iv_skew"), 0.0) or 0.0
        oi_skew = self._safe_float(self.state.get("_last_oi_skew"), 0.0) or 0.0
        trigger = max(0.0, filt.directional_slope_trigger) * max(1.0, filt.directional_confirm_mult)
        min_skew = max(0.0, filt.directional_min_skew)
        ema_alignment = self._ema_alignment()
        flags = self.state.get("breakout_flags", {})
        breakout_up = flags.get("breakout_up") or flags.get("prev_breakout_up")
        breakout_down = flags.get("breakout_down") or flags.get("prev_breakout_down")

        if atr_norm > filt.atr_vol_threshold:
            return False

        if regime == "trend_up":
            return (
                slope_short >= trigger
                and slope_long >= trigger * 0.5
                and iv_skew >= min_skew
                and oi_skew >= 0
                and ema_alignment == "up"
                and breakout_up
            )
        if regime == "trend_down":
            return (
                slope_short <= -trigger
                and slope_long <= -trigger * 0.5
                and iv_skew <= -min_skew
                and oi_skew <= 0
                and ema_alignment == "down"
                and breakout_down
            )
        return False

    def _determine_structure(self) -> Optional[str]:
        if self.state.get("event_directive") == "reduce":
            return "STRANGLE"
        if self.state.get("inside_day"):
            return "STRANGLE"
        hint = self.state.get("structure_hint")
        if hint == "resistance":
            return "BEAR_CALL_SPREAD" if self._directional_confirm("trend_down") else "STRANGLE"
        if hint == "support":
            return "BULL_PUT_SPREAD" if self._directional_confirm("trend_up") else "STRANGLE"
        regime = self._infer_regime()
        if regime == "trend_up":
            return "BULL_PUT_SPREAD" if self._directional_confirm(regime) else "STRANGLE"
        if regime == "trend_down":
            return "BEAR_CALL_SPREAD" if self._directional_confirm(regime) else "STRANGLE"
        if regime == "volatile":
            return None
        if regime == "range":
            return "IRON_FLY"
        if regime == "neutral":
            return "IRON_CONDOR"
        return "STRANGLE"

    def _record_equity(self, ts: datetime, equity: float) -> None:
        self.state.setdefault("equity_curve", []).append({"timestamp": ts, "equity": equity})

    def _max_entries_today(self) -> int:
        max_entries = self.state.get("max_entries_today")
        if max_entries is None:
            return self.cfg.max_entries_per_day
        return max_entries

    def _structure_cost(self, structure: str, qty: int) -> float:
        # Explicit long legs are priced via real LTPs, so no additional fixed charges.
        return 0.0

    def _estimate_margin_per_lot(self, legs: Dict[str, Any], bar: Dict[str, Any]) -> float:
        spot = self._safe_float(bar.get("spot")) or self._safe_float(legs.get("call_strike")) or self._safe_float(legs.get("put_strike")) or 1.0
        sizing = self.cfg.position_sizing
        margin_pct = max(0.02, float(sizing.margin_pct))
        margin = spot * self.cfg.lot_size * margin_pct
        call_short = self._safe_float(legs.get("call_strike"))
        call_long = self._safe_float(legs.get("long_call_strike"))
        put_short = self._safe_float(legs.get("put_strike"))
        put_long = self._safe_float(legs.get("long_put_strike"))
        spread_width = 0.0
        if call_short is not None and call_long is not None:
            spread_width = max(spread_width, abs(call_long - call_short))
        if put_short is not None and put_long is not None:
            spread_width = max(spread_width, abs(put_short - put_long))
        if spread_width > 0:
            margin = min(margin, spread_width * self.cfg.lot_size)
        return max(margin, 1.0)

    def _structure_qty(self, structure: str, legs: Dict[str, Any], bar: Dict[str, Any]) -> int:
        sizing = self.cfg.position_sizing
        if structure in {"IRON_FLY", "IRON_CONDOR"}:
            base = sizing.iron_qty
        elif structure in {"BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"}:
            base = sizing.spread_qty
        elif structure in {"STRANGLE", "SHORT_CALL", "SHORT_PUT"}:
            base = sizing.strangle_qty
        else:
            base = sizing.base_qty
        daily_realized = self.state.get("daily_realized", 0.0)
        trigger = sizing.aggressive_trigger or self.cfg.daily_target.min_profit
        if trigger and daily_realized >= trigger:
            base += max(0, sizing.aggressive_bonus)
        if sizing.drawdown_trigger and daily_realized <= -abs(sizing.drawdown_trigger):
            base = max(1, base - max(0, sizing.drawdown_penalty))
        base = max(1, base)
        margin = self._estimate_margin_per_lot(legs, bar)
        capital_cap = max(0.0, sizing.capital_pct_limit) * self.cfg.capital
        max_allowed = max(1, int(capital_cap / margin)) if capital_cap > 0 else sizing.max_qty
        qty = max(1, min(base, sizing.max_qty, max_allowed))
        return qty

    def _structure_blueprint(self, structure: str) -> Dict[str, Any]:
        offsets = self.cfg.structure_offsets
        def field(option_type: str, offset: int, metric: str) -> str:
            return self._selector_field(option_type, offset, metric)

        blueprint: Dict[str, Any] = {
            "structure": structure,
            "call_field": field("call", offsets.strangle_call_offset, "ltp"),
            "call_strike_field": field("call", offsets.strangle_call_offset, "strike"),
            "call_required": True,
            "put_field": field("put", offsets.strangle_put_offset, "ltp"),
            "put_strike_field": field("put", offsets.strangle_put_offset, "strike"),
            "put_required": True,
            "long_call_field": None,
            "long_call_strike_field": None,
            "long_call_required": False,
            "long_put_field": None,
            "long_put_strike_field": None,
            "long_put_required": False,
        }
        if structure == "SHORT_CALL":
            blueprint["call_field"] = field("call", offsets.short_call_offset, "ltp")
            blueprint["call_strike_field"] = field("call", offsets.short_call_offset, "strike")
            blueprint["put_field"] = None
            blueprint["put_required"] = False
        elif structure == "SHORT_PUT":
            blueprint["put_field"] = field("put", offsets.short_put_offset, "ltp")
            blueprint["put_strike_field"] = field("put", offsets.short_put_offset, "strike")
            blueprint["call_field"] = None
            blueprint["call_required"] = False
        elif structure == "BULL_PUT_SPREAD":
            blueprint["call_field"] = None
            blueprint["call_required"] = False
            blueprint["put_field"] = field("put", offsets.bull_put_short_offset, "ltp")
            blueprint["put_strike_field"] = field("put", offsets.bull_put_short_offset, "strike")
            blueprint["long_put_field"] = field("put", offsets.bull_put_long_offset, "ltp")
            blueprint["long_put_strike_field"] = field("put", offsets.bull_put_long_offset, "strike")
            blueprint["long_put_required"] = True
        elif structure == "BEAR_CALL_SPREAD":
            blueprint["put_field"] = None
            blueprint["put_required"] = False
            blueprint["call_field"] = field("call", offsets.bear_call_short_offset, "ltp")
            blueprint["call_strike_field"] = field("call", offsets.bear_call_short_offset, "strike")
            blueprint["long_call_field"] = field("call", offsets.bear_call_long_offset, "ltp")
            blueprint["long_call_strike_field"] = field("call", offsets.bear_call_long_offset, "strike")
            blueprint["long_call_required"] = True
        elif structure == "IRON_FLY":
            blueprint["long_call_field"] = field("call", offsets.iron_fly_call_long_offset, "ltp")
            blueprint["long_call_strike_field"] = field("call", offsets.iron_fly_call_long_offset, "strike")
            blueprint["long_call_required"] = True
            blueprint["long_put_field"] = field("put", offsets.iron_fly_put_long_offset, "ltp")
            blueprint["long_put_strike_field"] = field("put", offsets.iron_fly_put_long_offset, "strike")
            blueprint["long_put_required"] = True
            blueprint["call_field"] = field("call", offsets.iron_fly_call_short_offset, "ltp")
            blueprint["call_strike_field"] = field("call", offsets.iron_fly_call_short_offset, "strike")
            blueprint["put_field"] = field("put", offsets.iron_fly_put_short_offset, "ltp")
            blueprint["put_strike_field"] = field("put", offsets.iron_fly_put_short_offset, "strike")
        elif structure == "IRON_CONDOR":
            blueprint["call_field"] = field("call", offsets.iron_condor_call_short_offset, "ltp")
            blueprint["call_strike_field"] = field("call", offsets.iron_condor_call_short_offset, "strike")
            blueprint["put_field"] = field("put", offsets.iron_condor_put_short_offset, "ltp")
            blueprint["put_strike_field"] = field("put", offsets.iron_condor_put_short_offset, "strike")
            blueprint["long_call_field"] = field("call", offsets.iron_condor_call_long_offset, "ltp")
            blueprint["long_call_strike_field"] = field("call", offsets.iron_condor_call_long_offset, "strike")
            blueprint["long_call_required"] = True
            blueprint["long_put_field"] = field("put", offsets.iron_condor_put_long_offset, "ltp")
            blueprint["long_put_strike_field"] = field("put", offsets.iron_condor_put_long_offset, "strike")
            blueprint["long_put_required"] = True
        return blueprint

    def _resolve_structure(self, structure: str, bar: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        blueprint = self._structure_blueprint(structure)
        resolved: Dict[str, Any] = {"structure": structure}
        for leg in ("call", "put", "long_call", "long_put"):
            field = blueprint.get(f"{leg}_field")
            strike_field = blueprint.get(f"{leg}_strike_field")
            required = blueprint.get(f"{leg}_required", False)
            price = None
            strike = None
            if field:
                price = self._leg_price(bar, field)
                if price is None and required:
                    return None
                strike = bar.get(strike_field) if strike_field else None
            resolved[f"{leg}_field"] = field if price is not None else None
            resolved[f"{leg}_price"] = price
            resolved[f"{leg}_strike"] = strike
        if blueprint.get("call_required") and resolved.get("call_field") is None:
            return None
        if blueprint.get("put_required") and resolved.get("put_field") is None:
            return None
        if blueprint.get("long_call_required") and resolved.get("long_call_field") is None:
            return None
        if blueprint.get("long_put_required") and resolved.get("long_put_field") is None:
            return None
        return resolved

    def _prepare_structure(self, structure: str, bar: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        resolved = self._resolve_structure(structure, bar)
        if resolved is not None:
            return resolved
        if structure != "STRANGLE":
            fallback = self._resolve_structure("STRANGLE", bar)
            if fallback is not None:
                fallback["structure"] = "STRANGLE"
                return fallback
        return None

    def _load_event_calendar(self, path_str: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
        if not path_str:
            return {}
        path = Path(path_str)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
        except Exception:
            return {}
        if isinstance(data, dict):
            return {k: (v if isinstance(v, list) else [v]) for k, v in data.items()}
        events: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                date_key = entry.get("date")
                if not date_key:
                    continue
                events.setdefault(date_key, []).append(entry)
        return events

    def _event_directive(self, date_obj: Optional[datetime.date]) -> Optional[str]:
        if date_obj is None:
            return None
        key = date_obj.isoformat()
        entries = self._event_calendar.get(key, [])
        if not entries:
            return None
        severities = {str(e.get("severity", "")).lower(): e for e in entries}
        actions = [str(e.get("action", "")).lower() for e in entries]
        if any(action == "skip" for action in actions):
            return "skip"
        for sev in severities:
            if sev in self.cfg.event_skip_levels:
                return "skip"
        if any(action in {"light", "reduce"} for action in actions):
            return "reduce"
        for sev in severities:
            if sev in self.cfg.event_light_levels:
                return "reduce"
        return None

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
    if isinstance(data.get("structure_offsets"), dict):
        data["structure_offsets"] = StructureWingConfig(**data["structure_offsets"])
    if isinstance(data.get("position_sizing"), dict):
        data["position_sizing"] = PositionSizingConfig(**data["position_sizing"])
    if isinstance(data.get("ema"), dict):
        data["ema"] = EmaConfig(**data["ema"])
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
