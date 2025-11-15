"""
Weekly Theta Harvesting Strategy
================================

Simplified prototype focused on selling ATM strangles early in the week and
managing them through weekly expiry with MTM targets/stops.

This reuses the same intraday dataset produced for the intraday engine
(`reports/intraday_from_rolling_*.csv`). We collapse each trade date down to
its first/last ticks to approximate entry (open) and daily MTM (close).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pickle
import pandas as pd
import json
import logging
import numpy as np

LOG = logging.getLogger(__name__)

__all__ = ["WeeklyConfig", "WeeklyTargets", "EntryRules", "run_backtest", "WeeklyThetaStrangle"]


@dataclass
class EntryRules:
    min_premium_pct: float = 0.016
    min_iv_rank: float = 0.05
    max_iv_rank: float = 1.0
    max_prev_range_pct: float = 0.035
    min_prev_range_pct: float = 0.003
    demand_prev_break: bool = True
    min_days_to_target_expiry: int = 0
    max_gap_pct: float = 0.01  # 1% gap guard


@dataclass
class WeeklyTargets:
    pnl_target: float = 6_000.0
    pnl_stop: float = 4_000.0
    trailing_lock_pct: float = 0.25
    hard_exit_day: int = 3  # Thursday (0=Mon)
    max_hold_days: int = 5


@dataclass
class WeeklyConfig:
    lot_size: int = 50
    qty: int = 2
    entry_day: int = 0  # Monday
    exit_day: int = 3   # Thursday
    entry_rules: EntryRules = field(default_factory=EntryRules)
    targets: WeeklyTargets = field(default_factory=WeeklyTargets)
    timezone: str = "Asia/Kolkata"
    structure: str = "STRANGLE"  # default fallback
    wing_offset: int = 2
    hybrid_enabled: bool = False
    trend_range_threshold: float = 0.02
    condor_range_threshold: float = 0.01
    oi_distance_pct: float = 0.01
    event_calendar_path: Optional[str] = "data_engine/market_ai/state/events.json"
    skip_event_severities: Tuple[str, ...] = ("high",)
    expiry_offset_weeks: int = 0
    ml_exit_model_path: Optional[str] = None
    ml_exit_min_prob: float = 0.6
    directional_enabled: bool = True
    directional_bias_threshold: float = 0.02


@dataclass
class WeeklyPosition:
    entry_date: datetime
    entry_call: float
    entry_put: float
    qty: int
    lot_size: int
    structure: str = "STRANGLE"
    long_call_entry: float = 0.0
    long_put_entry: float = 0.0
    short_call_strike: Optional[float] = None
    short_put_strike: Optional[float] = None
    long_call_strike: Optional[float] = None
    long_put_strike: Optional[float] = None
    best_pnl: float = 0.0
    target_expiry: Optional[pd.Timestamp] = None

    def entry_credit(self) -> float:
        credit = (self.entry_call + self.entry_put) - (self.long_call_entry + self.long_put_entry)
        return credit * self.qty * self.lot_size

    def current_pnl(
        self,
        call_ltp: float,
        put_ltp: float,
        long_call_ltp: Optional[float] = None,
        long_put_ltp: Optional[float] = None,
    ) -> float:
        long_call_ltp = long_call_ltp or 0.0
        long_put_ltp = long_put_ltp or 0.0
        debit = (call_ltp + put_ltp - long_call_ltp - long_put_ltp) * self.qty * self.lot_size
        return self.entry_credit() - debit


class WeeklyThetaStrangle:
    def __init__(self, cfg: WeeklyConfig):
        self.cfg = cfg
        self.position: Optional[WeeklyPosition] = None
        self.week_id: Optional[Tuple[int, int]] = None
        self.trades: List[Dict[str, Any]] = []
        self._event_calendar = _load_event_calendar(cfg.event_calendar_path)
        self.daily_snapshots: List[Dict[str, Any]] = []
        self._last_close_reason: Optional[str] = None
        self._last_close_time: Optional[pd.Timestamp] = None
        self._last_close_pnl: Optional[float] = None
        self._last_closed_position: Optional[WeeklyPosition] = None
        self._ml_exit_helper: Optional[MLExitHelper] = None
        self._last_ml_prob: Optional[float] = None
        self._last_ml_decision: Optional[str] = None
        model_path = cfg.ml_exit_model_path
        if model_path:
            try:
                self._ml_exit_helper = MLExitHelper(Path(model_path), cfg.ml_exit_min_prob)
                LOG.info("Loaded ML exit model from %s (threshold=%.2f)", model_path, self._ml_exit_helper.threshold)
            except Exception as exc:
                LOG.warning("Could not load ML exit model %s: %s", model_path, exc)

    def on_new_day(self, day_rows: pd.DataFrame) -> None:
        if day_rows.empty:
            return
        day_rows = day_rows.sort_values("timestamp")
        morning, close = day_rows.iloc[0], day_rows.iloc[-1]
        current_date = pd.to_datetime(morning["timestamp"]).date()
        weekday = current_date.weekday()
        current_week = (current_date.isocalendar().year, current_date.isocalendar().week)
        entry_attempted = False
        entry_outcome = None
        entry_block_reason = None

        if self.position is None and weekday <= self.cfg.exit_day:
            entry_attempted = True
            allowed, block_reason = self._entry_allowed(day_rows)
            if allowed:
                self.week_id = current_week
                self._open_position(morning)
                entry_outcome = "OPENED"
            else:
                entry_block_reason = block_reason or "blocked"

        if self.position:
            if current_week != self.week_id:
                # Safety: week rolled but we still have a position -> force close at prev close price.
                self._close_position(close, reason="week_roll")
                return
            pnl = self.position.current_pnl(close["atm_call_ltp"], close["atm_put_ltp"])
            self.position.best_pnl = max(self.position.best_pnl, pnl)
            exit_reason = self._check_exit(weekday, pnl, morning, close)
            if exit_reason:
                self._close_position(close, reason=exit_reason)
        self._log_daily_snapshot(morning, close, weekday, entry_attempted, entry_outcome, entry_block_reason)

    def _open_position(self, row: pd.Series) -> None:
        structure = self._determine_structure(row)
        legs = self._resolve_entry_legs(row, structure)
        if legs is None:
            return
        expiry, days_to = self._target_expiry_info(row)
        if expiry is None:
            return
        pos = WeeklyPosition(
            entry_date=pd.to_datetime(row["timestamp"]),
            entry_call=legs["short_call"],
            entry_put=legs["short_put"],
            qty=self.cfg.qty,
            lot_size=self.cfg.lot_size,
            structure=legs["structure"],
            long_call_entry=legs.get("long_call", 0.0),
            long_put_entry=legs.get("long_put", 0.0),
            short_call_strike=legs.get("short_call_strike"),
            short_put_strike=legs.get("short_put_strike"),
            long_call_strike=legs.get("long_call_strike"),
            long_put_strike=legs.get("long_put_strike"),
            target_expiry=pd.to_datetime(expiry),
        )
        self.position = pos
        self.trades.append(
            {
                "timestamp": pos.entry_date,
                "action": "OPEN",
                "call_ltp": pos.entry_call,
                "put_ltp": pos.entry_put,
                "long_call_ltp": pos.long_call_entry or None,
                "long_put_ltp": pos.long_put_entry or None,
                "call_strike": pos.short_call_strike,
                "put_strike": pos.short_put_strike,
                "long_call_strike": pos.long_call_strike,
                "long_put_strike": pos.long_put_strike,
                "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else expiry,
                "target_days": days_to,
                "expiry_mode": "NEXT_WEEK" if self.cfg.expiry_offset_weeks else "FRONT_WEEK",
                "structure": pos.structure,
                "notes": "weekly_entry",
            }
        )

    def _close_position(self, row: pd.Series, reason: str) -> None:
        if self.position is None:
            return
        current_pos = self.position
        current_legs = self._resolve_close_legs(row, current_pos.structure)
        pnl = self.position.current_pnl(
            current_legs["short_call"],
            current_legs["short_put"],
            current_legs.get("long_call"),
            current_legs.get("long_put"),
        )
        self.trades.append(
            {
                "timestamp": pd.to_datetime(row["timestamp"]),
                "action": "CLOSE",
                "call_ltp": current_legs["short_call"],
                "put_ltp": current_legs["short_put"],
                "long_call_ltp": current_legs.get("long_call"),
                "long_put_ltp": current_legs.get("long_put"),
                "realized": pnl,
                "reason": reason,
            }
        )
        self._last_close_reason = reason
        self._last_close_time = pd.to_datetime(row["timestamp"])
        self._last_close_pnl = pnl
        self._last_closed_position = current_pos
        self.position = None
        self.week_id = None

    def _entry_allowed(self, day_rows: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        rules = self.cfg.entry_rules
        day_rows = day_rows.sort_values("timestamp")
        row = day_rows.iloc[0]
        premium_pct = row.get("combined_premium_pct")
        if pd.isna(premium_pct) or premium_pct < rules.min_premium_pct:
            return False, "premium_lt_min"
        iv_rank = row.get("iv_rank")
        if not pd.isna(iv_rank) and not (rules.min_iv_rank <= iv_rank <= rules.max_iv_rank):
            return False, "iv_rank_band"
        prev_range = row.get("prev_day_range")
        spot = row.get("spot")
        if not pd.isna(prev_range) and spot:
            prev_range_pct = abs(prev_range) / max(spot, 1.0)
            if prev_range_pct < rules.min_prev_range_pct or prev_range_pct > rules.max_prev_range_pct:
                return False, "prev_range_band"
        if rules.demand_prev_break:
            prev_high = row.get("prev_day_high")
            prev_low = row.get("prev_day_low")
            if not pd.isna(prev_high) and not pd.isna(prev_low):
                buffer = spot * 0.0008 if spot else 0.0
                day_high = day_rows["spot"].max()
                day_low = day_rows["spot"].min()
                if not ((day_high >= (prev_high + buffer)) or (day_low <= (prev_low - buffer))):
                    return False, "no_prev_break"
        prev_close = row.get("prev_day_close")
        if not pd.isna(prev_close) and spot:
            gap_pct = abs(spot - prev_close) / max(prev_close, 1.0)
            if gap_pct > getattr(rules, "max_gap_pct", 0.01):
                return False, "gap_guard"
        expiry, days_to = self._target_expiry_info(row)
        min_days = getattr(rules, "min_days_to_target_expiry", 0)
        if expiry is None:
            return False, "missing_expiry"
        if min_days and (days_to is None or days_to < min_days):
            return False, "days_to_expiry"
        if not self._event_allowed(pd.to_datetime(row["timestamp"]).date()):
            return False, "event_block"
        return True, None

    def _determine_structure(self, row: pd.Series) -> str:
        if not self.cfg.hybrid_enabled:
            return self.cfg.structure.upper()
        spot = row.get("spot")
        prev_range = row.get("prev_day_range")
        prev_range_pct = abs(prev_range) / max(spot, 1.0) if spot and prev_range else 0.0
        bias = row.get("spot_trend_20") or 0.0
        if self.cfg.directional_enabled and prev_range_pct >= self.cfg.directional_bias_threshold and bias:
            if bias >= 0:
                return "BULL_PUT_SPREAD"
            return "BEAR_CALL_SPREAD"
        if prev_range_pct >= self.cfg.trend_range_threshold:
            return "STRANGLE"
        if prev_range_pct <= self.cfg.condor_range_threshold and self._condor_viable(row):
            return "IRON_CONDOR"
        return self.cfg.structure.upper()

    def _condor_viable(self, row: pd.Series) -> bool:
        spot = row.get("spot")
        call_strike = row.get("call_oi_max_strike")
        put_strike = row.get("put_oi_max_strike")
        if pd.isna(spot) or pd.isna(call_strike) or pd.isna(put_strike):
            return False
        distance_call = abs(call_strike - spot) / max(spot, 1.0)
        distance_put = abs(spot - put_strike) / max(spot, 1.0)
        min_distance = max(0.005, self.cfg.oi_distance_pct)
        return distance_call >= min_distance and distance_put >= min_distance

    def _resolve_entry_legs(self, row: pd.Series, structure: str) -> Optional[Dict[str, float]]:
        structure = structure.upper()
        short_call = row.get("atm_call_ltp")
        short_put = row.get("atm_put_ltp")
        if pd.isna(short_call) or pd.isna(short_put):
            return None
        legs = {
            "short_call": float(short_call),
            "short_put": float(short_put),
            "short_call_strike": float(row.get("atm_call_strike", 0.0)) if not pd.isna(row.get("atm_call_strike")) else None,
            "short_put_strike": float(row.get("atm_put_strike", 0.0)) if not pd.isna(row.get("atm_put_strike")) else None,
            "structure": structure,
        }
        if structure == "BULL_PUT_SPREAD":
            legs["short_call"] = 0.0
            legs["short_call_strike"] = None
            long_put = row.get(_selector_field("put", -2, "ltp"))
            long_put_strike = row.get(_selector_field("put", -2, "strike"))
            if pd.isna(long_put) or pd.isna(long_put_strike):
                legs["structure"] = "STRANGLE"
                return legs
            legs["long_put"] = float(long_put)
            legs["long_put_strike"] = float(long_put_strike)
            legs["structure"] = "BULL_PUT_SPREAD"
            return legs
        if structure == "BEAR_CALL_SPREAD":
            legs["short_put"] = 0.0
            legs["short_put_strike"] = None
            long_call = row.get(_selector_field("call", 2, "ltp"))
            long_call_strike = row.get(_selector_field("call", 2, "strike"))
            if pd.isna(long_call) or pd.isna(long_call_strike):
                legs["structure"] = "STRANGLE"
                return legs
            legs["long_call"] = float(long_call)
            legs["long_call_strike"] = float(long_call_strike)
            legs["structure"] = "BEAR_CALL_SPREAD"
            return legs
        if structure != "IRON_CONDOR":
            legs["structure"] = "STRANGLE"
            return legs
        offset = max(1, int(self.cfg.wing_offset))
        long_call = row.get(_selector_field("call", offset, "ltp"))
        long_put = row.get(_selector_field("put", -offset, "ltp"))
        if pd.isna(long_call) or pd.isna(long_put):
            legs["structure"] = "STRANGLE"
            return legs
        legs["long_call"] = float(long_call)
        legs["long_put"] = float(long_put)
        legs["long_call_strike"] = float(row.get(_selector_field("call", offset, "strike"), 0.0)) if not pd.isna(row.get(_selector_field("call", offset, "strike"))) else None
        legs["long_put_strike"] = float(row.get(_selector_field("put", -offset, "strike"), 0.0)) if not pd.isna(row.get(_selector_field("put", -offset, "strike"))) else None
        return legs

    def _resolve_close_legs(self, row: pd.Series, structure: str) -> Dict[str, float]:
        legs = {
            "short_call": float(row.get("atm_call_ltp", 0.0)),
            "short_put": float(row.get("atm_put_ltp", 0.0)),
        }
        if structure == "BULL_PUT_SPREAD":
            long_put = row.get(_selector_field("put", -2, "ltp"))
            legs["long_put"] = float(long_put) if not pd.isna(long_put) else 0.0
            return legs
        if structure == "BEAR_CALL_SPREAD":
            long_call = row.get(_selector_field("call", 2, "ltp"))
            legs["long_call"] = float(long_call) if not pd.isna(long_call) else 0.0
            return legs
        if structure != "IRON_CONDOR":
            return legs
        offset = max(1, int(self.cfg.wing_offset))
        long_call = row.get(_selector_field("call", offset, "ltp"))
        long_put = row.get(_selector_field("put", -offset, "ltp"))
        legs["long_call"] = float(long_call) if not pd.isna(long_call) else 0.0
        legs["long_put"] = float(long_put) if not pd.isna(long_put) else 0.0
        return legs

    def _target_expiry_info(self, row: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
        base_expiry = pd.to_datetime(row.get("expiryDate"), errors="coerce") if "expiryDate" in row else None
        base_days = row.get("days_to_expiry")
        offset = max(0, int(self.cfg.expiry_offset_weeks))
        target_expiry = base_expiry
        target_days = float(base_days) if base_days is not None and not pd.isna(base_days) else None
        if offset > 0:
            if offset == 1 and "next_week_expiry" in row:
                next_expiry = pd.to_datetime(row.get("next_week_expiry"), errors="coerce")
                if next_expiry is not None and not pd.isna(next_expiry):
                    target_expiry = next_expiry
                next_days = row.get("days_to_next_expiry")
                if next_days is not None and not pd.isna(next_days):
                    target_days = float(next_days)
            if (target_expiry is None or pd.isna(target_expiry)) and base_expiry is not None and not pd.isna(base_expiry):
                target_expiry = base_expiry + pd.to_timedelta(7 * offset, unit="D")
            if (target_days is None) and base_days is not None and not pd.isna(base_days):
                target_days = float(base_days) + 7 * offset
        if target_expiry is not None and pd.isna(target_expiry):
            target_expiry = None
        return target_expiry, target_days

    def _events_for(self, current_date: datetime.date) -> List[Dict[str, Any]]:
        if not self._event_calendar:
            return []
        return self._event_calendar.get(current_date.isoformat(), [])

    def _event_allowed(self, current_date: datetime.date) -> bool:
        entries = self._events_for(current_date)
        skip_set = {s.lower() for s in self.cfg.skip_event_severities}
        for entry in entries:
            sev = str(entry.get("severity", "")).lower()
            if sev in skip_set:
                return False
        return True

    def _log_daily_snapshot(
        self,
        morning: pd.Series,
        close: pd.Series,
        weekday: int,
        entry_attempted: bool,
        entry_outcome: Optional[str],
        entry_block_reason: Optional[str],
    ) -> None:
        current_date = pd.to_datetime(morning["timestamp"]).date()
        events = self._events_for(current_date)
        events_json = json.dumps(events, default=str)
        event_tags = ",".join(
            sorted(
                {
                    str(evt.get("tag") or evt.get("event") or evt.get("title") or "").strip()
                    for evt in events
                    if evt
                }
            )
        ).strip(",")
        active_position = self.position
        last_closed_applicable = (
            self._last_closed_position if self._last_close_time and self._last_close_time.date() == current_date else None
        )
        pnl = None
        best_pnl = None
        target_expiry = None
        target_days_remaining = None
        entry_date = None
        structure = None
        if active_position:
            pnl = active_position.current_pnl(close["atm_call_ltp"], close["atm_put_ltp"])
            best_pnl = active_position.best_pnl
            target_expiry = active_position.target_expiry
            if target_expiry is not None and not pd.isna(target_expiry):
                target_days_remaining = (target_expiry.date() - current_date).days
            entry_date = active_position.entry_date
            structure = active_position.structure
        elif last_closed_applicable:
            target_expiry = last_closed_applicable.target_expiry
            if target_expiry is not None and not pd.isna(target_expiry):
                target_days_remaining = (target_expiry.date() - current_date).days
            entry_date = last_closed_applicable.entry_date
            structure = last_closed_applicable.structure
        snapshot = {
            "date": current_date.isoformat(),
            "weekday": weekday,
            "entry_attempted": entry_attempted,
            "entry_outcome": entry_outcome,
            "entry_block_reason": entry_block_reason,
            "spot_open": morning.get("spot"),
            "spot_close": close.get("spot"),
            "combined_premium_pct": morning.get("combined_premium_pct"),
            "iv_rank": morning.get("iv_rank"),
            "iv_skew": morning.get("iv_skew"),
            "oi_skew": morning.get("oi_skew"),
            "volume_skew": morning.get("volume_skew"),
            "call_oi_max_strike": morning.get("call_oi_max_strike"),
            "put_oi_max_strike": morning.get("put_oi_max_strike"),
            "spot_return_1": close.get("spot_return_1"),
            "spot_return_3": close.get("spot_return_3"),
            "spot_return_5": close.get("spot_return_5"),
            "spot_trend_20": close.get("spot_trend_20"),
            "spot_trend_50": close.get("spot_trend_50"),
            "spot_trend_100": close.get("spot_trend_100"),
            "spot_volatility": close.get("spot_volatility"),
            "spot_intraday_range_pct": close.get("spot_intraday_range_pct"),
            "call_oi_change": close.get("call_oi_change"),
            "put_oi_change": close.get("put_oi_change"),
            "events": events_json,
            "event_count": len(events),
            "event_tags": event_tags,
            "has_position": active_position is not None,
            "structure": structure,
            "entry_date": entry_date.isoformat() if entry_date is not None else None,
            "target_expiry": target_expiry.isoformat() if target_expiry is not None and not pd.isna(target_expiry) else None,
            "target_days_remaining": target_days_remaining,
            "current_pnl": pnl,
            "best_pnl": best_pnl,
            "close_reason": self._last_close_reason if last_closed_applicable else None,
            "close_pnl": self._last_close_pnl if last_closed_applicable else None,
            "ml_exit_prob": self._last_ml_prob,
            "ml_exit_decision": self._last_ml_decision,
        }
        self.daily_snapshots.append(snapshot)
        if last_closed_applicable:
            self._last_closed_position = None
            self._last_close_reason = None
            self._last_close_pnl = None
            self._last_close_time = None
        self._last_ml_prob = None
        self._last_ml_decision = None

    def _check_exit(self, weekday: int, pnl: float, morning_row: pd.Series, close_row: pd.Series) -> Optional[str]:
        targets = self.cfg.targets
        ml_reason = self._ml_exit_decision(weekday, pnl, morning_row, close_row)
        if ml_reason:
            return ml_reason
        if pnl >= targets.pnl_target:
            return "target_hit"
        if pnl <= -targets.pnl_stop:
            return "weekly_stop"
        if self.position:
            lock_floor = self.position.best_pnl * (1 - targets.trailing_lock_pct)
            if lock_floor and pnl <= lock_floor:
                return "trailing_lock"
        days_held = (weekday - self.cfg.entry_day) % 7
        if days_held >= targets.max_hold_days - 1:
            return "max_hold_days"
        if weekday >= targets.hard_exit_day:
            return "hard_exit"
        return None

    def _ml_exit_decision(self, weekday: int, pnl: float, morning_row: pd.Series, close_row: pd.Series) -> Optional[str]:
        self._last_ml_prob = None
        self._last_ml_decision = None
        helper = self._ml_exit_helper
        if not helper or self.position is None:
            return None
        features = self._ml_feature_vector(weekday, pnl, morning_row, close_row)
        if features is None:
            return None
        prob = helper.predict_prob(features)
        self._last_ml_prob = prob
        if prob >= helper.threshold:
            self._last_ml_decision = "EXIT"
            return "ml_exit"
        self._last_ml_decision = "HOLD"
        return None

    def _ml_feature_vector(self, weekday: int, pnl: float, morning_row: pd.Series, close_row: pd.Series) -> Optional[Dict[str, float]]:
        if self.position is None:
            return None
        current_date = pd.to_datetime(close_row["timestamp"]).date()
        entry_date = self.position.entry_date.date()
        target_expiry = self.position.target_expiry.date() if self.position.target_expiry is not None else None
        days_since_entry = (current_date - entry_date).days
        target_days_remaining = (target_expiry - current_date).days if target_expiry else None
        events = self._events_for(current_date)
        event_count = len(events)
        has_event_risk = 1 if event_count > 0 else 0
        drawdown_from_best = (self.position.best_pnl - pnl) if self.position.best_pnl is not None else None
        return {
            "weekday": weekday,
            "spot_open": float(morning_row.get("spot")) if not pd.isna(morning_row.get("spot")) else 0.0,
            "spot_close": float(close_row.get("spot")) if not pd.isna(close_row.get("spot")) else 0.0,
            "combined_premium_pct": float(morning_row.get("combined_premium_pct") or 0.0),
            "iv_rank": float(morning_row.get("iv_rank") or 0.0),
            "iv_skew": float(morning_row.get("iv_skew") or 0.0),
            "oi_skew": float(close_row.get("oi_skew") or 0.0),
            "volume_skew": float(close_row.get("volume_skew") or 0.0),
            "call_oi_max_strike": float(close_row.get("call_oi_max_strike") or 0.0),
            "put_oi_max_strike": float(close_row.get("put_oi_max_strike") or 0.0),
            "spot_return_1": float(close_row.get("spot_return_1") or 0.0),
            "spot_return_3": float(close_row.get("spot_return_3") or 0.0),
            "spot_return_5": float(close_row.get("spot_return_5") or 0.0),
            "spot_trend_20": float(close_row.get("spot_trend_20") or 0.0),
            "spot_trend_50": float(close_row.get("spot_trend_50") or 0.0),
            "spot_trend_100": float(close_row.get("spot_trend_100") or 0.0),
            "spot_volatility": float(close_row.get("spot_volatility") or 0.0),
            "spot_intraday_range_pct": float(close_row.get("spot_intraday_range_pct") or 0.0),
            "call_oi_change": float(close_row.get("call_oi_change") or 0.0),
            "put_oi_change": float(close_row.get("put_oi_change") or 0.0),
            "event_count": float(event_count),
            "has_event_risk": float(has_event_risk),
            "days_since_entry": float(days_since_entry if days_since_entry is not None else 0.0),
            "target_days_remaining": float(target_days_remaining if target_days_remaining is not None else 0.0),
            "current_pnl": float(pnl),
            "best_pnl": float(self.position.best_pnl or 0.0),
            "drawdown_from_best": float(drawdown_from_best or 0.0),
        }


def _coerce_config(cfg_input: Optional[Any]) -> WeeklyConfig:
    if cfg_input is None:
        return WeeklyConfig()
    if isinstance(cfg_input, WeeklyConfig):
        return cfg_input
    data = dict(cfg_input)
    if isinstance(data.get("entry_rules"), dict):
        data["entry_rules"] = EntryRules(**data["entry_rules"])
    if isinstance(data.get("targets"), dict):
        data["targets"] = WeeklyTargets(**data["targets"])
    return WeeklyConfig(**data)


def _selector_field(option: str, offset: int, metric: str) -> str:
    option = option.lower()
    if offset == 0:
        prefix = f"atm_{option}"
    elif offset > 0:
        prefix = f"{option}_atm_plus{offset}"
    else:
        prefix = f"{option}_atm_minus{abs(offset)}"
    return f"{prefix}_{metric}"


def _load_event_calendar(path_str: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        import json

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


def run_backtest(df: pd.DataFrame, cfg_dict: Optional[Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    cfg = _coerce_config(cfg_dict)
    strat = WeeklyThetaStrangle(cfg)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in ("expiryDate", "next_week_expiry"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("days_to_expiry", "days_to_next_expiry"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "source_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["source_date"]).dt.date
    else:
        df["trade_date"] = df["timestamp"].dt.date
    last_rows = None
    for date, day_rows in df.groupby("trade_date"):
        strat.on_new_day(day_rows)
        last_rows = day_rows.sort_values("timestamp")
    if strat.position and last_rows is not None:
        strat._close_position(last_rows.iloc[-1], reason="dataset_end")
    trades_df = pd.DataFrame(strat.trades)
    daily_df = pd.DataFrame(strat.daily_snapshots)
    total_realized = float(trades_df.loc[trades_df["action"] == "CLOSE", "realized"].sum()) if not trades_df.empty else 0.0
    summary = {
        "entries": int((trades_df["action"] == "OPEN").sum()) if not trades_df.empty else 0,
        "closes": int((trades_df["action"] == "CLOSE").sum()) if not trades_df.empty else 0,
        "total_realized": total_realized,
        "avg_per_trade": float(total_realized / max(1, (trades_df["action"] == "CLOSE").sum())) if not trades_df.empty else 0.0,
    }
    return trades_df, daily_df, summary
ML_FEATURES = [
    "weekday",
    "spot_open",
    "spot_close",
    "combined_premium_pct",
    "iv_rank",
    "iv_skew",
    "oi_skew",
    "volume_skew",
    "call_oi_max_strike",
    "put_oi_max_strike",
    "spot_return_1",
    "spot_return_3",
    "spot_return_5",
    "spot_trend_20",
    "spot_trend_50",
    "spot_trend_100",
    "spot_volatility",
    "spot_intraday_range_pct",
    "call_oi_change",
    "put_oi_change",
    "event_count",
    "has_event_risk",
    "days_since_entry",
    "target_days_remaining",
    "current_pnl",
    "best_pnl",
    "drawdown_from_best",
]


class MLExitHelper:
    def __init__(self, model_path: Path, fallback_threshold: float):
        with Path(model_path).open("rb") as fh:
            payload = pickle.load(fh)
        self.model = payload.get("estimator")
        self.feature_names = payload.get("feature_names", ML_FEATURES)
        self.threshold = float(payload.get("prob_threshold", fallback_threshold))
        if self.model is None:
            raise RuntimeError(f"Invalid model payload at {model_path}")

    def predict_prob(self, features: Dict[str, float]) -> float:
        data = {name: [float(features.get(name, 0.0))] for name in self.feature_names}
        X = pd.DataFrame(data)
        return float(self.model.predict_proba(X)[0][1])
