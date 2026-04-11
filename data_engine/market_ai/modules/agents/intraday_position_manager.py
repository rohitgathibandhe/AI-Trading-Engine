from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from market_ai.strategies import LegSide, OptionLeg, OptionType, StrategyType


def _now_ist() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _parse_hhmm(value: Any, fallback: dtime) -> dtime:
    try:
        hh, mm = str(value).strip().split(":")
        return dtime(int(hh), int(mm))
    except Exception:
        return fallback


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, "", "-", "--"):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _profit_protect_profile(volatility_regime: str) -> Tuple[float, float, float]:
    regime = str(volatility_regime or "NORMAL").upper()
    if regime == "HIGH":
        return (45.0, 0.55, 0.72)
    if regime == "LOW":
        return (50.0, 0.45, 0.60)
    return (60.0, 0.38, 0.52)


def _profit_protect_min_hold_minutes(volatility_regime: str) -> float:
    regime = str(volatility_regime or "NORMAL").upper()
    if regime == "HIGH":
        return 10.0
    if regime == "LOW":
        return 10.0
    return 15.0


def _leg_to_payload(leg: OptionLeg) -> Dict[str, Any]:
    return {
        "symbol": str(getattr(leg, "symbol", "") or ""),
        "expiry": leg.expiry.isoformat() if hasattr(getattr(leg, "expiry", None), "isoformat") else str(getattr(leg, "expiry", "") or ""),
        "strike": float(getattr(leg, "strike", 0.0) or 0.0),
        "option_type": getattr(getattr(leg, "option_type", None), "value", getattr(leg, "option_type", "")),
        "side": getattr(getattr(leg, "side", None), "value", getattr(leg, "side", "")),
        "quantity": int(getattr(leg, "quantity", 0) or 0),
        "entry_price": float(getattr(leg, "entry_price", 0.0) or 0.0),
        "security_id": str(getattr(leg, "security_id", "") or ""),
        "strategy_type": getattr(getattr(leg, "strategy_type", None), "name", "NONE"),
    }


def _payload_to_leg(payload: Dict[str, Any]) -> OptionLeg:
    expiry = datetime.fromisoformat(str(payload.get("expiry"))).date()
    opt_raw = str(payload.get("option_type") or "").upper()
    side_raw = str(payload.get("side") or "").upper()
    strategy_name = str(payload.get("strategy_type") or "NONE").upper()
    try:
        strategy_type = StrategyType[strategy_name]
    except Exception:
        strategy_type = StrategyType.NONE
    return OptionLeg(
        symbol=str(payload.get("symbol") or "NIFTY"),
        expiry=expiry,
        strike=float(payload.get("strike") or 0.0),
        option_type=OptionType.CALL if opt_raw == "CE" else OptionType.PUT,
        side=LegSide.BUY if side_raw == "BUY" else LegSide.SELL,
        quantity=int(payload.get("quantity") or 0),
        entry_price=float(payload.get("entry_price") or 0.0),
        security_id=str(payload.get("security_id") or "") or None,
        strategy_type=strategy_type,
    )


def _is_bullish_short_premium_strategy(strategy_type: str) -> bool:
    return str(strategy_type or "").upper() in {"PUT_CREDIT_SPREAD", "SHORT_PUT_WITH_HEDGE"}


def _is_bearish_short_premium_strategy(strategy_type: str) -> bool:
    return str(strategy_type or "").upper() in {"CALL_CREDIT_SPREAD", "SHORT_CALL_WITH_HEDGE"}


def _current_structure_level(
    *,
    strategy_type: str,
    spot: float,
    live_features: Dict[str, Any],
    entry_features: Dict[str, Any],
) -> Optional[float]:
    features = dict(entry_features or {})
    if isinstance(live_features, dict):
        for key in (
            "intraday_support",
            "intraday_resistance",
            "swing_support",
            "swing_resistance",
            "daily_support",
            "daily_resistance",
            "weekly_support",
            "weekly_resistance",
        ):
            if live_features.get(key) not in (None, "", "-", "--"):
                features[key] = live_features.get(key)
    spot_f = float(spot or 0.0)
    if _is_bullish_short_premium_strategy(strategy_type):
        supports = [
            _safe_float(features.get("intraday_support")),
            _safe_float(features.get("swing_support")),
            _safe_float(features.get("daily_support")),
            _safe_float(features.get("weekly_support")),
        ]
        valid = [float(level) for level in supports if level is not None and float(level) <= spot_f]
        return max(valid) if valid else None
    if _is_bearish_short_premium_strategy(strategy_type):
        resistances = [
            _safe_float(features.get("intraday_resistance")),
            _safe_float(features.get("swing_resistance")),
            _safe_float(features.get("daily_resistance")),
            _safe_float(features.get("weekly_resistance")),
        ]
        valid = [float(level) for level in resistances if level is not None and float(level) >= spot_f]
        return min(valid) if valid else None
    return None


@dataclass
class IntradayPositionState:
    status: str = "IDLE"  # IDLE|OPEN|CLOSED
    position_id: Optional[str] = None
    trade_mode: Optional[str] = None
    strategy_type: Optional[str] = None
    strategy_label: Optional[str] = None
    expiry: Optional[str] = None
    opened_at: Optional[str] = None
    closed_at: Optional[str] = None
    current_session_date: str = ""
    last_reason: Optional[str] = None
    entry_spot: Optional[float] = None
    current_spot: Optional[float] = None
    current_pnl_rs: float = 0.0
    peak_pnl_rs: float = 0.0
    trail_floor_rs: Optional[float] = None
    trailing_enabled: bool = True
    trailing_active: bool = False
    sl_total_rs: float = 0.0
    tp_total_rs: float = 0.0
    invalidation_spot_level: Optional[float] = None
    max_hold_till: str = "15:05"
    lots_multiplier: int = 1
    session_trade_count: int = 0
    entry_features: Dict[str, Any] = field(default_factory=dict)
    legs: List[Dict[str, Any]] = field(default_factory=list)
    last_evaluated_at: Optional[str] = None
    updated_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IntradayPositionManager:
    def __init__(self, *, state_path: Path, history_path: Optional[Path] = None, logger: Any = None) -> None:
        self.state_path = state_path
        self.history_path = history_path
        self.log = logger
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.history_path is not None:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        if not self.state.current_session_date:
            self.state.current_session_date = _now_ist().date().isoformat()
            self._persist()

    def _log(self, level: str, msg: str, *args: Any) -> None:
        if self.log and hasattr(self.log, level):
            getattr(self.log, level)(msg, *args)

    def _default_state(self) -> IntradayPositionState:
        now = _now_ist()
        return IntradayPositionState(
            status="IDLE",
            current_session_date=now.date().isoformat(),
            updated_at=now.isoformat(timespec="seconds"),
        )

    def _load_state(self) -> IntradayPositionState:
        if not self.state_path.exists():
            return self._default_state()
        try:
            payload = json.loads(self.state_path.read_text())
            if not isinstance(payload, dict):
                return self._default_state()
            state = IntradayPositionState()
            for key in IntradayPositionState.__dataclass_fields__.keys():
                if key in payload:
                    setattr(state, key, payload.get(key))
            if not isinstance(state.legs, list):
                state.legs = []
            if not isinstance(state.entry_features, dict):
                state.entry_features = {}
            return state
        except Exception:
            return self._default_state()

    def _persist(self, when: Optional[datetime] = None) -> None:
        now = when or _now_ist()
        self.state.updated_at = now.isoformat(timespec="seconds")
        if not self.state.current_session_date:
            self.state.current_session_date = now.date().isoformat()
        try:
            self.state_path.write_text(json.dumps(self.state.as_dict(), indent=2, default=str))
        except Exception:
            self._log("exception", "Failed to persist intraday position state")

    def snapshot(self) -> Dict[str, Any]:
        return self.state.as_dict()

    def _append_history(self, payload: Dict[str, Any]) -> None:
        if self.history_path is None:
            return
        try:
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            self._log("exception", "Failed to append intraday trade history")

    def has_open_position(self) -> bool:
        return str(self.state.status or "").upper() == "OPEN" and bool(self.state.legs)

    def build_option_legs(self) -> List[OptionLeg]:
        out: List[OptionLeg] = []
        for payload in self.state.legs or []:
            if not isinstance(payload, dict):
                continue
            try:
                out.append(_payload_to_leg(payload))
            except Exception:
                continue
        return out

    def session_trade_count(self, *, now: Optional[datetime] = None) -> int:
        ts = now or _now_ist()
        if str(self.state.current_session_date or "") != ts.date().isoformat():
            return 0
        return max(0, int(self.state.session_trade_count or 0))

    def can_open_new_position(self, *, max_trades_per_session: int, now: Optional[datetime] = None) -> bool:
        if self.has_open_position():
            return False
        allowed = max(1, int(max_trades_per_session or 1))
        return self.session_trade_count(now=now) < allowed

    def open_position(
        self,
        *,
        position_id: str,
        trade_mode: str,
        strategy_type: str,
        strategy_label: str,
        expiry: str,
        legs: List[OptionLeg],
        entry_spot: float,
        sl_total_rs: float,
        tp_total_rs: float,
        invalidation_spot_level: Optional[float],
        max_hold_till: str,
        trailing_enabled: bool,
        lots_multiplier: int,
        entry_features: Optional[Dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        ts = now or _now_ist()
        same_session = str(self.state.current_session_date or "") == ts.date().isoformat()
        next_trade_count = (max(0, int(self.state.session_trade_count or 0)) if same_session else 0) + 1
        self.state = IntradayPositionState(
            status="OPEN",
            position_id=str(position_id or ""),
            trade_mode=str(trade_mode or ""),
            strategy_type=str(strategy_type or ""),
            strategy_label=str(strategy_label or ""),
            expiry=str(expiry or ""),
            opened_at=ts.isoformat(timespec="seconds"),
            closed_at=None,
            current_session_date=ts.date().isoformat(),
            last_reason="OPENED",
            entry_spot=float(entry_spot or 0.0),
            current_spot=float(entry_spot or 0.0),
            current_pnl_rs=0.0,
            peak_pnl_rs=0.0,
            trail_floor_rs=None,
            trailing_enabled=bool(trailing_enabled),
            trailing_active=False,
            sl_total_rs=max(0.0, float(sl_total_rs or 0.0)),
            tp_total_rs=max(0.0, float(tp_total_rs or 0.0)),
            invalidation_spot_level=(
                None if invalidation_spot_level is None else float(invalidation_spot_level)
            ),
            max_hold_till=str(max_hold_till or "15:05"),
            lots_multiplier=max(1, int(lots_multiplier or 1)),
            session_trade_count=next_trade_count,
            entry_features=dict(entry_features or {}),
            legs=[_leg_to_payload(leg) for leg in (legs or [])],
            last_evaluated_at=None,
        )
        self._persist(ts)
        return self.snapshot()

    def close_position(
        self,
        *,
        reason: str,
        pnl_rs: float,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        ts = now or _now_ist()
        opened_at = _parse_dt(self.state.opened_at)
        hold_minutes: Optional[float] = None
        if opened_at is not None:
            hold_minutes = max(0.0, round((ts - opened_at).total_seconds() / 60.0, 2))
        self.state.status = "CLOSED"
        self.state.closed_at = ts.isoformat(timespec="seconds")
        self.state.last_reason = str(reason or "CLOSED")
        self.state.current_pnl_rs = float(pnl_rs or 0.0)
        self.state.current_spot = None if self.state.current_spot is None else float(self.state.current_spot)
        self.state.last_evaluated_at = ts.isoformat(timespec="seconds")
        self._persist(ts)
        history_row = {
            "position_id": self.state.position_id,
            "trade_mode": self.state.trade_mode,
            "strategy_type": self.state.strategy_type,
            "strategy_label": self.state.strategy_label,
            "expiry": self.state.expiry,
            "opened_at": self.state.opened_at,
            "closed_at": self.state.closed_at,
            "current_session_date": self.state.current_session_date,
            "entry_spot": self.state.entry_spot,
            "exit_spot": self.state.current_spot,
            "pnl_rs": round(float(self.state.current_pnl_rs or 0.0), 2),
            "peak_pnl_rs": round(float(self.state.peak_pnl_rs or 0.0), 2),
            "trail_floor_rs": None if self.state.trail_floor_rs is None else round(float(self.state.trail_floor_rs), 2),
            "sl_total_rs": round(float(self.state.sl_total_rs or 0.0), 2),
            "tp_total_rs": round(float(self.state.tp_total_rs or 0.0), 2),
            "invalidation_spot_level": self.state.invalidation_spot_level,
            "hold_minutes": hold_minutes,
            "reason": self.state.last_reason,
            "lots_multiplier": self.state.lots_multiplier,
            "session_trade_count": self.state.session_trade_count,
            "trailing_enabled": bool(self.state.trailing_enabled),
            "trailing_active": bool(self.state.trailing_active),
            "entry_features": dict(self.state.entry_features or {}),
            "result": (
                "WIN"
                if float(self.state.current_pnl_rs or 0.0) > 0
                else ("LOSS" if float(self.state.current_pnl_rs or 0.0) < 0 else "FLAT")
            ),
            "legs": list(self.state.legs or []),
        }
        self._append_history(history_row)
        snap = self.snapshot()
        snap["history_row"] = history_row
        return snap

    def reset(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        self.state = self._default_state()
        self._persist(now)
        return self.snapshot()

    def evaluate(
        self,
        *,
        now: datetime,
        spot: float,
        chain_rows: List[Dict[str, Any]],
        signal_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.has_open_position():
            self._persist(now)
            return {"has_position": False, "action": "NONE", "reason": "NO_OPEN_POSITION"}

        legs = self.build_option_legs()
        if not legs:
            self.state.last_reason = "NO_POSITION_LEGS"
            self._persist(now)
            return {"has_position": False, "action": "NONE", "reason": "NO_POSITION_LEGS"}

        lookup: Dict[Tuple[str, float], float] = {}
        for row in chain_rows or []:
            try:
                opt = str(row.get("option_type") or "").upper()
                strike = float(row.get("strike") or 0.0)
                ltp = _safe_float(row.get("ltp"))
                if opt in {"CE", "PE"} and strike > 0 and ltp is not None:
                    lookup[(opt, strike)] = float(ltp)
            except Exception:
                continue

        pnl_rs = 0.0
        missing_quotes = 0
        for leg in legs:
            opt = leg.option_type.value if hasattr(leg.option_type, "value") else str(leg.option_type)
            ltp = lookup.get((str(opt).upper(), float(leg.strike)))
            if ltp is None:
                ltp = float(getattr(leg, "entry_price", 0.0) or 0.0)
                missing_quotes += 1
            sign = 1.0 if leg.side == LegSide.BUY else -1.0
            pnl_rs += sign * (float(ltp) - float(leg.entry_price or 0.0)) * float(leg.quantity or 0)

        self.state.current_pnl_rs = round(float(pnl_rs), 2)
        self.state.current_spot = round(float(spot or 0.0), 2)
        self.state.last_evaluated_at = now.isoformat(timespec="seconds")
        self.state.peak_pnl_rs = max(float(self.state.peak_pnl_rs or 0.0), float(self.state.current_pnl_rs or 0.0))

        close_reason: Optional[str] = None
        strategy_type = str(self.state.strategy_type or "").upper()
        entry_features = self.state.entry_features if isinstance(self.state.entry_features, dict) else {}
        live_entry_features = (
            signal_payload.get("entry_features")
            if isinstance(signal_payload, dict) and isinstance(signal_payload.get("entry_features"), dict)
            else {}
        )
        invalidation = _safe_float(self.state.invalidation_spot_level)
        tp_total = max(0.0, float(self.state.tp_total_rs or 0.0))
        sl_total = max(0.0, float(self.state.sl_total_rs or 0.0))
        current_pnl = float(self.state.current_pnl_rs or 0.0)
        peak_pnl = max(0.0, float(self.state.peak_pnl_rs or 0.0))
        opened_at = _parse_dt(self.state.opened_at)
        hold_minutes_live = (
            max(0.0, (now - opened_at).total_seconds() / 60.0)
            if opened_at is not None
            else 0.0
        )
        signal_vol_regime = (
            str((signal_payload or {}).get("volatility_regime") or "").upper()
            if isinstance(signal_payload, dict)
            else ""
        )
        position_vol_regime = signal_vol_regime or str(entry_features.get("volatility_regime") or "NORMAL").upper()
        arm_threshold, keep_frac, degrade_keep_frac = _profit_protect_profile(position_vol_regime)
        min_arm_hold_minutes = _profit_protect_min_hold_minutes(position_vol_regime)
        unlimited_profit_mode = bool(entry_features.get("unlimited_profit_trailing"))
        trail_arm_profit_rs = max(
            0.0,
            float(_safe_float(entry_features.get("profit_trail_arm_rs"), 0.0) or 0.0),
        )
        trail_floor_min_profit_rs = max(
            0.0,
            float(_safe_float(entry_features.get("trail_floor_min_profit_rs"), trail_arm_profit_rs) or trail_arm_profit_rs or 0.0),
        )
        configured_keep_pct = _safe_float(entry_features.get("trail_keep_pct"))
        if configured_keep_pct is not None:
            keep_frac = max(0.25, min(0.95, float(configured_keep_pct)))
        effective_arm_threshold = trail_arm_profit_rs if unlimited_profit_mode and trail_arm_profit_rs > 0 else arm_threshold
        live_structure_level = _current_structure_level(
            strategy_type=strategy_type,
            spot=float(spot or 0.0),
            live_features=live_entry_features if isinstance(live_entry_features, dict) else {},
            entry_features=entry_features,
        )
        structure_buffer = max(
            12.0,
            float(_safe_float(entry_features.get("trade_plan_pullback_zone_high")) or 0.0) * 0.0004,
            float(_safe_float(entry_features.get("trade_plan_pullback_zone_low")) or 0.0) * 0.0004,
        )

        if sl_total > 0 and self.state.current_pnl_rs <= -sl_total:
            close_reason = "SL_HIT"

        if close_reason is None and invalidation is not None:
            if _is_bearish_short_premium_strategy(strategy_type) and float(spot or 0.0) >= invalidation:
                close_reason = "SPOT_INVALIDATION"
            elif _is_bullish_short_premium_strategy(strategy_type) and float(spot or 0.0) <= invalidation:
                close_reason = "SPOT_INVALIDATION"

        if close_reason is None and unlimited_profit_mode and self.state.trailing_enabled:
            if peak_pnl >= trail_arm_profit_rs > 0 and hold_minutes_live >= min_arm_hold_minutes:
                self.state.trailing_active = True
            if self.state.trailing_active:
                candidate_floor = max(trail_floor_min_profit_rs, peak_pnl * keep_frac)
                prev_floor = _safe_float(self.state.trail_floor_rs)
                self.state.trail_floor_rs = (
                    candidate_floor if prev_floor is None else max(float(prev_floor), candidate_floor)
                )
                if live_structure_level is not None:
                    if _is_bearish_short_premium_strategy(strategy_type) and float(spot or 0.0) >= float(live_structure_level) + structure_buffer:
                        close_reason = "STRUCTURE_TRAIL_BREAK"
                    elif _is_bullish_short_premium_strategy(strategy_type) and float(spot or 0.0) <= float(live_structure_level) - structure_buffer:
                        close_reason = "STRUCTURE_TRAIL_BREAK"
                if close_reason is None and current_pnl <= float(self.state.trail_floor_rs or 0.0):
                    close_reason = "PROFIT_TRAIL_HIT"
        elif close_reason is None and tp_total > 0:
            if self.state.trailing_enabled:
                dynamic_arm = max(arm_threshold, min(220.0, tp_total * 0.05 if tp_total > 0 else arm_threshold))
                if current_pnl >= dynamic_arm and hold_minutes_live >= min_arm_hold_minutes:
                    self.state.trailing_active = True
                if self.state.trailing_active:
                    candidate_floor = max(15.0, peak_pnl * keep_frac)
                    prev_floor = _safe_float(self.state.trail_floor_rs)
                    self.state.trail_floor_rs = (
                        candidate_floor if prev_floor is None else max(float(prev_floor), candidate_floor)
                    )
                    if current_pnl <= float(self.state.trail_floor_rs or 0.0):
                        close_reason = "TRAIL_FLOOR_HIT"
            elif self.state.current_pnl_rs >= tp_total:
                close_reason = "TARGET_HIT"

        if close_reason is None and signal_payload:
            expected_bias = "BEARISH" if _is_bearish_short_premium_strategy(strategy_type) else ("BULLISH" if _is_bullish_short_premium_strategy(strategy_type) else "NEUTRAL")
            opposite_bias = "BULLISH" if expected_bias == "BEARISH" else ("BEARISH" if expected_bias == "BULLISH" else "NEUTRAL")
            market_bias = str(signal_payload.get("market_bias") or "NEUTRAL").upper()
            breakout = str(signal_payload.get("breakout_confirmation") or "NONE").upper()
            trend_conf = _safe_float(signal_payload.get("trend_confidence"), 0.0) or 0.0
            conflict = _safe_float(signal_payload.get("signal_conflict_score"), 100.0) or 100.0
            price_action_bias = str(signal_payload.get("price_action_bias") or "NEUTRAL").upper()
            price_action_confirmation = str(signal_payload.get("price_action_confirmation") or "NONE").upper()
            retest_status = str(signal_payload.get("retest_status") or "NONE").upper()
            pcr_bias = str(signal_payload.get("pcr_bias") or "NEUTRAL").upper()
            oi_build_bias = str(signal_payload.get("oi_build_bias") or "UNKNOWN").upper()
            pcr_unbalanced_side = str(signal_payload.get("pcr_unbalanced_side") or "NEUTRAL").upper()

            opposing_price_action = (
                expected_bias in {"BULLISH", "BEARISH"}
                and price_action_bias == opposite_bias
                and price_action_confirmation
                in {
                    "CANDLE_CONFIRMED",
                    "RETEST_CONFIRMED",
                    "CANDLE_AND_RETEST_CONFIRMED",
                    "REVERSAL_CONFIRMED",
                    "BREAKOUT_CONFIRMED",
                }
            )
            opposing_retest = False
            if _is_bearish_short_premium_strategy(strategy_type):
                opposing_retest = retest_status in {"SUPPORT_HOLD", "BREAKOUT_RETEST_HOLD"}
            elif _is_bullish_short_premium_strategy(strategy_type):
                opposing_retest = retest_status in {"RESISTANCE_HOLD", "BREAKDOWN_RETEST_HOLD"}
            chain_against = (
                expected_bias in {"BULLISH", "BEARISH"}
                and (
                    pcr_bias == opposite_bias
                    or pcr_unbalanced_side == opposite_bias
                    or opposite_bias in oi_build_bias
                )
            )
            weakening_signal = (
                (market_bias == opposite_bias and breakout in {"UP_CONFIRMED", "DOWN_CONFIRMED"})
                or opposing_price_action
                or opposing_retest
                or chain_against
            )
            giveback_floor = max(25.0, peak_pnl * keep_frac) if peak_pnl > 0 else 0.0
            if unlimited_profit_mode and trail_floor_min_profit_rs > 0:
                giveback_floor = max(giveback_floor, trail_floor_min_profit_rs)

            if (
                opposing_price_action
                and opposing_retest
                and (
                    position_vol_regime == "HIGH"
                    or current_pnl <= max(100.0, tp_total * 0.15 if tp_total > 0 else 100.0)
                )
            ):
                close_reason = "PRICE_ACTION_REVERSAL"
            elif (
                peak_pnl >= effective_arm_threshold
                and hold_minutes_live >= min_arm_hold_minutes
                and weakening_signal
                and current_pnl <= max(giveback_floor, peak_pnl * degrade_keep_frac)
            ):
                close_reason = "PROFIT_PROTECT"
            elif chain_against and opposing_price_action and current_pnl < 0 and (
                conflict >= 45.0 or position_vol_regime == "HIGH"
            ):
                close_reason = "EARLY_RISK_OFF"
            elif chain_against and current_pnl <= -max(
                80.0,
                sl_total * (0.10 if position_vol_regime == "HIGH" else 0.18) if sl_total > 0 else 100.0,
            ) and (
                conflict >= 50.0 or position_vol_regime == "HIGH"
            ):
                close_reason = "CHAIN_CONFLICT_EXIT"

            if (
                close_reason is None
                and self.state.trailing_enabled
                and peak_pnl >= effective_arm_threshold
                and hold_minutes_live >= min_arm_hold_minutes
                and weakening_signal
            ):
                tightened_floor = max(15.0, peak_pnl * degrade_keep_frac)
                if unlimited_profit_mode and trail_floor_min_profit_rs > 0:
                    tightened_floor = max(tightened_floor, trail_floor_min_profit_rs)
                prev_floor = _safe_float(self.state.trail_floor_rs)
                self.state.trail_floor_rs = tightened_floor if prev_floor is None else max(float(prev_floor), tightened_floor)
                self.state.trailing_active = True
                if current_pnl <= float(self.state.trail_floor_rs or 0.0):
                    close_reason = "PROFIT_PROTECT"

        if close_reason is None and signal_payload:
            market_bias = str(signal_payload.get("market_bias") or "NEUTRAL").upper()
            breakout = str(signal_payload.get("breakout_confirmation") or "NONE").upper()
            trend_conf = _safe_float(signal_payload.get("trend_confidence"), 0.0) or 0.0
            conflict = _safe_float(signal_payload.get("signal_conflict_score"), 100.0) or 100.0
            if _is_bearish_short_premium_strategy(strategy_type) and market_bias == "BULLISH" and breakout == "UP_CONFIRMED" and trend_conf >= 0.70 and conflict <= 43.5:
                close_reason = "SIGNAL_REVERSAL"
            elif _is_bullish_short_premium_strategy(strategy_type) and market_bias == "BEARISH" and breakout == "DOWN_CONFIRMED" and trend_conf >= 0.70 and conflict <= 43.5:
                close_reason = "SIGNAL_REVERSAL"

        hold_till = _parse_hhmm(self.state.max_hold_till, dtime(15, 5))
        now_time = now.timetz().replace(tzinfo=None) if getattr(now, "tzinfo", None) else now.time()
        if close_reason is None and now.date().isoformat() != str(self.state.current_session_date or ""):
            close_reason = "SESSION_ROLLOVER"
        if close_reason is None and now_time >= hold_till:
            close_reason = "TIME_EXIT"

        self.state.last_reason = close_reason or "HOLD"
        self._persist(now)
        return {
            "has_position": True,
            "action": "CLOSE" if close_reason else "HOLD",
            "reason": close_reason or "HOLD",
            "pnl_rs": round(float(self.state.current_pnl_rs or 0.0), 2),
            "peak_pnl_rs": round(float(self.state.peak_pnl_rs or 0.0), 2),
            "trail_floor_rs": None if self.state.trail_floor_rs is None else round(float(self.state.trail_floor_rs), 2),
            "trailing_active": bool(self.state.trailing_active),
            "missing_quotes": int(missing_quotes),
            "expiry": self.state.expiry,
        }
