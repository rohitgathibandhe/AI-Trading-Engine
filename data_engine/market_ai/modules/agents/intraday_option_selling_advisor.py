from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from market_ai.modules.analytics.intraday_trade_planner import (
    build_trade_plan_from_runtime_context,
)

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _now_ist() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


def _parse_hhmm(value: Any, fallback: dtime) -> dtime:
    try:
        hh, mm = str(value).strip().split(":")
        return dtime(int(hh), int(mm))
    except Exception:
        return fallback


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "-", "--"):
        return None
    try:
        return float(value)
    except Exception:
        return None


@dataclass
class IntradayOptionSellingAdvisorConfig:
    enabled: bool = True
    mode: str = "ADVISOR"
    refresh_sec: float = 60.0
    market_open_time: str = "09:15"
    entry_not_before: str = "09:45"
    last_new_entry_time: str = "14:20"
    max_hold_till: str = "15:05"
    lot_size: int = 65
    allow_parallel_with_bkm: bool = False
    no_trade_conflict_threshold: float = 60.0
    directional_max_conflict: float = 50.0
    range_max_conflict: float = 35.0
    directional_min_trend_confidence: float = 0.62
    range_max_trend_confidence: float = 0.58
    min_sr_distance_points: float = 160.0
    sr_safety_buffer_points: float = 80.0
    fallback_otm_buffer_points: float = 220.0
    spread_width_points_low_vol: int = 150
    spread_width_points_normal_vol: int = 200
    spread_width_points_high_vol: int = 250
    ic_target_capture_pct: float = 0.30
    spread_target_capture_pct: float = 0.40
    ic_stop_credit_multiple: float = 1.8
    spread_stop_credit_multiple: float = 1.6
    min_credit_per_set_rs: float = 500.0
    directional_structure: str = "CREDIT_SPREAD"  # CREDIT_SPREAD|SHORT_OPTION_WITH_HEDGE
    emergency_hedge_distance_points_low_vol: int = 300
    emergency_hedge_distance_points_normal_vol: int = 400
    emergency_hedge_distance_points_high_vol: int = 500
    naked_operational_max_loss_rs: float = 3000.0
    naked_profit_trail_arm_rs: float = 5000.0
    naked_profit_trail_keep_pct: float = 0.72
    max_signal_age_sec: float = 180.0
    preferred_bias: str = "NEUTRAL"  # NEUTRAL|BULLISH|BEARISH
    require_ema_alignment: bool = True
    require_orb_confirmation: bool = True
    require_price_action_confirmation: bool = True
    signal_persistence_bars: int = 2
    trend_day_fast_track_enabled: bool = True
    trend_day_fast_track_min_trend_confidence: float = 0.78
    trend_day_fast_track_max_conflict: float = 45.0
    directional_flip_block_enabled: bool = True
    directional_flip_block_min_prev_trend_confidence: float = 0.72
    directional_flip_block_max_prev_conflict: float = 45.0
    directional_flip_override_min_trend_delta: float = 0.08
    directional_flip_override_min_conflict_improvement: float = 8.0
    high_vol_bear_call_not_before: str = "10:15"
    high_vol_bull_put_enabled_for_bearish_preference: bool = False

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "IntradayOptionSellingAdvisorConfig":
        lot_size = int(settings.get("nifty_lot_size", settings.get("lot_size", 65)) or 65)
        return cls(
            enabled=bool(settings.get("intraday_ai_enabled", True)),
            mode=str(settings.get("intraday_ai_mode", "ADVISOR")).strip().upper() or "ADVISOR",
            refresh_sec=max(15.0, float(settings.get("intraday_ai_refresh_sec", 60.0))),
            market_open_time=str(settings.get("intraday_ai_market_open_time", "09:15")),
            entry_not_before=str(settings.get("intraday_ai_entry_not_before", "09:45")),
            last_new_entry_time=str(settings.get("intraday_ai_last_new_entry_time", "14:20")),
            max_hold_till=str(settings.get("intraday_ai_max_hold_till", "15:05")),
            lot_size=max(1, lot_size),
            allow_parallel_with_bkm=bool(settings.get("intraday_ai_allow_parallel_with_bkm", False)),
            no_trade_conflict_threshold=max(0.0, float(settings.get("intraday_ai_no_trade_conflict_threshold", 60.0))),
            directional_max_conflict=max(0.0, float(settings.get("intraday_ai_directional_max_conflict", 50.0))),
            range_max_conflict=max(0.0, float(settings.get("intraday_ai_range_max_conflict", 35.0))),
            directional_min_trend_confidence=max(
                0.0, min(1.0, float(settings.get("intraday_ai_directional_min_trend_confidence", 0.62)))
            ),
            range_max_trend_confidence=max(
                0.0, min(1.0, float(settings.get("intraday_ai_range_max_trend_confidence", 0.58)))
            ),
            min_sr_distance_points=max(50.0, float(settings.get("intraday_ai_min_sr_distance_points", 160.0))),
            sr_safety_buffer_points=max(20.0, float(settings.get("intraday_ai_sr_safety_buffer_points", 80.0))),
            fallback_otm_buffer_points=max(50.0, float(settings.get("intraday_ai_fallback_otm_buffer_points", 220.0))),
            spread_width_points_low_vol=max(50, int(settings.get("intraday_ai_spread_width_points_low_vol", 150))),
            spread_width_points_normal_vol=max(50, int(settings.get("intraday_ai_spread_width_points_normal_vol", 200))),
            spread_width_points_high_vol=max(50, int(settings.get("intraday_ai_spread_width_points_high_vol", 250))),
            ic_target_capture_pct=max(0.05, min(0.95, float(settings.get("intraday_ai_ic_target_capture_pct", 0.30)))),
            spread_target_capture_pct=max(
                0.05, min(0.95, float(settings.get("intraday_ai_spread_target_capture_pct", 0.40)))
            ),
            ic_stop_credit_multiple=max(0.5, float(settings.get("intraday_ai_ic_stop_credit_multiple", 1.8))),
            spread_stop_credit_multiple=max(
                0.5, float(settings.get("intraday_ai_spread_stop_credit_multiple", 1.6))
            ),
            min_credit_per_set_rs=max(0.0, float(settings.get("intraday_ai_min_credit_per_set_rs", 500.0))),
            directional_structure=str(
                settings.get("intraday_ai_directional_structure", "CREDIT_SPREAD")
            ).strip().upper()
            or "CREDIT_SPREAD",
            emergency_hedge_distance_points_low_vol=max(
                100,
                int(settings.get("intraday_ai_emergency_hedge_distance_points_low_vol", 300) or 300),
            ),
            emergency_hedge_distance_points_normal_vol=max(
                100,
                int(settings.get("intraday_ai_emergency_hedge_distance_points_normal_vol", 400) or 400),
            ),
            emergency_hedge_distance_points_high_vol=max(
                100,
                int(settings.get("intraday_ai_emergency_hedge_distance_points_high_vol", 500) or 500),
            ),
            naked_operational_max_loss_rs=max(
                500.0, float(settings.get("intraday_ai_naked_operational_max_loss_rs", 3000.0))
            ),
            naked_profit_trail_arm_rs=max(
                500.0, float(settings.get("intraday_ai_naked_profit_trail_arm_rs", 5000.0))
            ),
            naked_profit_trail_keep_pct=max(
                0.35,
                min(0.95, float(settings.get("intraday_ai_naked_profit_trail_keep_pct", 0.72))),
            ),
            max_signal_age_sec=max(10.0, float(settings.get("intraday_ai_max_signal_age_sec", 180.0))),
            preferred_bias=str(settings.get("intraday_ai_preferred_bias", "NEUTRAL")).upper(),
            require_ema_alignment=bool(settings.get("intraday_ai_require_ema_alignment", True)),
            require_orb_confirmation=bool(settings.get("intraday_ai_require_orb_confirmation", True)),
            require_price_action_confirmation=bool(
                settings.get("intraday_ai_require_price_action_confirmation", True)
            ),
            signal_persistence_bars=max(1, int(settings.get("intraday_ai_signal_persistence_bars", 2) or 2)),
            trend_day_fast_track_enabled=bool(settings.get("intraday_ai_trend_day_fast_track_enabled", True)),
            trend_day_fast_track_min_trend_confidence=max(
                0.0,
                min(1.0, float(settings.get("intraday_ai_trend_day_fast_track_min_trend_confidence", 0.78))),
            ),
            trend_day_fast_track_max_conflict=max(
                0.0, float(settings.get("intraday_ai_trend_day_fast_track_max_conflict", 45.0))
            ),
            directional_flip_block_enabled=bool(settings.get("intraday_ai_directional_flip_block_enabled", True)),
            directional_flip_block_min_prev_trend_confidence=max(
                0.0,
                min(
                    1.0,
                    float(settings.get("intraday_ai_directional_flip_block_min_prev_trend_confidence", 0.72)),
                ),
            ),
            directional_flip_block_max_prev_conflict=max(
                0.0, float(settings.get("intraday_ai_directional_flip_block_max_prev_conflict", 45.0))
            ),
            directional_flip_override_min_trend_delta=max(
                0.0, float(settings.get("intraday_ai_directional_flip_override_min_trend_delta", 0.08))
            ),
            directional_flip_override_min_conflict_improvement=max(
                0.0, float(settings.get("intraday_ai_directional_flip_override_min_conflict_improvement", 8.0))
            ),
            high_vol_bear_call_not_before=str(settings.get("intraday_ai_high_vol_bear_call_not_before", "10:15")),
            high_vol_bull_put_enabled_for_bearish_preference=bool(
                settings.get("intraday_ai_high_vol_bull_put_enabled_for_bearish_preference", False)
            ),
        )


@dataclass
class IntradayOptionSellingAdvisorState:
    status: str = "IDLE"  # IDLE|ACTIVE
    mode: str = "ADVISOR"
    signal: str = "NO_TRADE"  # ENTER_NOW|WAIT|NO_TRADE
    priority: str = "INFO"  # INFO|WARN|CRITICAL
    strategy: Optional[str] = None
    expiry: Optional[str] = None
    spot: Optional[float] = None
    market_bias: str = "NEUTRAL"
    trend_confidence: float = 0.0
    signal_conflict_score: float = 0.0
    pcr_bias: str = "NEUTRAL"
    oi_build_bias: str = "UNKNOWN"
    pcr_unbalanced: bool = False
    pcr_unbalanced_side: str = "NEUTRAL"
    volatility_regime: str = "UNKNOWN"
    breakout_confirmation: str = "NONE"
    ema_alignment_5m: str = "NEUTRAL"
    ema_alignment_15m: str = "NEUTRAL"
    orb_confirmation: str = "NONE"
    price_action_bias: str = "NEUTRAL"
    price_action_confirmation: str = "NONE"
    retest_status: str = "NONE"
    has_open_bkm: bool = False
    market_context: Dict[str, Any] = None  # type: ignore[assignment]
    recommendation: Dict[str, Any] = None  # type: ignore[assignment]
    reasons: List[str] = None  # type: ignore[assignment]
    candidate_setup_signature: Optional[str] = None
    candidate_signal_streak: int = 0
    current_session_date: str = ""
    updated_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if payload.get("recommendation") is None:
            payload["recommendation"] = {}
        if payload.get("market_context") is None:
            payload["market_context"] = {}
        if payload.get("reasons") is None:
            payload["reasons"] = []
        return payload


class IntradayOptionSellingAdvisor:
    def __init__(self, *, config: IntradayOptionSellingAdvisorConfig, status_path: Path, logger: Any = None) -> None:
        self.config = config
        self.status_path = status_path
        self.log = logger
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self._last_eval_ts: Optional[float] = None

    def _log(self, level: str, msg: str, *args: Any) -> None:
        if self.log and hasattr(self.log, level):
            getattr(self.log, level)(msg, *args)

    def _load_state(self) -> IntradayOptionSellingAdvisorState:
        if not self.status_path.exists():
            st = IntradayOptionSellingAdvisorState()
            st.reasons = []
            st.recommendation = {}
            st.current_session_date = _now_ist().date().isoformat()
            return st
        try:
            payload = json.loads(self.status_path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("invalid intraday advisor payload")
            st = IntradayOptionSellingAdvisorState()
            for k in IntradayOptionSellingAdvisorState.__dataclass_fields__.keys():
                if k in payload:
                    setattr(st, k, payload[k])
            if not isinstance(st.reasons, list):
                st.reasons = []
            if not isinstance(st.recommendation, dict):
                st.recommendation = {}
            if not isinstance(st.market_context, dict):
                st.market_context = {}
            return st
        except Exception:
            st = IntradayOptionSellingAdvisorState()
            st.reasons = []
            st.recommendation = {}
            st.market_context = {}
            st.current_session_date = _now_ist().date().isoformat()
            return st

    def _persist(self, when: Optional[datetime] = None) -> None:
        now = when or _now_ist()
        self.state.updated_at = now.isoformat(timespec="seconds")
        if not self.state.current_session_date:
            self.state.current_session_date = now.date().isoformat()
        try:
            self.status_path.write_text(json.dumps(self.state.as_dict(), indent=2, default=str))
        except Exception:
            self._log("exception", "Failed to persist intraday advisor status")

    def snapshot(self) -> Dict[str, Any]:
        return self.state.as_dict()

    def _strike_step(self, strikes: List[float]) -> float:
        uniq = sorted(set(float(s) for s in strikes if s is not None))
        if len(uniq) < 2:
            return 50.0
        diffs = [round(uniq[i + 1] - uniq[i], 6) for i in range(len(uniq) - 1) if (uniq[i + 1] - uniq[i]) > 0]
        return float(median(diffs)) if diffs else 50.0

    def _pick_leq(self, strikes: List[float], target: float) -> Optional[float]:
        cands = [s for s in strikes if s <= target]
        return max(cands) if cands else None

    def _pick_geq(self, strikes: List[float], target: float) -> Optional[float]:
        cands = [s for s in strikes if s >= target]
        return min(cands) if cands else None

    def _row_lookup(self, chain_rows: List[Dict[str, Any]]) -> Dict[Tuple[str, float], Dict[str, Any]]:
        out: Dict[Tuple[str, float], Dict[str, Any]] = {}
        for row in chain_rows or []:
            try:
                opt = str(row.get("option_type") or "").upper()
                strike = float(row.get("strike") or 0.0)
                if opt not in {"CE", "PE"} or strike <= 0:
                    continue
                out[(opt, strike)] = row
            except Exception:
                continue
        return out

    def _ltp(self, lookup: Dict[Tuple[str, float], Dict[str, Any]], opt: str, strike: float) -> Optional[float]:
        row = lookup.get((opt.upper(), float(strike)))
        if not row:
            return None
        return _safe_float(row.get("ltp"))

    def _vol_regime(self, trend_ctx: Dict[str, Any], structure_ctx: Dict[str, Any]) -> str:
        return str(
            (structure_ctx or {}).get("volatility_regime")
            or (trend_ctx or {}).get("volatility_regime")
            or "NORMAL"
        ).upper()

    def _spread_width(self, vol_regime: str, strike_step: float) -> float:
        if vol_regime == "HIGH":
            width = float(self.config.spread_width_points_high_vol)
        elif vol_regime == "LOW":
            width = float(self.config.spread_width_points_low_vol)
        else:
            width = float(self.config.spread_width_points_normal_vol)
        step = max(1.0, strike_step)
        return max(step, round(width / step) * step)

    def _directional_thresholds(
        self,
        *,
        market_bias: str,
        breakout_confirmation: str,
        trend_confidence: float,
        signal_conflict_score: float,
        tf_map: Dict[str, Any],
        oc_ctx: Dict[str, Any],
    ) -> Tuple[float, float, bool]:
        min_trend_conf = float(self.config.directional_min_trend_confidence)
        max_conflict = float(self.config.directional_max_conflict)
        tf15 = tf_map.get("15") if isinstance(tf_map.get("15"), dict) else {}
        tf60 = tf_map.get("60") if isinstance(tf_map.get("60"), dict) else {}
        pat15 = str(tf15.get("pattern") or "").upper()
        pat60 = str(tf60.get("pattern") or "").upper()
        pcr_bias = str(oc_ctx.get("pcr_bias") or "").upper()
        oi_build = oc_ctx.get("oi_build") if isinstance(oc_ctx.get("oi_build"), dict) else {}
        oi_bias = str(oi_build.get("bias") or "").upper()

        strong_directional_alignment = False
        if market_bias == "BEARISH":
            strong_directional_alignment = (
                breakout_confirmation == "DOWN_CONFIRMED"
                and pat15 == "DOWNTREND"
                and pat60 == "DOWNTREND"
                and (pcr_bias in {"", "BEARISH"} or "BEARISH" in oi_bias)
            )
        elif market_bias == "BULLISH":
            strong_directional_alignment = (
                breakout_confirmation == "UP_CONFIRMED"
                and pat15 == "UPTREND"
                and pat60 == "UPTREND"
                and (pcr_bias in {"", "BULLISH"} or "BULLISH" in oi_bias)
            )

        if strong_directional_alignment:
            min_trend_conf = max(0.50, min_trend_conf - 0.06)
            max_conflict = min(58.0, max_conflict + 8.0)

        return min_trend_conf, max_conflict, strong_directional_alignment

    def _dynamic_credit_floor(
        self,
        *,
        setup_type: str,
        base_credit_floor_rs: float,
        vol_width: float,
        volatility_regime: str,
        breakout_confirmation: str,
        trend_confidence: float,
        signal_conflict_score: float,
        strong_directional_alignment: bool,
    ) -> float:
        width_ref = max(50.0, float(self.config.spread_width_points_normal_vol))
        floor_rs = float(base_credit_floor_rs)
        width_factor = max(0.75, min(1.30, float(vol_width) / width_ref))
        floor_rs *= width_factor

        if setup_type in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"} and strong_directional_alignment:
            if breakout_confirmation in {"UP_CONFIRMED", "DOWN_CONFIRMED"} and trend_confidence >= 0.72:
                floor_rs *= 0.82
            if signal_conflict_score <= 20.0:
                floor_rs *= 0.95

        if volatility_regime == "HIGH":
            floor_rs *= 1.05

        min_floor = max(250.0, float(base_credit_floor_rs) * 0.70)
        max_floor = max(min_floor, float(base_credit_floor_rs) * 1.35)
        return max(min_floor, min(max_floor, floor_rs))

    def _candidate_signature(
        self,
        *,
        setup_type: str,
        market_bias: str,
        orb_confirmation: str,
        expiry: str,
        legs: List[Dict[str, Any]],
    ) -> str:
        payload = {
            "setup_type": str(setup_type or ""),
            "market_bias": str(market_bias or ""),
            "orb_confirmation": str(orb_confirmation or "NONE"),
            "expiry": str(expiry or ""),
        }
        return json.dumps(payload, sort_keys=True)

    def _uses_short_option_with_hedge(self) -> bool:
        return str(getattr(self.config, "directional_structure", "CREDIT_SPREAD") or "CREDIT_SPREAD").upper() == "SHORT_OPTION_WITH_HEDGE"

    def _directional_strategy_name(self, setup_type: str) -> str:
        text = str(setup_type or "").upper()
        if not self._uses_short_option_with_hedge():
            return text
        if text == "PUT_CREDIT_SPREAD":
            return "SHORT_PUT_WITH_HEDGE"
        if text == "CALL_CREDIT_SPREAD":
            return "SHORT_CALL_WITH_HEDGE"
        return text

    @staticmethod
    def _directional_bias_for_setup(setup_type: str) -> str:
        text = str(setup_type or "").upper()
        if text in {"PUT_CREDIT_SPREAD", "SHORT_PUT_WITH_HEDGE"}:
            return "BULLISH"
        if text in {"CALL_CREDIT_SPREAD", "SHORT_CALL_WITH_HEDGE"}:
            return "BEARISH"
        return "NEUTRAL"

    def _emergency_hedge_distance(self, volatility_regime: str, strike_step: float) -> float:
        regime = str(volatility_regime or "NORMAL").upper()
        if regime == "HIGH":
            width = float(self.config.emergency_hedge_distance_points_high_vol)
        elif regime == "LOW":
            width = float(self.config.emergency_hedge_distance_points_low_vol)
        else:
            width = float(self.config.emergency_hedge_distance_points_normal_vol)
        step = max(1.0, float(strike_step or 50.0))
        return max(step, round(width / step) * step)

    def _trend_day_fast_track_ready(
        self,
        *,
        setup_type: str,
        market_bias: str,
        trend_confidence: float,
        signal_conflict_score: float,
        strong_directional_alignment: bool,
        price_action_confirmation: str,
    ) -> bool:
        if not bool(self.config.trend_day_fast_track_enabled):
            return False
        if setup_type not in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"}:
            return False
        if self._directional_bias_for_setup(setup_type) != str(market_bias or "").upper():
            return False
        preferred_bias = str(getattr(self.config, "preferred_bias", "NEUTRAL") or "NEUTRAL").upper()
        if preferred_bias not in {"BULLISH", "BEARISH"}:
            return False
        current_bias = self._directional_bias_for_setup(setup_type)
        if current_bias != preferred_bias:
            return False
        if not strong_directional_alignment:
            return False
        if float(trend_confidence or 0.0) < float(self.config.trend_day_fast_track_min_trend_confidence):
            return False
        if float(signal_conflict_score or 0.0) > float(self.config.trend_day_fast_track_max_conflict):
            return False
        return str(price_action_confirmation or "NONE").upper() in {
            "CANDLE_CONFIRMED",
            "RETEST_CONFIRMED",
            "CANDLE_AND_RETEST_CONFIRMED",
            "REVERSAL_CONFIRMED",
            "BREAKOUT_CONFIRMED",
        }

    def _should_block_directional_flip(
        self,
        *,
        now: datetime,
        setup_type: str,
        market_bias: str,
        trend_confidence: float,
        signal_conflict_score: float,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        if not bool(self.config.directional_flip_block_enabled):
            return False, None
        current_bias = self._directional_bias_for_setup(setup_type)
        if current_bias not in {"BULLISH", "BEARISH"}:
            return False, None
        if current_bias != str(market_bias or "").upper():
            return False, None
        same_session = str(self.state.current_session_date or "") == now.date().isoformat()
        if not same_session:
            return False, None
        prev_strategy = str(self.state.strategy or "")
        prev_bias = self._directional_bias_for_setup(prev_strategy)
        prev_signature = str(self.state.candidate_setup_signature or "")
        prev_streak = int(self.state.candidate_signal_streak or 0)
        if prev_bias not in {"BULLISH", "BEARISH"} or prev_bias == current_bias:
            return False, None
        if not prev_signature or prev_streak <= 0:
            return False, None
        prev_trend_conf = float(self.state.trend_confidence or 0.0)
        prev_conflict = float(self.state.signal_conflict_score or 100.0)
        if prev_trend_conf < float(self.config.directional_flip_block_min_prev_trend_confidence):
            return False, None
        if prev_conflict > float(self.config.directional_flip_block_max_prev_conflict):
            return False, None
        preferred_bias = str(getattr(self.config, "preferred_bias", "NEUTRAL") or "NEUTRAL").upper()
        if preferred_bias in {"BULLISH", "BEARISH"} and current_bias == preferred_bias and prev_bias != preferred_bias:
            return False, None
        materially_stronger_flip = (
            float(trend_confidence or 0.0)
            >= prev_trend_conf + float(self.config.directional_flip_override_min_trend_delta)
            and float(signal_conflict_score or 0.0)
            <= max(
                0.0,
                prev_conflict - float(self.config.directional_flip_override_min_conflict_improvement),
            )
        )
        if materially_stronger_flip:
            return False, None
        return True, {
            "prev_bias": prev_bias,
            "prev_strategy": prev_strategy,
            "prev_signature": prev_signature,
            "prev_streak": prev_streak,
            "prev_trend_confidence": prev_trend_conf,
            "prev_conflict": prev_conflict,
        }

    def _time_ok(self, now: datetime) -> Tuple[bool, bool, str]:
        open_t = _parse_hhmm(self.config.market_open_time, dtime(9, 15))
        not_before_t = _parse_hhmm(self.config.entry_not_before, dtime(9, 45))
        last_entry_t = _parse_hhmm(self.config.last_new_entry_time, dtime(14, 20))
        now_t = now.timetz().replace(tzinfo=None) if getattr(now, "tzinfo", None) else now.time()
        if now.weekday() >= 5:
            return False, False, "WEEKEND"
        if now_t < open_t:
            return False, False, "PRE_MARKET"
        if now_t < not_before_t:
            return True, False, "WAIT_OPEN_SETTLE"
        if now_t > last_entry_t:
            return True, False, "LATE_ENTRY_CUTOFF"
        return True, True, "ENTRY_WINDOW_OPEN"

    def _base_result(
        self,
        *,
        now: datetime,
        signal: str,
        priority: str,
        reasons: List[str],
        market_bias: str,
        trend_confidence: float,
        signal_conflict_score: float,
        pcr_unbalanced: bool,
        pcr_unbalanced_side: str,
        volatility_regime: str,
        breakout_confirmation: str,
        price_action_bias: str = "NEUTRAL",
        price_action_confirmation: str = "NONE",
        retest_status: str = "NONE",
        has_open_bkm: bool,
        pcr_bias: str = "NEUTRAL",
        oi_build_bias: str = "UNKNOWN",
        recommendation: Optional[Dict[str, Any]] = None,
        expiry: Optional[str] = None,
        spot: Optional[float] = None,
        strategy: Optional[str] = None,
        candidate_setup_signature: Optional[str] = None,
        candidate_signal_streak: int = 0,
        entry_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "status": "ACTIVE",
            "mode": str(self.config.mode or "ADVISOR").upper(),
            "signal": signal,
            "priority": priority,
            "strategy": strategy,
            "expiry": expiry,
            "spot": None if spot is None else round(float(spot), 2),
            "market_bias": market_bias,
            "trend_confidence": round(float(trend_confidence or 0.0), 3),
            "signal_conflict_score": round(float(signal_conflict_score or 0.0), 1),
            "pcr_bias": str(pcr_bias or "NEUTRAL"),
            "oi_build_bias": str(oi_build_bias or "UNKNOWN"),
            "pcr_unbalanced": bool(pcr_unbalanced),
            "pcr_unbalanced_side": pcr_unbalanced_side,
            "volatility_regime": volatility_regime,
            "breakout_confirmation": breakout_confirmation,
            "price_action_bias": str(price_action_bias or "NEUTRAL"),
            "price_action_confirmation": str(price_action_confirmation or "NONE"),
            "retest_status": str(retest_status or "NONE"),
            "has_open_bkm": bool(has_open_bkm),
            "recommendation": recommendation or {},
            "reasons": list(reasons or []),
            "candidate_setup_signature": candidate_setup_signature,
            "candidate_signal_streak": max(0, int(candidate_signal_streak or 0)),
            "current_session_date": now.date().isoformat(),
            "entry_features": entry_features or {},
        }

    def _summarize_context(
        self,
        *,
        trend_ctx: Dict[str, Any],
        oc_ctx: Dict[str, Any],
        structure_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        trend_conf = float(structure_ctx.get("trend_confidence") or 0.0)
        conflict = float(structure_ctx.get("signal_conflict_score") or 0.0)
        dominant_bias = str(
            structure_ctx.get("dominant_signal_bias")
            or trend_ctx.get("bias")
            or oc_ctx.get("pcr_bias")
            or "NEUTRAL"
        ).upper()
        if dominant_bias not in {"BULLISH", "BEARISH", "NEUTRAL"}:
            dominant_bias = "NEUTRAL"
        preferred_bias = str(getattr(self.config, "preferred_bias", "NEUTRAL") or "NEUTRAL").upper()
        if preferred_bias not in {"BULLISH", "BEARISH"}:
            preferred_bias = "NEUTRAL"
        if dominant_bias == "NEUTRAL" and preferred_bias in {"BULLISH", "BEARISH"}:
            dominant_bias = preferred_bias
        pcr_unbalanced = bool(structure_ctx.get("pcr_unbalanced", False))
        pcr_unbalanced_side = str(structure_ctx.get("pcr_unbalanced_side") or "NEUTRAL").upper()
        pcr_bias = str(oc_ctx.get("pcr_bias") or "NEUTRAL").upper()
        oi_build_bias = str(((oc_ctx.get("oi_build") or {}) if isinstance(oc_ctx.get("oi_build"), dict) else {}).get("bias") or "UNKNOWN").upper()
        vol_regime = self._vol_regime(trend_ctx, structure_ctx)
        breakout_confirmation = str(trend_ctx.get("breakout_confirmation") or "NONE").upper()
        price_action_bias = str(structure_ctx.get("price_action_bias") or "NEUTRAL").upper()
        price_action_confirmation = str(structure_ctx.get("price_action_confirmation") or "NONE").upper()
        retest_status = str(structure_ctx.get("retest_status") or "NONE").upper()
        swing_structure_bias = str(structure_ctx.get("swing_structure_bias") or "NEUTRAL").upper()
        swing_structure_label = str(structure_ctx.get("swing_structure_label") or "RANGE").upper()
        swing_structure_confidence = float(_safe_float(structure_ctx.get("swing_structure_confidence")) or 0.0)
        return {
            "market_bias": dominant_bias,
            "trend_confidence": trend_conf,
            "signal_conflict_score": conflict,
            "pcr_bias": pcr_bias,
            "oi_build_bias": oi_build_bias,
            "pcr_unbalanced": pcr_unbalanced,
            "pcr_unbalanced_side": pcr_unbalanced_side,
            "volatility_regime": vol_regime,
            "breakout_confirmation": breakout_confirmation,
            "price_action_bias": price_action_bias,
            "price_action_confirmation": price_action_confirmation,
            "retest_status": retest_status,
            "swing_structure_bias": swing_structure_bias,
            "swing_structure_label": swing_structure_label,
            "swing_structure_confidence": swing_structure_confidence,
        }

    def _directional_chain_conflict(
        self,
        *,
        setup_type: str,
        oc_ctx: Dict[str, Any],
        pcr_unbalanced_side: str,
    ) -> Tuple[float, bool, List[str]]:
        if setup_type not in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"}:
            return 0.0, False, []
        expected_bias = "BULLISH" if setup_type == "PUT_CREDIT_SPREAD" else "BEARISH"
        opposite_bias = "BEARISH" if expected_bias == "BULLISH" else "BULLISH"
        pcr_bias = str(oc_ctx.get("pcr_bias") or "NEUTRAL").upper()
        pcr_total = _safe_float(oc_ctx.get("pcr_total"))
        pcr_near = _safe_float(oc_ctx.get("pcr_near_atm"))
        oi_build = oc_ctx.get("oi_build") if isinstance(oc_ctx.get("oi_build"), dict) else {}
        oi_build_bias = str(oi_build.get("bias") or "UNKNOWN").upper()
        penalty = 0.0
        details: List[str] = []

        if pcr_bias == opposite_bias:
            penalty += 10.0
            details.append(f"PCR bias is {pcr_bias}, which is against the setup.")
        if expected_bias == "BEARISH":
            if pcr_total is not None and pcr_total >= 1.20:
                penalty += 8.0
                details.append(f"PCR total is elevated at {pcr_total:.2f}.")
            if pcr_near is not None and pcr_near >= 1.18:
                penalty += 10.0
                details.append(f"Near-ATM PCR is elevated at {pcr_near:.2f}.")
        else:
            if pcr_total is not None and pcr_total <= 0.82:
                penalty += 8.0
                details.append(f"PCR total is weak at {pcr_total:.2f}.")
            if pcr_near is not None and pcr_near <= 0.84:
                penalty += 10.0
                details.append(f"Near-ATM PCR is weak at {pcr_near:.2f}.")
        if pcr_unbalanced_side == opposite_bias:
            penalty += 9.0
            details.append(f"PCR is unbalanced to the {pcr_unbalanced_side.lower()} side.")
        if opposite_bias in oi_build_bias:
            penalty += 9.0
            details.append(f"OI build is showing {oi_build_bias.lower()}, against the trade.")

        severe_conflict = penalty >= 24.0
        return penalty, severe_conflict, details

    def _directional_price_action_expectations(
        self,
        *,
        setup_type: str,
        structure_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        if setup_type not in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"}:
            return {
                "price_action_ok": True,
                "candle_confirmation_ok": False,
                "retest_failed_against_setup": False,
                "hard_block_pattern_against_setup": False,
                "hard_block_patterns": [],
                "details": [],
            }
        expected_bias = "BULLISH" if setup_type == "PUT_CREDIT_SPREAD" else "BEARISH"
        price_action_bias = str(structure_ctx.get("price_action_bias") or "NEUTRAL").upper()
        price_action_confirmation = str(structure_ctx.get("price_action_confirmation") or "NONE").upper()
        retest_status = str(structure_ctx.get("retest_status") or "NONE").upper()
        retest_bias = str(structure_ctx.get("retest_bias") or "NEUTRAL").upper()
        candle_patterns = structure_ctx.get("price_action_patterns")
        candle_patterns = candle_patterns if isinstance(candle_patterns, list) else []
        normalized_patterns = {
            str(item or "").upper().split(":")[-1]
            for item in candle_patterns
            if str(item or "").strip()
        }
        supportive_confirmations = {
            "CANDLE_CONFIRMED",
            "RETEST_CONFIRMED",
            "CANDLE_AND_RETEST_CONFIRMED",
            "REVERSAL_CONFIRMED",
            "BREAKOUT_CONFIRMED",
        }
        price_action_ok = (
            price_action_bias == expected_bias
            and price_action_confirmation in supportive_confirmations
        )
        retest_ok = (
            retest_bias == expected_bias
            and retest_status
            in {
                "SUPPORT_HOLD",
                "RESISTANCE_HOLD",
                "BREAKOUT_RETEST_HOLD",
                "BREAKDOWN_RETEST_HOLD",
                "DOUBLE_BOTTOM_NECKLINE_BREAK",
                "DOUBLE_TOP_NECKLINE_BREAK",
            }
        )
        retest_failed_against_setup = (
            retest_status in {"RETEST_FAILED", "FAILED_BREAKOUT_RETEST", "FAILED_BREAKDOWN_RETEST"}
            and retest_bias not in {"NEUTRAL", expected_bias}
        )
        if expected_bias == "BULLISH":
            hard_block_patterns = sorted(
                normalized_patterns.intersection({"DOUBLE_TOP_M_CONFIRMED", "FAKE_BREAKOUT_UP"})
            )
        else:
            hard_block_patterns = sorted(
                normalized_patterns.intersection({"DOUBLE_BOTTOM_W_CONFIRMED", "FAKE_BREAKDOWN_DOWN"})
            )
        details: List[str] = []
        if candle_patterns:
            details.append("Price action patterns: " + ", ".join(str(item) for item in candle_patterns[:4]))
        details.append(f"Price action bias is {price_action_bias}.")
        details.append(f"Retest status is {retest_status}.")
        if hard_block_patterns:
            details.append("Pattern is reversing against the setup: " + ", ".join(hard_block_patterns) + ".")
        return {
            "price_action_ok": bool(price_action_ok or retest_ok),
            "candle_confirmation_ok": bool(price_action_ok),
            "retest_failed_against_setup": bool(retest_failed_against_setup),
            "hard_block_pattern_against_setup": bool(hard_block_patterns),
            "hard_block_patterns": hard_block_patterns,
            "details": details,
        }

    @staticmethod
    def _supportive_retest_for_setup(setup_type: str, retest_status: str) -> bool:
        status = str(retest_status or "NONE").upper()
        setup = str(setup_type or "").upper()
        if setup == "CALL_CREDIT_SPREAD":
            return status in {"RESISTANCE_HOLD", "BREAKDOWN_RETEST_HOLD"}
        if setup == "PUT_CREDIT_SPREAD":
            return status in {"SUPPORT_HOLD", "BREAKOUT_RETEST_HOLD"}
        return False

    @staticmethod
    def _swing_structure_supports_setup(
        *,
        setup_type: str,
        structure_ctx: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        if setup_type not in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"}:
            return True, []
        expected_bias = "BULLISH" if setup_type == "PUT_CREDIT_SPREAD" else "BEARISH"
        opposite_bias = "BEARISH" if expected_bias == "BULLISH" else "BULLISH"
        swing_bias = str(structure_ctx.get("swing_structure_bias") or "NEUTRAL").upper()
        swing_label = str(structure_ctx.get("swing_structure_label") or "RANGE").upper()
        swing_conf = float(_safe_float(structure_ctx.get("swing_structure_confidence")) or 0.0)
        details = [
            f"Swing structure bias is {swing_bias}.",
            f"Swing structure label is {swing_label}.",
        ]
        if swing_bias == opposite_bias and swing_conf >= 0.55:
            details.append("The current swing structure is materially against the setup.")
            return False, details
        return True, details

    def _entry_feature_snapshot(
        self,
        *,
        now: datetime,
        expiry: str,
        spot: float,
        strategy: Optional[str],
        market_bias: str,
        trend_confidence: float,
        signal_conflict_score: float,
        volatility_regime: str,
        breakout_confirmation: str,
        ema_alignment_5m: str,
        ema_alignment_15m: str,
        orb_confirmation: str,
        structure_ctx: Dict[str, Any],
        oc_ctx: Dict[str, Any],
        recommendation: Dict[str, Any],
        trade_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        intraday_support = structure_ctx.get("intraday_support") if isinstance(structure_ctx.get("intraday_support"), dict) else {}
        intraday_resistance = structure_ctx.get("intraday_resistance") if isinstance(structure_ctx.get("intraday_resistance"), dict) else {}
        swing_support = structure_ctx.get("swing_support") if isinstance(structure_ctx.get("swing_support"), dict) else {}
        swing_resistance = structure_ctx.get("swing_resistance") if isinstance(structure_ctx.get("swing_resistance"), dict) else {}
        daily_support = structure_ctx.get("daily_support") if isinstance(structure_ctx.get("daily_support"), dict) else {}
        daily_resistance = structure_ctx.get("daily_resistance") if isinstance(structure_ctx.get("daily_resistance"), dict) else {}
        return {
            "captured_at": now.isoformat(timespec="seconds"),
            "expiry": str(expiry or ""),
            "spot": round(float(spot or 0.0), 2),
            "strategy_type": str(strategy or ""),
            "market_bias": str(market_bias or "NEUTRAL"),
            "trend_confidence": round(float(trend_confidence or 0.0), 3),
            "signal_conflict_score": round(float(signal_conflict_score or 0.0), 1),
            "volatility_regime": str(volatility_regime or "UNKNOWN"),
            "breakout_confirmation": str(breakout_confirmation or "NONE"),
            "ema_alignment_5m": str(ema_alignment_5m or "NEUTRAL"),
            "ema_alignment_15m": str(ema_alignment_15m or "NEUTRAL"),
            "orb_confirmation": str(orb_confirmation or "NONE"),
            "price_action_bias": str(structure_ctx.get("price_action_bias") or "NEUTRAL"),
            "price_action_confirmation": str(structure_ctx.get("price_action_confirmation") or "NONE"),
            "primary_pattern": str(structure_ctx.get("primary_pattern") or "NONE"),
            "price_action_patterns": list(structure_ctx.get("price_action_patterns") or []),
            "retest_status": str(structure_ctx.get("retest_status") or "NONE"),
            "retest_bias": str(structure_ctx.get("retest_bias") or "NEUTRAL"),
            "pcr_total": _safe_float(oc_ctx.get("pcr_total")),
            "pcr_near_atm": _safe_float(oc_ctx.get("pcr_near_atm")),
            "pcr_bias": str(oc_ctx.get("pcr_bias") or "NEUTRAL"),
            "oi_build_bias": str(((oc_ctx.get("oi_build") or {}) if isinstance(oc_ctx.get("oi_build"), dict) else {}).get("bias") or "UNKNOWN"),
            "intraday_support": _safe_float(intraday_support.get("level")),
            "intraday_resistance": _safe_float(intraday_resistance.get("level")),
            "swing_support": _safe_float(swing_support.get("level")),
            "swing_resistance": _safe_float(swing_resistance.get("level")),
            "daily_support": _safe_float(daily_support.get("level")),
            "daily_resistance": _safe_float(daily_resistance.get("level")),
            "weekly_support": _safe_float(structure_ctx.get("weekly_support")),
            "weekly_resistance": _safe_float(structure_ctx.get("weekly_resistance")),
            "swing_structure_bias": str(structure_ctx.get("swing_structure_bias") or "NEUTRAL"),
            "swing_structure_label": str(structure_ctx.get("swing_structure_label") or "RANGE"),
            "swing_structure_confidence": float(_safe_float(structure_ctx.get("swing_structure_confidence")) or 0.0),
            "sr_alignment_hits": int(structure_ctx.get("sr_alignment_hits") or 0),
            "legs": list(recommendation.get("legs") or []),
            "trade_plan_posture": str((trade_plan or {}).get("posture") or "UNKNOWN"),
            "trade_plan_setup_type": str((trade_plan or {}).get("setup_type") or ""),
            "trade_plan_pullback_zone_low": _safe_float((trade_plan or {}).get("pullback_zone_low")),
            "trade_plan_pullback_zone_high": _safe_float((trade_plan or {}).get("pullback_zone_high")),
            "trade_plan_invalidation_spot": _safe_float((trade_plan or {}).get("invalidation_spot")),
            "unlimited_profit_trailing": bool((recommendation.get("trail") or {}).get("unlimited_profit_mode")),
            "profit_trail_arm_rs": _safe_float((recommendation.get("trail") or {}).get("arm_profit_rs_per_set")),
            "trail_floor_min_profit_rs": _safe_float((recommendation.get("trail") or {}).get("min_locked_profit_rs_per_set")),
            "trail_keep_pct": _safe_float((recommendation.get("trail") or {}).get("keep_pct")),
        }

    def evaluate(
        self,
        *,
        now: datetime,
        expiry: str,
        spot: float,
        chain_rows: List[Dict[str, Any]],
        context: Dict[str, Any],
        has_open_bkm: bool,
    ) -> Dict[str, Any]:
        trend_ctx = context.get("trend") if isinstance(context.get("trend"), dict) else {}
        oc_ctx = context.get("option_chain") if isinstance(context.get("option_chain"), dict) else {}
        structure_ctx = context.get("structure") if isinstance(context.get("structure"), dict) else {}
        ctx_summary = self._summarize_context(trend_ctx=trend_ctx, oc_ctx=oc_ctx, structure_ctx=structure_ctx)
        market_bias = str(ctx_summary["market_bias"])
        trend_confidence = float(ctx_summary["trend_confidence"])
        signal_conflict_score = float(ctx_summary["signal_conflict_score"])
        pcr_bias = str(ctx_summary.get("pcr_bias") or "NEUTRAL")
        oi_build_bias = str(ctx_summary.get("oi_build_bias") or "UNKNOWN")
        pcr_unbalanced = bool(ctx_summary["pcr_unbalanced"])
        pcr_unbalanced_side = str(ctx_summary["pcr_unbalanced_side"])
        volatility_regime = str(ctx_summary["volatility_regime"])
        breakout_confirmation = str(ctx_summary["breakout_confirmation"])
        price_action_bias = str(ctx_summary.get("price_action_bias") or "NEUTRAL")
        price_action_confirmation = str(ctx_summary.get("price_action_confirmation") or "NONE")
        retest_status = str(ctx_summary.get("retest_status") or "NONE")
        swing_structure_bias = str(ctx_summary.get("swing_structure_bias") or "NEUTRAL")
        swing_structure_label = str(ctx_summary.get("swing_structure_label") or "RANGE")
        tf_map = trend_ctx.get("timeframes") if isinstance(trend_ctx.get("timeframes"), dict) else {}
        tf5 = tf_map.get("5") if isinstance(tf_map.get("5"), dict) else {}
        tf15 = tf_map.get("15") if isinstance(tf_map.get("15"), dict) else {}
        ema_alignment_5m = str(tf5.get("ema_alignment") or "NEUTRAL").upper()
        ema_alignment_15m = str(tf15.get("ema_alignment") or "NEUTRAL").upper()
        orb_ctx = trend_ctx.get("orb") if isinstance(trend_ctx.get("orb"), dict) else {}
        orb_confirmation = str(orb_ctx.get("breakout_confirmation") or "NONE").upper()

        market_open, entry_window_open, time_reason = self._time_ok(now)
        reasons: List[str] = []
        if not market_open:
            reasons.append(time_reason)
            return self._base_result(
                now=now,
                signal="NO_TRADE",
                priority="INFO",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                recommendation={
                    "headline": "No trade now.",
                    "signal_text": "Market is closed.",
                    "what_to_enter": "No intraday options-selling setup while market is closed.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Wait for market hours.",
                    "why": ["Market is closed, so the advisor is parked."],
                },
            )

        if has_open_bkm and not bool(self.config.allow_parallel_with_bkm):
            reasons.append("OPEN_BKM_POSITION_PRESENT")
            return self._base_result(
                now=now,
                signal="NO_TRADE",
                priority="WARN",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                recommendation={
                    "headline": "No new intraday trade.",
                    "signal_text": "Avoid overtrading while your Batman position is already open.",
                    "what_to_enter": "No intraday option-selling entry now.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Focus on the existing Batman trade unless you intentionally allow parallel trades.",
                    "why": ["Batman BKM trade is already open. This advisor is blocking new intraday setups to reduce overtrading."],
                },
            )

        if signal_conflict_score >= float(self.config.no_trade_conflict_threshold):
            reasons.append("SIGNAL_CONFLICT_HIGH")
            return self._base_result(
                now=now,
                signal="NO_TRADE",
                priority="WARN",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                recommendation={
                    "headline": "No trade: signals are conflicting.",
                    "signal_text": "Wait. Market direction is not clean enough for option selling.",
                    "what_to_enter": "No new trade now.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Wait for clearer alignment in trend and option-chain signals.",
                    "why": [
                        f"Signal conflict score is {int(round(signal_conflict_score))}% (too high).",
                        "Taking a trade now increases the chance of emotional exits and overtrading.",
                    ],
                },
            )

        if not entry_window_open:
            reasons.append(time_reason)
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="INFO",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                recommendation={
                    "headline": "Wait for the right time window.",
                    "signal_text": "Do not enter yet.",
                    "what_to_enter": "No trade now.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Wait until the intraday entry window opens.",
                    "why": ["The advisor is avoiding early/late entries to reduce overtrading and fear-based exits."],
                },
            )

        plan = build_trade_plan_from_runtime_context(
            spot=spot,
            trend_ctx=trend_ctx,
            oc_ctx=oc_ctx,
            structure_ctx=structure_ctx,
            market_bias=market_bias,
            bias_confidence=trend_confidence,
        )

        def _planner_wait_result(
            *,
            reason_code: str,
            headline: str,
            signal_text: str,
            hold_text: str,
            why_extra: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            plan_reasons = list(plan.get("reason_summary") or [])
            why = list(why_extra or [])
            why.extend(plan_reasons[:4])
            if not why:
                why.append("The shared chart plan is not ready for a disciplined entry yet.")
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="INFO",
                reasons=[reason_code],
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_bias=pcr_bias,
                oi_build_bias=oi_build_bias,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                price_action_bias=price_action_bias,
                price_action_confirmation=price_action_confirmation,
                retest_status=retest_status,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                strategy=str(plan.get("setup_type") or "") or None,
                recommendation={
                    "headline": headline,
                    "signal_text": signal_text,
                    "what_to_enter": str(plan.get("strike_hint") or "No trade now."),
                    "sl_text": "Not applicable yet",
                    "tp_text": "Not applicable yet",
                    "hold_text": hold_text,
                    "why": why,
                    "plan": plan,
                },
            )

        chain_rows = [r for r in (chain_rows or []) if isinstance(r, dict)]
        strikes = sorted(
            {
                float(r.get("strike") or 0.0)
                for r in chain_rows
                if str(r.get("option_type") or "").upper() in {"CE", "PE"} and _safe_float(r.get("strike")) is not None
            }
        )
        if not strikes:
            reasons.append("NO_OPTION_CHAIN_STRIKES")
            return self._base_result(
                now=now,
                signal="NO_TRADE",
                priority="WARN",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                recommendation={
                    "headline": "No trade: option chain unavailable.",
                    "signal_text": "Wait for stable option-chain data.",
                    "what_to_enter": "No trade now.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Wait for fresh quotes.",
                    "why": ["The advisor could not read enough option-chain strikes to build a safe setup."],
                },
            )

        strike_step = self._strike_step(strikes)
        lookup = self._row_lookup(chain_rows)
        step = max(1.0, strike_step)
        vol_width = self._spread_width(volatility_regime, step)
        safety_buf = max(float(self.config.sr_safety_buffer_points), step * 2.0)
        fallback_buf = max(float(self.config.fallback_otm_buffer_points), step * 3.0)
        min_sr_dist = max(float(self.config.min_sr_distance_points), step * 3.0)

        intraday_support = structure_ctx.get("intraday_support") if isinstance(structure_ctx.get("intraday_support"), dict) else {}
        intraday_resistance = structure_ctx.get("intraday_resistance") if isinstance(structure_ctx.get("intraday_resistance"), dict) else {}
        support_level = _safe_float(intraday_support.get("level"))
        resistance_level = _safe_float(intraday_resistance.get("level"))
        put_wall_below = oc_ctx.get("put_wall_below") if isinstance(oc_ctx.get("put_wall_below"), dict) else {}
        call_wall_above = oc_ctx.get("call_wall_above") if isinstance(oc_ctx.get("call_wall_above"), dict) else {}
        put_wall_strike = _safe_float(put_wall_below.get("strike"))
        call_wall_strike = _safe_float(call_wall_above.get("strike"))

        tf15 = tf_map.get("15") if isinstance(tf_map.get("15"), dict) else {}
        atr15 = _safe_float(tf15.get("atr_like_points")) or 0.0
        dynamic_pad = max(safety_buf, round((atr15 * 0.5) / step) * step if atr15 > 0 else 0.0)
        if dynamic_pad <= 0:
            dynamic_pad = safety_buf
        directional_min_conf, directional_max_conflict, strong_directional_alignment = self._directional_thresholds(
            market_bias=market_bias,
            breakout_confirmation=breakout_confirmation,
            trend_confidence=trend_confidence,
            signal_conflict_score=signal_conflict_score,
            tf_map=tf_map,
            oc_ctx=oc_ctx,
        )

        setup_type: Optional[str] = None
        setup_reasons: List[str] = []
        planned_setup_type = str(plan.get("setup_type") or "").upper()
        planned_setup_family = str(plan.get("setup_family") or "PULLBACK").upper()
        planner_confirmation_score = float(_safe_float(plan.get("confirmation_score")) or 0.0)
        planner_structure_conflict = float(_safe_float(plan.get("structure_conflict_score")) or 0.0)
        planner_option_chain_conflict = float(_safe_float(plan.get("option_chain_conflict_score")) or 0.0)
        planner_reversal_conflict = float(_safe_float(plan.get("reversal_conflict_score")) or 0.0)
        planner_total_conflict = float(_safe_float(plan.get("total_conflict_score")) or 0.0) * 100.0
        family_min_conf = planner_relaxed_min_conf = max(0.42, directional_min_conf - 0.20)
        family_max_conflict = planner_relaxed_max_conflict = min(58.0, directional_max_conflict + 5.0)
        if planned_setup_family == "CONTINUATION":
            family_min_conf = max(0.38, directional_min_conf - 0.12)
            family_max_conflict = min(64.0, planner_relaxed_max_conflict + 4.0)
        elif planned_setup_family == "REVERSAL":
            family_min_conf = max(0.48, directional_min_conf - 0.06)
            family_max_conflict = min(52.0, planner_relaxed_max_conflict)
        planner_confirmed_directional = bool(
            planned_setup_type in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"}
            and bool(plan.get("entry_ready"))
            and bool(plan.get("confirmation_ready"))
            and planner_confirmation_score >= 0.66
            and (
                bool(plan.get("higher_tf_ok"))
                or float(_safe_float(plan.get("higher_tf_score")) or 0.0) >= 0.02
            )
            and bool(plan.get("buyer_seller_ok"))
            and bool(plan.get("divergence_supportive"))
            and not bool(plan.get("fake_breakout_conflict"))
            and planner_total_conflict <= family_max_conflict
        )
        planner_weighted_htf_score = float(_safe_float(plan.get("higher_tf_score")) or 0.0)
        if (
            planned_setup_type in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"}
            and not (
                planned_setup_type == "PUT_CREDIT_SPREAD" and breakout_confirmation == "DOWN_CONFIRMED" and planned_setup_family != "REVERSAL"
            )
            and not (
                planned_setup_type == "CALL_CREDIT_SPREAD" and breakout_confirmation == "UP_CONFIRMED" and planned_setup_family != "REVERSAL"
            )
            and (
                (
                    trend_confidence >= family_min_conf
                    and max(signal_conflict_score, planner_total_conflict) <= family_max_conflict
                )
                or (
                    planner_confirmed_directional
                    and trend_confidence >= family_min_conf
                    and max(signal_conflict_score, planner_total_conflict) <= family_max_conflict
                    and planner_weighted_htf_score >= 0.02
                )
            )
        ):
            setup_type = planned_setup_type
            setup_reasons.append(
                f"Shared chart plan and market context agree on a {planned_setup_family.lower()} defined-risk spread."
            )
            if strong_directional_alignment:
                setup_reasons.append(
                    "Multi-timeframe trend is aligned, so the directional spread threshold is slightly relaxed."
                )
            elif planner_confirmed_directional and trend_confidence < directional_min_conf:
                setup_reasons.append(
                    "Planner-confirmed confluence is strong enough to allow a slightly lower trend-confidence threshold."
                )
            if planner_total_conflict > 0:
                setup_reasons.append(
                    "Conflict split: structure="
                    f"{planner_structure_conflict:.2f}, chain={planner_option_chain_conflict:.2f}, reversal={planner_reversal_conflict:.2f}."
                )
        else:
            sr_width = None
            if support_level is not None and resistance_level is not None and resistance_level > support_level:
                sr_width = resistance_level - support_level
            range_ok = (
                trend_confidence <= float(self.config.range_max_trend_confidence)
                and signal_conflict_score <= float(self.config.range_max_conflict)
                and breakout_confirmation == "NONE"
                and sr_width is not None
                and sr_width >= max(2 * min_sr_dist, 4 * step)
            )
            if range_ok and planned_setup_type not in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"}:
                setup_type = "IRON_CONDOR"
                setup_reasons.append("Range conditions look acceptable for a defined-risk option selling setup.")

        if not setup_type:
            reasons.append("NO_CLEAR_SETUP")
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="INFO",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                recommendation={
                    "headline": "Wait: no clean setup yet.",
                    "signal_text": "Do not enter now.",
                    "what_to_enter": str(plan.get("strike_hint") or "No trade recommendation at this moment."),
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": str(plan.get("trigger_window") or "Wait for clearer trend/range structure and lower signal conflict."),
                    "why": [
                        f"Bias={market_bias}, trend confidence={int(round(trend_confidence*100))}%, conflict={int(round(signal_conflict_score))}%.",
                        "Setup quality is not high enough for a disciplined intraday short option entry.",
                    ]
                    + list(plan.get("reason_summary") or [])[:3],
                    "plan": plan,
                },
            )

        flip_blocked, flip_context = self._should_block_directional_flip(
            now=now,
            setup_type=setup_type,
            market_bias=market_bias,
            trend_confidence=trend_confidence,
            signal_conflict_score=signal_conflict_score,
        )
        if flip_blocked and flip_context is not None:
            reasons.append("COUNTERTREND_FLIP_BLOCKED")
            prev_bias = str(flip_context.get("prev_bias") or "NEUTRAL")
            prev_streak = int(flip_context.get("prev_streak") or 0)
            prev_signature = str(flip_context.get("prev_signature") or "")
            prev_trend_conf = float(flip_context.get("prev_trend_confidence") or 0.0)
            prev_conflict = float(flip_context.get("prev_conflict") or 0.0)
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="INFO",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                price_action_bias=price_action_bias,
                price_action_confirmation=price_action_confirmation,
                retest_status=retest_status,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                strategy=setup_type,
                recommendation={
                    "headline": "Wait: one opposite bar is not enough to reverse the plan.",
                    "signal_text": "Do not flip direction yet.",
                    "what_to_enter": "No new trade now.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Wait for the opposite setup to prove itself with materially stronger structure.",
                    "why": [
                        f"Previous {prev_bias.lower()} candidate was cleaner (trend {int(round(prev_trend_conf * 100))}%, conflict {int(round(prev_conflict))}%).",
                        f"Current opposite setup is not strong enough yet to cancel that plan.",
                    ],
                },
                candidate_setup_signature=prev_signature,
                candidate_signal_streak=prev_streak,
            )

        if setup_type in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"}:
            continuation_ready = bool(plan.get("continuation_ready", False))
            reversal_ready = bool(plan.get("reversal_ready", False))
            pullback_ok = bool(plan.get("pullback_ok", plan.get("pullback_touched"))) or continuation_ready
            pullback_score = float(_safe_float(plan.get("pullback_score")) or 0.0)
            pullback_tolerance = _safe_float(plan.get("pullback_tolerance_points"))
            higher_tf_score = float(_safe_float(plan.get("higher_tf_score")) or 0.0)
            confirmation_score = float(_safe_float(plan.get("confirmation_score")) or 0.0)
            strong_candle_ok = bool(plan.get("strong_candle_ok"))
            strong_candle_text = str(plan.get("strong_candle_text") or "No strong 5m confirmation candle yet.")
            ema2050_ok = bool(plan.get("ema_2050_ok"))
            ema2050_5m = str(plan.get("ema_2050_alignment_5m") or "NEUTRAL")
            ema2050_15m = str(plan.get("ema_2050_alignment_15m") or "NEUTRAL")
            if not pullback_ok:
                return _planner_wait_result(
                    reason_code="PULLBACK_ZONE_NOT_REACHED",
                    headline="Wait: price is still outside the ATR-tolerant pullback zone.",
                    signal_text="Do not sell premium before price gets close enough to the planned zone.",
                    hold_text=str(plan.get("trigger_window") or "Wait for the planned pullback zone."),
                    why_extra=[
                        (
                            f"Pullback quality is only {int(round(pullback_score * 100))}%"
                            + (
                                f" within an ATR-tolerant buffer of {float(pullback_tolerance):.0f} points."
                                if pullback_tolerance is not None
                                else "."
                            )
                        ),
                    ],
                )
            if continuation_ready:
                reasons.append("TREND_CONTINUATION_READY")
            if reversal_ready:
                reasons.append("REVERSAL_SETUP_READY")
            if not continuation_ready and not ema2050_ok:
                return _planner_wait_result(
                    reason_code="EMA2050_CONFIRMATION_MISSING",
                    headline="Wait: 20/50 EMA trend is not aligned.",
                    signal_text="Do not enter until the 20/50 EMA trend agrees with the setup.",
                    hold_text="Wait for 5m EMA20/50 alignment and non-opposing 15m EMA20/50 trend.",
                    why_extra=[
                        f"5m EMA20/50 alignment is {ema2050_5m}.",
                        f"15m EMA20/50 alignment is {ema2050_15m}.",
                    ],
                )
            if not continuation_ready and not strong_candle_ok:
                return _planner_wait_result(
                    reason_code="STRONG_5M_CANDLE_MISSING",
                    headline="Wait: strong 5-minute confirmation candle is missing.",
                    signal_text="Do not enter until a strong 5m candle confirms the pullback.",
                    hold_text="Wait for a decisive red/green 5m candle in the trade direction after the pullback.",
                    why_extra=[strong_candle_text],
                )
            if reversal_ready and not bool(plan.get("divergence_supportive")):
                return _planner_wait_result(
                    reason_code="RSI_DIVERGENCE_AGAINST_SETUP",
                    headline="Wait: reversal setup is fighting RSI divergence.",
                    signal_text="Do not enter the reversal while divergence is still against it.",
                    hold_text="Wait for RSI divergence to stop fighting the reversal setup.",
                    why_extra=["Reversal entries need supportive divergence or a neutral RSI read."],
                )
            if bool(plan.get("confirmation_ready")) and not bool(plan.get("buyer_seller_ok")):
                reason_code = (
                    "SELLER_ZONE_UNCONFIRMED" if setup_type == "CALL_CREDIT_SPREAD" else "BUYER_ZONE_UNCONFIRMED"
                )
                return _planner_wait_result(
                    reason_code=reason_code,
                    headline="Wait: participation is not visible at the decision zone.",
                    signal_text="Do not enter until buyers/sellers show up at the planned level.",
                    hold_text="Wait for clear buyer/seller defense at the pullback zone.",
                    why_extra=[
                        "The chart plan requires visible participation at support/resistance before entry.",
                    ],
                )
            if bool(plan.get("confirmation_ready")) and not bool(plan.get("higher_tf_ok")) and confirmation_score < 0.67:
                return _planner_wait_result(
                    reason_code="HTF_CONFLUENCE_MISSING",
                    headline="Wait: higher timeframe structure is not aligned enough.",
                    signal_text="Do not enter until 15m/60m structure agrees with the trade.",
                    hold_text="Wait for stronger multi-timeframe confluence.",
                    why_extra=[
                        f"Current weighted higher-timeframe score is {higher_tf_score:+.2f}; the setup needs stronger 15m/60m alignment.",
                    ],
                )
            if planner_total_conflict > family_max_conflict:
                return _planner_wait_result(
                    reason_code="SIGNAL_CONFLICT_HIGH",
                    headline="Wait: split conflicts are still too high.",
                    signal_text="Do not enter while structure/chain/reversal conflicts are elevated.",
                    hold_text="Wait for cleaner alignment between chart structure, option chain, and reversal context.",
                    why_extra=[
                        f"Structure conflict={planner_structure_conflict:.2f}, option-chain conflict={planner_option_chain_conflict:.2f}, reversal conflict={planner_reversal_conflict:.2f}.",
                    ],
                )
            if planned_setup_family == "REVERSAL":
                against_breakout = (
                    setup_type == "PUT_CREDIT_SPREAD" and breakout_confirmation == "DOWN_CONFIRMED"
                ) or (
                    setup_type == "CALL_CREDIT_SPREAD" and breakout_confirmation == "UP_CONFIRMED"
                )
                if against_breakout and not reversal_ready:
                    return _planner_wait_result(
                        reason_code="REVERSAL_NOT_CONFIRMED",
                        headline="Wait: reversal setup is still fighting the active breakout.",
                        signal_text="Do not fade the breakout until the reversal confirmation is stronger.",
                        hold_text="Wait for M/W confirmation, failed breakout retest, and supportive divergence together.",
                        why_extra=["Countertrend reversals need stronger confirmation than continuation trades."],
                    )

        now_t = now.timetz().replace(tzinfo=None) if getattr(now, "tzinfo", None) else now.time()
        high_vol_bear_call_not_before = _parse_hhmm(self.config.high_vol_bear_call_not_before, dtime(10, 15))
        if (
            setup_type == "CALL_CREDIT_SPREAD"
            and volatility_regime == "HIGH"
            and now_t < high_vol_bear_call_not_before
        ):
            reasons.append("HIGH_VOL_BEAR_CALL_DELAY")
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="INFO",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                price_action_bias=price_action_bias,
                price_action_confirmation=price_action_confirmation,
                retest_status=retest_status,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                strategy=setup_type,
                recommendation={
                    "headline": "Wait: high-volatility bear call setup is too early.",
                    "signal_text": "Do not enter yet.",
                    "what_to_enter": "No trade now.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Let the first volatility burst settle before entering a bear call spread.",
                    "why": [
                        f"Current time is {now_t.strftime('%H:%M')}.",
                        f"In high volatility, bearish call spreads are delayed until after {high_vol_bear_call_not_before.strftime('%H:%M')}.",
                    ],
                },
            )

        chain_conflict_penalty, severe_chain_conflict, chain_conflict_details = self._directional_chain_conflict(
            setup_type=setup_type,
            oc_ctx=oc_ctx,
            pcr_unbalanced_side=pcr_unbalanced_side,
        )
        if chain_conflict_penalty > 0:
            signal_conflict_score = min(100.0, float(signal_conflict_score) + float(chain_conflict_penalty))
            setup_reasons.extend(chain_conflict_details[:2])
        if severe_chain_conflict and setup_type in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"}:
            reasons.append("OPTION_CHAIN_AGAINST_SETUP")
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="INFO",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_bias=pcr_bias,
                oi_build_bias=oi_build_bias,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                price_action_bias=price_action_bias,
                price_action_confirmation=price_action_confirmation,
                retest_status=retest_status,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                strategy=setup_type,
                recommendation={
                    "headline": "Wait: option-chain context is against the setup.",
                    "signal_text": "Do not enter yet.",
                    "what_to_enter": "No trade now.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Wait for PCR and OI build-up to stop fighting the trade direction.",
                    "why": chain_conflict_details
                    + ["The setup looks directional, but the option chain is not supporting it cleanly enough."],
                },
            )

        expected_ema_alignment = None
        expected_orb_confirmation = None
        if setup_type == "PUT_CREDIT_SPREAD":
            expected_ema_alignment = "BULLISH"
            expected_orb_confirmation = "UP_CONFIRMED"
        elif setup_type == "CALL_CREDIT_SPREAD":
            expected_ema_alignment = "BEARISH"
            expected_orb_confirmation = "DOWN_CONFIRMED"

        if expected_ema_alignment and bool(self.config.require_ema_alignment):
            ema_15_ok = ema_alignment_15m == expected_ema_alignment
            ema_5_not_against = ema_alignment_5m in {expected_ema_alignment, "NEUTRAL"}
            if not (ema_15_ok and ema_5_not_against):
                reasons.append("EMA_ALIGNMENT_MISSING")
                return self._base_result(
                    now=now,
                    signal="WAIT",
                    priority="INFO",
                    reasons=reasons,
                    market_bias=market_bias,
                    trend_confidence=trend_confidence,
                    signal_conflict_score=signal_conflict_score,
                    pcr_unbalanced=pcr_unbalanced,
                    pcr_unbalanced_side=pcr_unbalanced_side,
                    volatility_regime=volatility_regime,
                    breakout_confirmation=breakout_confirmation,
                    has_open_bkm=has_open_bkm,
                    expiry=expiry,
                    spot=spot,
                    strategy=setup_type,
                    recommendation={
                        "headline": "Wait: EMA trend is not aligned.",
                        "signal_text": "Do not enter yet.",
                        "what_to_enter": "No trade now.",
                        "sl_text": "Not applicable",
                        "tp_text": "Not applicable",
                        "hold_text": "Wait until 15m EMA(5/20) aligns with the trade and 5m EMA is not fighting it.",
                        "why": [
                            f"5m EMA alignment is {ema_alignment_5m}.",
                            f"15m EMA alignment is {ema_alignment_15m}.",
                            "This filter is blocking weak early entries and reducing drawdown.",
                        ],
                    },
                )

        if expected_orb_confirmation and bool(self.config.require_orb_confirmation):
            if orb_confirmation != expected_orb_confirmation:
                reasons.append("ORB_CONFIRMATION_MISSING")
                return self._base_result(
                    now=now,
                    signal="WAIT",
                    priority="INFO",
                    reasons=reasons,
                    market_bias=market_bias,
                    trend_confidence=trend_confidence,
                    signal_conflict_score=signal_conflict_score,
                    pcr_unbalanced=pcr_unbalanced,
                    pcr_unbalanced_side=pcr_unbalanced_side,
                    volatility_regime=volatility_regime,
                    breakout_confirmation=breakout_confirmation,
                    has_open_bkm=has_open_bkm,
                    expiry=expiry,
                    spot=spot,
                    strategy=setup_type,
                    recommendation={
                        "headline": "Wait: 15-minute ORB is not confirmed.",
                        "signal_text": "Do not enter yet.",
                        "what_to_enter": "No trade now.",
                        "sl_text": "Not applicable",
                        "tp_text": "Not applicable",
                        "hold_text": "Wait for a clean 15-minute ORB breakout in the same direction as the setup.",
                        "why": [
                            f"Current ORB confirmation is {orb_confirmation}.",
                            "This filter is blocking first-move noise and improving entry quality.",
                        ],
                    },
                )

        price_action_eval = self._directional_price_action_expectations(
            setup_type=setup_type,
            structure_ctx=structure_ctx,
        )
        price_action_ok = bool(price_action_eval.get("price_action_ok"))
        retest_failed_against_setup = bool(price_action_eval.get("retest_failed_against_setup"))
        hard_block_pattern_against_setup = bool(price_action_eval.get("hard_block_pattern_against_setup"))
        hard_block_patterns = list(price_action_eval.get("hard_block_patterns") or [])
        price_action_details = list(price_action_eval.get("details") or [])
        if retest_failed_against_setup:
            reasons.append("RETEST_FAILED_AGAINST_SETUP")
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="WARN",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                price_action_bias=price_action_bias,
                price_action_confirmation=price_action_confirmation,
                retest_status=retest_status,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                strategy=setup_type,
                recommendation={
                    "headline": "Wait: retest failed against the setup.",
                    "signal_text": "Do not enter yet.",
                    "what_to_enter": "No trade now.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Wait for price to reclaim structure before taking a defined-risk entry.",
                    "why": price_action_details + [
                        "Price tested the level and failed in the wrong direction, which increases whipsaw risk.",
                    ],
                },
            )
        if hard_block_pattern_against_setup:
            reasons.append("REVERSAL_PATTERN_AGAINST_SETUP")
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="WARN",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                price_action_bias=price_action_bias,
                price_action_confirmation=price_action_confirmation,
                retest_status=retest_status,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                strategy=setup_type,
                recommendation={
                    "headline": "Wait: reversal pattern is against the trade.",
                    "signal_text": "Do not enter yet.",
                    "what_to_enter": "No trade now.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Wait for the fake break or reversal pattern to resolve before taking a credit spread.",
                    "why": price_action_details + [
                        "The chart is showing a confirmed reversal/fake-break pattern against this setup: "
                        + ", ".join(hard_block_patterns)
                        + ".",
                    ],
                },
            )
        if setup_type in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"} and bool(
            self.config.require_price_action_confirmation
        ):
            if not price_action_ok:
                reasons.append("PRICE_ACTION_CONFIRMATION_MISSING")
                return self._base_result(
                    now=now,
                    signal="WAIT",
                    priority="INFO",
                    reasons=reasons,
                    market_bias=market_bias,
                    trend_confidence=trend_confidence,
                    signal_conflict_score=signal_conflict_score,
                    pcr_unbalanced=pcr_unbalanced,
                    pcr_unbalanced_side=pcr_unbalanced_side,
                    volatility_regime=volatility_regime,
                    breakout_confirmation=breakout_confirmation,
                    price_action_bias=price_action_bias,
                    price_action_confirmation=price_action_confirmation,
                    retest_status=retest_status,
                    has_open_bkm=has_open_bkm,
                    expiry=expiry,
                    spot=spot,
                    strategy=setup_type,
                    recommendation={
                        "headline": "Wait: price action is not confirming the trade.",
                        "signal_text": "Do not enter yet.",
                        "what_to_enter": "No trade now.",
                        "sl_text": "Not applicable",
                        "tp_text": "Not applicable",
                        "hold_text": "Wait for candle confirmation or a clean retest hold in the same direction.",
                        "why": price_action_details + [
                            "Trend, EMA, and ORB may align, but the immediate price action is still not clean enough.",
                        ],
                    },
                )

        swing_structure_ok, swing_structure_details = self._swing_structure_supports_setup(
            setup_type=setup_type,
            structure_ctx=structure_ctx,
        )
        if not swing_structure_ok:
            reasons.append("SWING_STRUCTURE_AGAINST_SETUP")
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="INFO",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                price_action_bias=price_action_bias,
                price_action_confirmation=price_action_confirmation,
                retest_status=retest_status,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                strategy=setup_type,
                recommendation={
                    "headline": "Wait: swing structure is against the trade.",
                    "signal_text": "Do not enter yet.",
                    "what_to_enter": "No trade now.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Wait until the NIFTY swing structure aligns with the setup.",
                    "why": swing_structure_details + [
                        f"Current structure is {swing_structure_label} with {swing_structure_bias} bias.",
                    ],
                },
            )

        if setup_type in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"} and volatility_regime == "HIGH":
            supportive_retest = self._supportive_retest_for_setup(setup_type, retest_status)
            if not supportive_retest:
                reasons.append("HIGH_VOL_RETEST_REQUIRED")
                return self._base_result(
                    now=now,
                    signal="WAIT",
                    priority="INFO",
                    reasons=reasons,
                    market_bias=market_bias,
                    trend_confidence=trend_confidence,
                    signal_conflict_score=signal_conflict_score,
                    pcr_unbalanced=pcr_unbalanced,
                    pcr_unbalanced_side=pcr_unbalanced_side,
                    volatility_regime=volatility_regime,
                    breakout_confirmation=breakout_confirmation,
                    price_action_bias=price_action_bias,
                    price_action_confirmation=price_action_confirmation,
                    retest_status=retest_status,
                    has_open_bkm=has_open_bkm,
                    expiry=expiry,
                    spot=spot,
                    strategy=setup_type,
                    recommendation={
                        "headline": "Wait: high-volatility setup needs a clean retest.",
                        "signal_text": "Do not enter yet.",
                        "what_to_enter": "No trade now.",
                        "sl_text": "Not applicable",
                        "tp_text": "Not applicable",
                        "hold_text": "In high volatility, wait for a support/resistance retest before shorting premium.",
                        "why": [
                            f"Volatility regime is {volatility_regime}.",
                            "Candle confirmation alone is not enough in fast markets.",
                        ],
                    },
                )
            preferred_bias = str(getattr(self.config, "preferred_bias", "NEUTRAL") or "NEUTRAL").upper()
            if (
                setup_type == "PUT_CREDIT_SPREAD"
                and preferred_bias == "BEARISH"
                and not bool(self.config.high_vol_bull_put_enabled_for_bearish_preference)
            ):
                reasons.append("HIGH_VOL_BULL_PUT_DISABLED")
                return self._base_result(
                    now=now,
                    signal="WAIT",
                    priority="INFO",
                    reasons=reasons,
                    market_bias=market_bias,
                    trend_confidence=trend_confidence,
                    signal_conflict_score=signal_conflict_score,
                    pcr_unbalanced=pcr_unbalanced,
                    pcr_unbalanced_side=pcr_unbalanced_side,
                    volatility_regime=volatility_regime,
                    breakout_confirmation=breakout_confirmation,
                    price_action_bias=price_action_bias,
                    price_action_confirmation=price_action_confirmation,
                    retest_status=retest_status,
                    has_open_bkm=has_open_bkm,
                    expiry=expiry,
                    spot=spot,
                    strategy=setup_type,
                    recommendation={
                        "headline": "Wait: high-volatility bull put spread is disabled.",
                        "signal_text": "Do not enter this setup.",
                        "what_to_enter": "No trade now.",
                        "sl_text": "Not applicable",
                        "tp_text": "Not applicable",
                        "hold_text": "Stay out of high-volatility bullish put spreads while the system is tuned for bearish preference.",
                        "why": [
                            "This setup family is still a major source of drawdown in the broad replay.",
                            "The bearish side is currently more reliable than the bullish side in high volatility.",
                        ],
                    },
                )
            if (
                setup_type == "PUT_CREDIT_SPREAD"
                and preferred_bias == "BEARISH"
                and (trend_confidence < 0.86 or signal_conflict_score > 20.0)
            ):
                reasons.append("HIGH_VOL_COUNTERTREND_FILTER")
                return self._base_result(
                    now=now,
                    signal="WAIT",
                    priority="INFO",
                    reasons=reasons,
                    market_bias=market_bias,
                    trend_confidence=trend_confidence,
                    signal_conflict_score=signal_conflict_score,
                    pcr_unbalanced=pcr_unbalanced,
                    pcr_unbalanced_side=pcr_unbalanced_side,
                    volatility_regime=volatility_regime,
                    breakout_confirmation=breakout_confirmation,
                    price_action_bias=price_action_bias,
                    price_action_confirmation=price_action_confirmation,
                    retest_status=retest_status,
                    has_open_bkm=has_open_bkm,
                    expiry=expiry,
                    spot=spot,
                    strategy=setup_type,
                    recommendation={
                        "headline": "Wait: bullish countertrend spread is too risky in high volatility.",
                        "signal_text": "Do not enter yet.",
                        "what_to_enter": "No trade now.",
                        "sl_text": "Not applicable",
                        "tp_text": "Not applicable",
                        "hold_text": "Wait for a much cleaner bullish structure before taking a put spread against a bearish preference.",
                        "why": [
                            f"Trend confidence is {int(round(trend_confidence * 100))}% and conflict is {int(round(signal_conflict_score))}%.",
                            "High-volatility countertrend entries are a major source of drawdown.",
                        ],
                    },
                )

        if setup_type in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"} and not bool(plan.get("divergence_supportive")):
            reasons.append("RSI_DIVERGENCE_AGAINST_SETUP")
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="INFO",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_bias=pcr_bias,
                oi_build_bias=oi_build_bias,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                price_action_bias=price_action_bias,
                price_action_confirmation=price_action_confirmation,
                retest_status=retest_status,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                strategy=setup_type,
                recommendation={
                    "headline": "Wait: RSI divergence is against the setup.",
                    "signal_text": "Do not enter while momentum divergence is fighting the trade.",
                    "what_to_enter": str(plan.get("strike_hint") or "No trade now."),
                    "sl_text": "Not applicable yet",
                    "tp_text": "Not applicable yet",
                    "hold_text": "Wait for divergence to align with the planned direction or neutralize.",
                    "why": [
                        "The execution engine now treats RSI divergence as a hard confluence filter.",
                    ]
                    + list(plan.get("reason_summary") or [])[:4],
                    "plan": plan,
                },
            )

        def _round_to_step(value: float) -> float:
            return round(float(value) / step) * step

        strategy = setup_type
        strategy_emit = self._directional_strategy_name(setup_type)
        legs: List[Dict[str, Any]] = []
        est_credit_pts = 0.0
        invalidation_price: Optional[float] = None
        invalidation_text = ""
        planned_short_hint = _safe_float(plan.get("short_strike_hint"))
        planned_hedge_hint = _safe_float(plan.get("hedge_strike_hint"))
        planned_invalidation = _safe_float(plan.get("invalidation_spot"))
        emergency_hedge_width = self._emergency_hedge_distance(volatility_regime, step)

        if setup_type == "PUT_CREDIT_SPREAD":
            put_pad = dynamic_pad
            if strong_directional_alignment and market_bias == "BULLISH" and trend_confidence >= 0.72 and signal_conflict_score <= 25.0:
                put_pad = max(step, round((dynamic_pad * 0.80) / step) * step)
            support_ref = support_level if support_level is not None else (put_wall_strike if put_wall_strike is not None else spot - fallback_buf)
            target_short = min(
                support_ref - put_pad,
                (put_wall_strike - step if put_wall_strike is not None and put_wall_strike < spot else support_ref - put_pad),
            )
            heuristic_short_put = self._pick_leq(strikes, _round_to_step(target_short))
            if heuristic_short_put is None:
                heuristic_short_put = self._pick_leq(
                    strikes, planned_short_hint if planned_short_hint is not None else (spot - fallback_buf)
                )
            if heuristic_short_put is None:
                heuristic_short_put = self._pick_leq(strikes, spot - (4 * step))
            hedge_width = emergency_hedge_width if self._uses_short_option_with_hedge() else vol_width
            heuristic_hedge_put = (
                self._pick_leq(strikes, float(heuristic_short_put or 0.0) - hedge_width)
                if heuristic_short_put is not None
                else None
            )
            planned_short_put = self._pick_leq(strikes, planned_short_hint) if planned_short_hint is not None else None
            planned_hedge_put = (
                self._pick_leq(strikes, planned_hedge_hint)
                if planned_hedge_hint is not None
                else (
                    self._pick_leq(strikes, float(planned_short_put or 0.0) - hedge_width)
                    if planned_short_put is not None
                    else None
                )
            )
            candidates: List[Tuple[float, float, float]] = []
            for cand_short, cand_hedge in (
                (heuristic_short_put, heuristic_hedge_put),
                (planned_short_put, planned_hedge_put),
            ):
                if cand_short is None or cand_hedge is None or cand_hedge >= cand_short:
                    continue
                credit = max(
                    0.0,
                    (self._ltp(lookup, "PE", cand_short) or 0.0) - (self._ltp(lookup, "PE", cand_hedge) or 0.0),
                )
                candidates.append((credit, cand_short, cand_hedge))
            if candidates:
                _, short_put, hedge_put = max(candidates, key=lambda item: (item[0], -item[1]))
            else:
                short_put = None
                hedge_put = None
            if short_put is None or hedge_put is None or hedge_put >= short_put:
                reasons.append("STRIKE_SELECTION_FAILED_PUT_SPREAD")
                setup_type = ""
            else:
                sell_ltp = self._ltp(lookup, "PE", short_put) or 0.0
                buy_ltp = self._ltp(lookup, "PE", hedge_put) or 0.0
                est_credit_pts = max(0.0, sell_ltp - buy_ltp)
                legs = [
                    {"side": "SELL", "option_type": "PE", "strike": short_put, "qty_lots": 1},
                    {"side": "BUY", "option_type": "PE", "strike": hedge_put, "qty_lots": 1},
                ]
                strategy_emit = self._directional_strategy_name(setup_type)
                invalidation_price = planned_invalidation
                if invalidation_price is None:
                    invalidation_price = min(float(short_put) - put_pad * 0.5, (support_level or short_put) - put_pad * 0.35)
                invalidation_text = (
                    f"Exit if NIFTY sustains below {int(round(invalidation_price))} "
                    f"(bearish breakdown against bullish setup)."
                )
                if strategy_emit == "SHORT_PUT_WITH_HEDGE":
                    setup_reasons.append("Bullish bias supports short put selling with a far emergency hedge.")
                else:
                    setup_reasons.append("Bullish bias supports a put credit spread.")
        elif setup_type == "CALL_CREDIT_SPREAD":
            call_pad = dynamic_pad
            if strong_directional_alignment and market_bias == "BEARISH" and trend_confidence >= 0.72 and signal_conflict_score <= 25.0:
                call_pad = max(step, round((dynamic_pad * 0.80) / step) * step)
            resistance_ref = resistance_level if resistance_level is not None else (call_wall_strike if call_wall_strike is not None else spot + fallback_buf)
            target_short = max(
                resistance_ref + call_pad,
                (call_wall_strike + step if call_wall_strike is not None and call_wall_strike > spot else resistance_ref + call_pad),
            )
            heuristic_short_call = self._pick_geq(strikes, _round_to_step(target_short))
            if heuristic_short_call is None:
                heuristic_short_call = self._pick_geq(
                    strikes, planned_short_hint if planned_short_hint is not None else (spot + fallback_buf)
                )
            if heuristic_short_call is None:
                heuristic_short_call = self._pick_geq(strikes, spot + (4 * step))
            hedge_width = emergency_hedge_width if self._uses_short_option_with_hedge() else vol_width
            heuristic_hedge_call = (
                self._pick_geq(strikes, float(heuristic_short_call or 0.0) + hedge_width)
                if heuristic_short_call is not None
                else None
            )
            planned_short_call = self._pick_geq(strikes, planned_short_hint) if planned_short_hint is not None else None
            planned_hedge_call = (
                self._pick_geq(strikes, planned_hedge_hint)
                if planned_hedge_hint is not None
                else (
                    self._pick_geq(strikes, float(planned_short_call or 0.0) + hedge_width)
                    if planned_short_call is not None
                    else None
                )
            )
            candidates = []
            for cand_short, cand_hedge in (
                (heuristic_short_call, heuristic_hedge_call),
                (planned_short_call, planned_hedge_call),
            ):
                if cand_short is None or cand_hedge is None or cand_hedge <= cand_short:
                    continue
                credit = max(
                    0.0,
                    (self._ltp(lookup, "CE", cand_short) or 0.0) - (self._ltp(lookup, "CE", cand_hedge) or 0.0),
                )
                candidates.append((credit, cand_short, cand_hedge))
            if candidates:
                _, short_call, hedge_call = max(candidates, key=lambda item: (item[0], item[1]))
            else:
                short_call = None
                hedge_call = None
            if short_call is None or hedge_call is None or hedge_call <= short_call:
                reasons.append("STRIKE_SELECTION_FAILED_CALL_SPREAD")
                setup_type = ""
            else:
                sell_ltp = self._ltp(lookup, "CE", short_call) or 0.0
                buy_ltp = self._ltp(lookup, "CE", hedge_call) or 0.0
                est_credit_pts = max(0.0, sell_ltp - buy_ltp)
                legs = [
                    {"side": "SELL", "option_type": "CE", "strike": short_call, "qty_lots": 1},
                    {"side": "BUY", "option_type": "CE", "strike": hedge_call, "qty_lots": 1},
                ]
                strategy_emit = self._directional_strategy_name(setup_type)
                invalidation_price = planned_invalidation
                if invalidation_price is None:
                    invalidation_price = max(float(short_call) + call_pad * 0.5, (resistance_level or short_call) + call_pad * 0.35)
                invalidation_text = (
                    f"Exit if NIFTY sustains above {int(round(invalidation_price))} "
                    f"(bullish breakout against bearish setup)."
                )
                if strategy_emit == "SHORT_CALL_WITH_HEDGE":
                    setup_reasons.append("Bearish bias supports short call selling with a far emergency hedge.")
                else:
                    setup_reasons.append("Bearish bias supports a call credit spread.")
        elif setup_type == "IRON_CONDOR":
            sup_ref = support_level if support_level is not None else (spot - fallback_buf)
            res_ref = resistance_level if resistance_level is not None else (spot + fallback_buf)
            short_put = self._pick_leq(strikes, _round_to_step(sup_ref - dynamic_pad))
            short_call = self._pick_geq(strikes, _round_to_step(res_ref + dynamic_pad))
            hedge_put = self._pick_leq(strikes, float(short_put or 0.0) - vol_width) if short_put is not None else None
            hedge_call = self._pick_geq(strikes, float(short_call or 0.0) + vol_width) if short_call is not None else None
            if (
                short_put is None
                or short_call is None
                or hedge_put is None
                or hedge_call is None
                or not (hedge_put < short_put < spot < short_call < hedge_call)
            ):
                reasons.append("STRIKE_SELECTION_FAILED_IRON_CONDOR")
                setup_type = ""
            else:
                pe_credit = max(0.0, (self._ltp(lookup, "PE", short_put) or 0.0) - (self._ltp(lookup, "PE", hedge_put) or 0.0))
                ce_credit = max(0.0, (self._ltp(lookup, "CE", short_call) or 0.0) - (self._ltp(lookup, "CE", hedge_call) or 0.0))
                est_credit_pts = pe_credit + ce_credit
                legs = [
                    {"side": "SELL", "option_type": "PE", "strike": short_put, "qty_lots": 1},
                    {"side": "BUY", "option_type": "PE", "strike": hedge_put, "qty_lots": 1},
                    {"side": "SELL", "option_type": "CE", "strike": short_call, "qty_lots": 1},
                    {"side": "BUY", "option_type": "CE", "strike": hedge_call, "qty_lots": 1},
                ]
                lower_inv = min(float(short_put) - dynamic_pad * 0.35, (support_level or short_put) - dynamic_pad * 0.25)
                upper_inv = max(float(short_call) + dynamic_pad * 0.35, (resistance_level or short_call) + dynamic_pad * 0.25)
                invalidation_text = (
                    f"Exit if NIFTY breaks below {int(round(lower_inv))} or above {int(round(upper_inv))} "
                    "with sustained move (range break)."
                )
                setup_reasons.append("Range conditions favor an iron condor.")

        if not setup_type or not legs:
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="WARN",
                reasons=(reasons or ["STRIKE_SELECTION_FAILED"]),
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                recommendation={
                    "headline": "Wait: strike selection not clean.",
                    "signal_text": "Do not enter now.",
                    "what_to_enter": "No trade recommendation yet.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Wait for cleaner strikes/liquidity.",
                    "why": ["The advisor could not build a clean risk-defined structure from the available strikes."],
                },
            )

        est_credit_rs = float(est_credit_pts) * float(self.config.lot_size)
        min_credit_floor_rs = self._dynamic_credit_floor(
            setup_type=setup_type,
            base_credit_floor_rs=float(self.config.min_credit_per_set_rs),
            vol_width=vol_width,
            volatility_regime=volatility_regime,
            breakout_confirmation=breakout_confirmation,
            trend_confidence=trend_confidence,
            signal_conflict_score=signal_conflict_score,
            strong_directional_alignment=strong_directional_alignment,
        )
        if est_credit_rs < min_credit_floor_rs:
            reasons.append("LOW_PREMIUM_CREDIT")
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="INFO",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                has_open_bkm=has_open_bkm,
                expiry=expiry,
                spot=spot,
                recommendation={
                    "headline": "Wait: premium is too low.",
                    "signal_text": "Do not enter yet.",
                    "what_to_enter": "Setup exists, but premium is not attractive enough right now.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Wait for better premium or clearer move.",
                    "why": [
                        f"Estimated credit per 1-lot set is only Rs {est_credit_rs:,.0f}.",
                        f"Current minimum acceptable credit for this setup is about Rs {min_credit_floor_rs:,.0f}.",
                    ],
                },
            )

        trail_payload: Dict[str, Any] = {}
        unlimited_profit_mode = strategy_emit in {"SHORT_PUT_WITH_HEDGE", "SHORT_CALL_WITH_HEDGE"}
        if setup_type == "IRON_CONDOR":
            tp_capture = float(self.config.ic_target_capture_pct)
            sl_multiple = float(self.config.ic_stop_credit_multiple)
            tp_rs = max(0.0, est_credit_rs * tp_capture)
            sl_rs = max(0.0, est_credit_rs * sl_multiple)
        else:
            if unlimited_profit_mode:
                tp_capture = 0.0
                sl_multiple = 0.0
                tp_rs = 0.0
                sl_rs = float(self.config.naked_operational_max_loss_rs)
                trail_payload = {
                    "unlimited_profit_mode": True,
                    "arm_profit_rs_per_set": round(float(self.config.naked_profit_trail_arm_rs), 2),
                    "min_locked_profit_rs_per_set": round(float(self.config.naked_profit_trail_arm_rs), 2),
                    "keep_pct": round(float(self.config.naked_profit_trail_keep_pct), 3),
                    "text": (
                        f"No fixed cap. Once profit reaches about Rs {float(self.config.naked_profit_trail_arm_rs):,.0f} "
                        "per 1-lot set, trail it using live structure."
                    ),
                }
            else:
                tp_capture = float(self.config.spread_target_capture_pct)
                sl_multiple = float(self.config.spread_stop_credit_multiple)
                tp_rs = max(0.0, est_credit_rs * tp_capture)
                sl_rs = max(0.0, est_credit_rs * sl_multiple)
        hold_till = _parse_hhmm(self.config.max_hold_till, dtime(15, 5))

        if strategy_emit == "SHORT_PUT_WITH_HEDGE":
            strategy_label = "Short Put With Emergency Hedge"
            setup_reasons.append("Trend, structure, and support are aligned for bullish premium selling.")
            setup_reasons.append("Emergency hedge is kept far to preserve credit; operational risk is controlled by MTM stop.")
        elif strategy_emit == "SHORT_CALL_WITH_HEDGE":
            strategy_label = "Short Call With Emergency Hedge"
            setup_reasons.append("Trend, structure, and resistance are aligned for bearish premium selling.")
            setup_reasons.append("Emergency hedge is kept far to preserve credit; operational risk is controlled by MTM stop.")
        elif setup_type == "PUT_CREDIT_SPREAD":
            strategy_label = "Bull Put Credit Spread"
            setup_reasons.append("5m and 15m EMA(5/20) are aligned bullish.")
            setup_reasons.append("15m ORB confirms bullish continuation.")
        elif setup_type == "CALL_CREDIT_SPREAD":
            strategy_label = "Bear Call Credit Spread"
            setup_reasons.append("5m and 15m EMA(5/20) are aligned bearish.")
            setup_reasons.append("15m ORB confirms bearish continuation.")
        else:
            strategy_label = "Iron Condor (Defined Risk)"

        legs_txt = ", ".join(
            [
                f"{leg['side']} {int(leg['strike'])} {leg['option_type']} x{int(leg.get('qty_lots', 1))} lot"
                for leg in legs
            ]
        )
        what_to_enter = f"{strategy_label} ({expiry}) -> {legs_txt}"
        signal_text = "ENTER NOW" if entry_window_open else "WAIT"
        headline = f"{signal_text}: {strategy_label}"
        entry_not_before = _parse_hhmm(self.config.entry_not_before, dtime(9, 45))
        last_entry_time = _parse_hhmm(self.config.last_new_entry_time, dtime(14, 20))
        risk_notes = [
            "Scale lots only after your first set is working. Avoid adding size immediately.",
            "Do not exit early out of fear unless SL / invalidation is hit.",
        ]
        if pcr_unbalanced:
            risk_notes.append(f"PCR is unbalanced ({pcr_unbalanced_side.lower()}); position can move faster than usual.")
        if volatility_regime == "HIGH":
            risk_notes.append("Volatility is high. Expect faster MTM swings.")

        recommendation = {
            "signal": "ENTER_NOW",
            "headline": headline,
            "signal_text": signal_text,
            "strategy_type": strategy_emit,
            "strategy_label": strategy_label,
            "expiry": expiry,
            "what_to_enter": what_to_enter,
            "entry": {
                "window_text": (
                    f"Entry window: {entry_not_before.strftime('%H:%M')} to {last_entry_time.strftime('%H:%M')} IST."
                ),
                "trigger_text": str(plan.get("entry_trigger_text") or plan.get("trigger_window") or ""),
                "pullback_zone_low": _safe_float(plan.get("pullback_zone_low")),
                "pullback_zone_high": _safe_float(plan.get("pullback_zone_high")),
                "entry_zone_mid": _safe_float(plan.get("entry_zone_mid")),
            },
            "legs": legs,
            "est_credit_points_per_set": round(est_credit_pts, 2),
            "est_credit_rs_per_set": round(est_credit_rs, 2),
            "sl": {
                "kind": "POSITION_PNL_OR_INVALIDATION",
                "loss_rs_per_set": round(sl_rs, 2),
                "text": (
                    f"SL: Exit if loss on 1-lot set reaches about Rs {sl_rs:,.0f}, or invalidation is hit."
                    if not unlimited_profit_mode
                    else (
                        f"Operational SL: kill the trade near Rs {sl_rs:,.0f} loss on 1-lot set, "
                        "or exit earlier if structure invalidates."
                    )
                ),
            },
            "tp": {
                "kind": "UNLIMITED_TRAIL" if unlimited_profit_mode else "PREMIUM_CAPTURE",
                "profit_rs_per_set": round(tp_rs, 2),
                "capture_pct": round(tp_capture * 100.0, 1),
                "text": (
                    trail_payload.get("text")
                    if unlimited_profit_mode
                    else f"TP: Book around Rs {tp_rs:,.0f} profit on 1-lot set (about {int(round(tp_capture*100))}% premium capture)."
                ),
            },
            "trail": trail_payload,
            "hold": {
                "max_hold_till": hold_till.strftime("%H:%M"),
                "text": (
                    f"Hold till TP / SL / invalidation, or max till {hold_till.strftime('%I:%M %p').lstrip('0')} IST."
                ),
            },
            "invalidation": {
                "text": invalidation_text or "Exit if price action clearly breaks the setup structure.",
                "spot_level": None if invalidation_price is None else round(float(invalidation_price), 2),
            },
            "why": setup_reasons + [
                f"Trend confidence: {int(round(trend_confidence*100))}%, conflict: {int(round(signal_conflict_score))}%.",
                f"Market bias: {market_bias}. Volatility: {volatility_regime}.",
            ],
            "risk_notes": risk_notes,
            "discipline": {
                "avoid_overtrading": "Take only one setup unless this one is closed and conditions are still valid.",
                "fear_rule": "Do not exit early if AI still says HOLD/WATCH and none of TP/SL/invalidation is hit.",
            },
            "plan": plan,
        }
        if unlimited_profit_mode:
            recommendation["risk_notes"].append(
                "The far hedge is tail protection, not a mathematical guarantee. The real control is the operational MTM stop."
            )
            recommendation["risk_notes"].append(
                f"Once profit reaches about Rs {float(self.config.naked_profit_trail_arm_rs):,.0f}, the agent will trail it instead of using a fixed TP cap."
            )
        entry_features = self._entry_feature_snapshot(
            now=now,
            expiry=expiry,
            spot=spot,
            strategy=strategy_emit,
            market_bias=market_bias,
            trend_confidence=trend_confidence,
            signal_conflict_score=signal_conflict_score,
            volatility_regime=volatility_regime,
            breakout_confirmation=breakout_confirmation,
            ema_alignment_5m=ema_alignment_5m,
            ema_alignment_15m=ema_alignment_15m,
            orb_confirmation=orb_confirmation,
            structure_ctx=structure_ctx,
            oc_ctx=oc_ctx,
            recommendation=recommendation,
            trade_plan=plan,
        )
        recommendation["entry_features"] = entry_features
        candidate_signature = self._candidate_signature(
            setup_type=strategy_emit,
            market_bias=market_bias,
            orb_confirmation=orb_confirmation,
            expiry=expiry,
            legs=legs,
        )
        same_session = str(self.state.current_session_date or "") == now.date().isoformat()
        prev_signature = str(self.state.candidate_setup_signature or "") if same_session else ""
        prev_streak = int(self.state.candidate_signal_streak or 0) if same_session else 0
        candidate_streak = (prev_streak + 1) if (prev_signature and prev_signature == candidate_signature) else 1
        required_streak = max(1, int(self.config.signal_persistence_bars or 1))
        if self._trend_day_fast_track_ready(
            setup_type=setup_type,
            market_bias=market_bias,
            trend_confidence=trend_confidence,
            signal_conflict_score=signal_conflict_score,
            strong_directional_alignment=strong_directional_alignment,
            price_action_confirmation=price_action_confirmation,
        ):
            required_streak = 1
        if candidate_streak < required_streak:
            reasons.append("SIGNAL_NOT_PERSISTED_YET")
            return self._base_result(
                now=now,
                signal="WAIT",
                priority="INFO",
                reasons=reasons,
                market_bias=market_bias,
                trend_confidence=trend_confidence,
                signal_conflict_score=signal_conflict_score,
                pcr_unbalanced=pcr_unbalanced,
                pcr_unbalanced_side=pcr_unbalanced_side,
                volatility_regime=volatility_regime,
                breakout_confirmation=breakout_confirmation,
                price_action_bias=price_action_bias,
                price_action_confirmation=price_action_confirmation,
                retest_status=retest_status,
                has_open_bkm=has_open_bkm,
                recommendation={
                    "headline": "Wait: setup needs one more confirmation bar.",
                    "signal_text": "Do not enter on the first qualifying tick.",
                    "what_to_enter": what_to_enter,
                    "sl_text": "Not applicable yet",
                    "tp_text": "Not applicable yet",
                    "hold_text": "If the same setup survives the next bar, the advisor can promote it to ENTER NOW.",
                    "why": [
                        f"Persistence streak is {candidate_streak}/{required_streak}.",
                        "This filter reduces whipsaw entries and lowers drawdown.",
                    ],
                },
                expiry=expiry,
                spot=spot,
                strategy=strategy_emit,
                candidate_setup_signature=candidate_signature,
                candidate_signal_streak=candidate_streak,
                entry_features=entry_features,
            )
        return self._base_result(
            now=now,
            signal="ENTER_NOW",
            priority="INFO",
            reasons=reasons or ["SETUP_READY"],
            market_bias=market_bias,
            trend_confidence=trend_confidence,
            signal_conflict_score=signal_conflict_score,
            pcr_unbalanced=pcr_unbalanced,
            pcr_unbalanced_side=pcr_unbalanced_side,
            volatility_regime=volatility_regime,
            breakout_confirmation=breakout_confirmation,
            price_action_bias=price_action_bias,
            price_action_confirmation=price_action_confirmation,
            retest_status=retest_status,
            has_open_bkm=has_open_bkm,
            recommendation=recommendation,
            expiry=expiry,
            spot=spot,
            strategy=strategy_emit,
            candidate_setup_signature=candidate_signature,
            candidate_signal_streak=candidate_streak,
            entry_features=entry_features,
        )

    def update(
        self,
        *,
        now: datetime,
        expiry: str,
        spot: float,
        chain_rows: List[Dict[str, Any]],
        context: Dict[str, Any],
        has_open_bkm: bool,
    ) -> Dict[str, Any]:
        out = self.evaluate(
            now=now,
            expiry=expiry,
            spot=spot,
            chain_rows=chain_rows,
            context=context,
            has_open_bkm=has_open_bkm,
        )
        oc_ctx = context.get("option_chain") if isinstance(context.get("option_chain"), dict) else {}
        oi_build_ctx = oc_ctx.get("oi_build") if isinstance(oc_ctx.get("oi_build"), dict) else {}
        out["pcr_bias"] = str(out.get("pcr_bias") or oc_ctx.get("pcr_bias") or "NEUTRAL")
        out["oi_build_bias"] = str(out.get("oi_build_bias") or oi_build_ctx.get("bias") or "UNKNOWN")
        prev_signal = str(self.state.signal or "NO_TRADE")
        prev_strategy = str(self.state.strategy or "")
        prev_updated = self.state.updated_at

        self.state.status = str(out.get("status") or "ACTIVE")
        self.state.signal = str(out.get("signal") or "NO_TRADE")
        self.state.priority = str(out.get("priority") or "INFO")
        self.state.strategy = out.get("strategy")
        self.state.expiry = out.get("expiry")
        self.state.spot = out.get("spot")
        self.state.market_bias = str(out.get("market_bias") or "NEUTRAL")
        self.state.trend_confidence = float(out.get("trend_confidence") or 0.0)
        self.state.signal_conflict_score = float(out.get("signal_conflict_score") or 0.0)
        self.state.pcr_bias = str(out.get("pcr_bias") or "NEUTRAL")
        self.state.oi_build_bias = str(out.get("oi_build_bias") or "UNKNOWN")
        self.state.pcr_unbalanced = bool(out.get("pcr_unbalanced", False))
        self.state.pcr_unbalanced_side = str(out.get("pcr_unbalanced_side") or "NEUTRAL")
        self.state.volatility_regime = str(out.get("volatility_regime") or "UNKNOWN")
        self.state.breakout_confirmation = str(out.get("breakout_confirmation") or "NONE")
        self.state.price_action_bias = str(out.get("price_action_bias") or structure_ctx.get("price_action_bias") or "NEUTRAL")
        self.state.price_action_confirmation = str(
            out.get("price_action_confirmation") or structure_ctx.get("price_action_confirmation") or "NONE"
        )
        self.state.retest_status = str(out.get("retest_status") or structure_ctx.get("retest_status") or "NONE")
        tf_map = context.get("trend", {}).get("timeframes") if isinstance(context.get("trend"), dict) else {}
        tf5 = tf_map.get("5") if isinstance(tf_map.get("5"), dict) else {}
        tf15 = tf_map.get("15") if isinstance(tf_map.get("15"), dict) else {}
        orb_ctx = context.get("trend", {}).get("orb") if isinstance(context.get("trend"), dict) else {}
        orb_ctx = orb_ctx if isinstance(orb_ctx, dict) else {}
        self.state.ema_alignment_5m = str(tf5.get("ema_alignment") or "NEUTRAL")
        self.state.ema_alignment_15m = str(tf15.get("ema_alignment") or "NEUTRAL")
        self.state.orb_confirmation = str(orb_ctx.get("breakout_confirmation") or "NONE")
        self.state.has_open_bkm = bool(out.get("has_open_bkm", False))
        self.state.market_context = context if isinstance(context, dict) else {}
        self.state.recommendation = out.get("recommendation") if isinstance(out.get("recommendation"), dict) else {}
        self.state.reasons = list(out.get("reasons") or [])
        self.state.candidate_setup_signature = out.get("candidate_setup_signature")
        self.state.candidate_signal_streak = max(0, int(out.get("candidate_signal_streak") or 0))
        self.state.current_session_date = str(out.get("current_session_date") or now.date().isoformat())
        self._persist(now)

        signal_changed = (self.state.signal != prev_signal) or (str(self.state.strategy or "") != prev_strategy)
        out["signal_changed"] = signal_changed
        out["updated_at"] = self.state.updated_at
        out["previous_updated_at"] = prev_updated
        return out
