from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

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

    @staticmethod
    def _directional_bias_for_setup(setup_type: str) -> str:
        text = str(setup_type or "").upper()
        if text == "PUT_CREDIT_SPREAD":
            return "BULLISH"
        if text == "CALL_CREDIT_SPREAD":
            return "BEARISH"
        return "NEUTRAL"

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
    ) -> Tuple[bool, bool, bool, List[str]]:
        if setup_type not in {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"}:
            return True, False, False, []
        expected_bias = "BULLISH" if setup_type == "PUT_CREDIT_SPREAD" else "BEARISH"
        price_action_bias = str(structure_ctx.get("price_action_bias") or "NEUTRAL").upper()
        price_action_confirmation = str(structure_ctx.get("price_action_confirmation") or "NONE").upper()
        retest_status = str(structure_ctx.get("retest_status") or "NONE").upper()
        retest_bias = str(structure_ctx.get("retest_bias") or "NEUTRAL").upper()
        candle_patterns = structure_ctx.get("price_action_patterns")
        candle_patterns = candle_patterns if isinstance(candle_patterns, list) else []
        price_action_ok = (
            price_action_bias == expected_bias
            and price_action_confirmation
            in {
                "CANDLE_CONFIRMED",
                "RETEST_CONFIRMED",
                "CANDLE_AND_RETEST_CONFIRMED",
            }
        )
        retest_ok = (
            retest_bias == expected_bias
            and retest_status
            in {
                "SUPPORT_HOLD",
                "RESISTANCE_HOLD",
                "BREAKOUT_RETEST_HOLD",
                "BREAKDOWN_RETEST_HOLD",
            }
        )
        retest_failed_against_setup = (
            retest_status in {"RETEST_FAILED", "FAILED_BREAKOUT_RETEST"}
            and retest_bias not in {"NEUTRAL", expected_bias}
        )
        details: List[str] = []
        if candle_patterns:
            details.append("Price action patterns: " + ", ".join(str(item) for item in candle_patterns[:4]))
        details.append(f"Price action bias is {price_action_bias}.")
        details.append(f"Retest status is {retest_status}.")
        return (price_action_ok or retest_ok), price_action_ok, retest_failed_against_setup, details

    @staticmethod
    def _supportive_retest_for_setup(setup_type: str, retest_status: str) -> bool:
        status = str(retest_status or "NONE").upper()
        setup = str(setup_type or "").upper()
        if setup == "CALL_CREDIT_SPREAD":
            return status in {"RESISTANCE_HOLD", "BREAKDOWN_RETEST_HOLD"}
        if setup == "PUT_CREDIT_SPREAD":
            return status in {"SUPPORT_HOLD", "BREAKOUT_RETEST_HOLD"}
        return False

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
    ) -> Dict[str, Any]:
        intraday_support = structure_ctx.get("intraday_support") if isinstance(structure_ctx.get("intraday_support"), dict) else {}
        intraday_resistance = structure_ctx.get("intraday_resistance") if isinstance(structure_ctx.get("intraday_resistance"), dict) else {}
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
            "price_action_patterns": list(structure_ctx.get("price_action_patterns") or []),
            "retest_status": str(structure_ctx.get("retest_status") or "NONE"),
            "retest_bias": str(structure_ctx.get("retest_bias") or "NEUTRAL"),
            "pcr_total": _safe_float(oc_ctx.get("pcr_total")),
            "pcr_near_atm": _safe_float(oc_ctx.get("pcr_near_atm")),
            "pcr_bias": str(oc_ctx.get("pcr_bias") or "NEUTRAL"),
            "oi_build_bias": str(((oc_ctx.get("oi_build") or {}) if isinstance(oc_ctx.get("oi_build"), dict) else {}).get("bias") or "UNKNOWN"),
            "intraday_support": _safe_float(intraday_support.get("level")),
            "intraday_resistance": _safe_float(intraday_resistance.get("level")),
            "sr_alignment_hits": int(structure_ctx.get("sr_alignment_hits") or 0),
            "legs": list(recommendation.get("legs") or []),
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
        if (
            market_bias == "BULLISH"
            and trend_confidence >= directional_min_conf
            and signal_conflict_score <= directional_max_conflict
            and breakout_confirmation != "DOWN_CONFIRMED"
        ):
            setup_type = "PUT_CREDIT_SPREAD"
            setup_reasons.append("Bias is bullish with acceptable signal conflict.")
            if strong_directional_alignment:
                setup_reasons.append("Multi-timeframe trend is aligned, so the bullish spread threshold is slightly relaxed.")
        elif (
            market_bias == "BEARISH"
            and trend_confidence >= directional_min_conf
            and signal_conflict_score <= directional_max_conflict
            and breakout_confirmation != "UP_CONFIRMED"
        ):
            setup_type = "CALL_CREDIT_SPREAD"
            setup_reasons.append("Bias is bearish with acceptable signal conflict.")
            if strong_directional_alignment:
                setup_reasons.append("Multi-timeframe trend is aligned, so the bearish spread threshold is slightly relaxed.")
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
            if range_ok:
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
                    "what_to_enter": "No trade recommendation at this moment.",
                    "sl_text": "Not applicable",
                    "tp_text": "Not applicable",
                    "hold_text": "Wait for clearer trend/range structure and lower signal conflict.",
                    "why": [
                        f"Bias={market_bias}, trend confidence={int(round(trend_confidence*100))}%, conflict={int(round(signal_conflict_score))}%.",
                        "Setup quality is not high enough for a disciplined intraday short option entry.",
                    ],
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

        price_action_ok, candle_confirmation_ok, retest_failed_against_setup, price_action_details = (
            self._directional_price_action_expectations(
                setup_type=setup_type,
                structure_ctx=structure_ctx,
            )
        )
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

        def _round_to_step(value: float) -> float:
            return round(float(value) / step) * step

        strategy = setup_type
        legs: List[Dict[str, Any]] = []
        est_credit_pts = 0.0
        invalidation_price: Optional[float] = None
        invalidation_text = ""

        if setup_type == "PUT_CREDIT_SPREAD":
            put_pad = dynamic_pad
            if strong_directional_alignment and market_bias == "BULLISH" and trend_confidence >= 0.72 and signal_conflict_score <= 25.0:
                put_pad = max(step, round((dynamic_pad * 0.80) / step) * step)
            support_ref = support_level if support_level is not None else (put_wall_strike if put_wall_strike is not None else spot - fallback_buf)
            target_short = min(
                support_ref - put_pad,
                (put_wall_strike - step if put_wall_strike is not None and put_wall_strike < spot else support_ref - put_pad),
            )
            short_put = self._pick_leq(strikes, _round_to_step(target_short))
            if short_put is None:
                short_put = self._pick_leq(strikes, spot - fallback_buf)
            if short_put is None:
                short_put = self._pick_leq(strikes, spot - (4 * step))
            hedge_put = self._pick_leq(strikes, float(short_put or 0.0) - vol_width) if short_put is not None else None
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
                invalidation_price = min(float(short_put) - put_pad * 0.5, (support_level or short_put) - put_pad * 0.35)
                invalidation_text = (
                    f"Exit if NIFTY sustains below {int(round(invalidation_price))} "
                    f"(bearish breakdown against bullish setup)."
                )
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
            short_call = self._pick_geq(strikes, _round_to_step(target_short))
            if short_call is None:
                short_call = self._pick_geq(strikes, spot + fallback_buf)
            if short_call is None:
                short_call = self._pick_geq(strikes, spot + (4 * step))
            hedge_call = self._pick_geq(strikes, float(short_call or 0.0) + vol_width) if short_call is not None else None
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
                invalidation_price = max(float(short_call) + call_pad * 0.5, (resistance_level or short_call) + call_pad * 0.35)
                invalidation_text = (
                    f"Exit if NIFTY sustains above {int(round(invalidation_price))} "
                    f"(bullish breakout against bearish setup)."
                )
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

        if setup_type == "IRON_CONDOR":
            tp_capture = float(self.config.ic_target_capture_pct)
            sl_multiple = float(self.config.ic_stop_credit_multiple)
        else:
            tp_capture = float(self.config.spread_target_capture_pct)
            sl_multiple = float(self.config.spread_stop_credit_multiple)
        tp_rs = max(0.0, est_credit_rs * tp_capture)
        sl_rs = max(0.0, est_credit_rs * sl_multiple)
        hold_till = _parse_hhmm(self.config.max_hold_till, dtime(15, 5))

        if setup_type == "PUT_CREDIT_SPREAD":
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
            "strategy_type": setup_type,
            "strategy_label": strategy_label,
            "expiry": expiry,
            "what_to_enter": what_to_enter,
            "legs": legs,
            "est_credit_points_per_set": round(est_credit_pts, 2),
            "est_credit_rs_per_set": round(est_credit_rs, 2),
            "sl": {
                "kind": "POSITION_PNL_OR_INVALIDATION",
                "loss_rs_per_set": round(sl_rs, 2),
                "text": f"SL: Exit if loss on 1-lot set reaches about Rs {sl_rs:,.0f}, or invalidation is hit.",
            },
            "tp": {
                "kind": "PREMIUM_CAPTURE",
                "profit_rs_per_set": round(tp_rs, 2),
                "capture_pct": round(tp_capture * 100.0, 1),
                "text": f"TP: Book around Rs {tp_rs:,.0f} profit on 1-lot set (about {int(round(tp_capture*100))}% premium capture).",
            },
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
        }
        entry_features = self._entry_feature_snapshot(
            now=now,
            expiry=expiry,
            spot=spot,
            strategy=setup_type,
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
        )
        recommendation["entry_features"] = entry_features
        candidate_signature = self._candidate_signature(
            setup_type=setup_type,
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
                strategy=setup_type,
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
            strategy=setup_type,
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
