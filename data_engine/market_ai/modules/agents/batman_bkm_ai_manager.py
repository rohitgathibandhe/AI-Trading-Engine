from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


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


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except Exception:
        return None


@dataclass
class BatmanBKMAIConfig:
    enabled: bool = True
    mode: str = "ADVISOR"
    sample_window_sec: float = 600.0
    open_grace_sec: float = 180.0
    near_short_buffer_points: float = 200.0
    velocity_warn_pts_per_min: float = 90.0
    warn_score: float = 30.0
    reduce_score: float = 55.0
    flatten_score: float = 80.0
    loss_ratio_warn: float = 0.50
    loss_ratio_critical: float = 0.90
    drawdown_ratio_warn: float = 0.60
    drawdown_ratio_critical: float = 0.90
    event_emit_min_interval_sec: float = 30.0
    pcr_bullish_threshold: float = 1.15
    pcr_bearish_threshold: float = 0.85
    trend_bias_threshold: float = 0.35
    context_risk_score_cap: float = 20.0
    protective_lock_min_action: str = "REDUCE_RISK"
    protective_watch_escalates_alert: bool = True

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "BatmanBKMAIConfig":
        return cls(
            enabled=bool(settings.get("batman_bkm_ai_enabled", True)),
            mode=str(settings.get("batman_bkm_ai_mode", "ADVISOR")).strip().upper() or "ADVISOR",
            sample_window_sec=max(30.0, float(settings.get("batman_bkm_ai_sample_window_sec", 600.0))),
            open_grace_sec=max(0.0, float(settings.get("batman_bkm_ai_open_grace_sec", 180.0))),
            near_short_buffer_points=max(10.0, float(settings.get("batman_bkm_ai_near_short_buffer_points", 200.0))),
            velocity_warn_pts_per_min=max(1.0, float(settings.get("batman_bkm_ai_velocity_warn_pts_per_min", 90.0))),
            warn_score=max(1.0, float(settings.get("batman_bkm_ai_warn_score", 30.0))),
            reduce_score=max(1.0, float(settings.get("batman_bkm_ai_reduce_score", 55.0))),
            flatten_score=max(1.0, float(settings.get("batman_bkm_ai_flatten_score", 80.0))),
            loss_ratio_warn=max(0.05, float(settings.get("batman_bkm_ai_loss_ratio_warn", 0.50))),
            loss_ratio_critical=max(0.10, float(settings.get("batman_bkm_ai_loss_ratio_critical", 0.90))),
            drawdown_ratio_warn=max(0.05, float(settings.get("batman_bkm_ai_drawdown_ratio_warn", 0.60))),
            drawdown_ratio_critical=max(0.10, float(settings.get("batman_bkm_ai_drawdown_ratio_critical", 0.90))),
            event_emit_min_interval_sec=max(
                0.0,
                float(settings.get("batman_bkm_ai_event_emit_min_interval_sec", 30.0)),
            ),
            pcr_bullish_threshold=max(1.0, float(settings.get("batman_bkm_ai_pcr_bullish_threshold", 1.15))),
            pcr_bearish_threshold=min(1.0, float(settings.get("batman_bkm_ai_pcr_bearish_threshold", 0.85))),
            trend_bias_threshold=max(0.1, float(settings.get("batman_bkm_ai_trend_bias_threshold", 0.35))),
            context_risk_score_cap=max(0.0, float(settings.get("batman_bkm_ai_context_risk_score_cap", 20.0))),
            protective_lock_min_action=str(settings.get("batman_bkm_ai_protective_lock_min_action", "REDUCE_RISK")).strip().upper() or "REDUCE_RISK",
            protective_watch_escalates_alert=bool(settings.get("batman_bkm_ai_protective_watch_escalates_alert", True)),
        )


@dataclass
class BatmanBKMAIState:
    status: str = "IDLE"
    mode: str = "ADVISOR"
    action: str = "HOLD"
    severity: str = "INFO"
    score: float = 0.0
    confidence: float = 0.0
    reasons: List[str] = None  # type: ignore[assignment]
    current_session_date: str = ""
    basket_expiry: Optional[str] = None
    spot: Optional[float] = None
    pnl: Optional[float] = None
    tp: Optional[float] = None
    sl: Optional[float] = None
    dte: Optional[int] = None
    short_range_low: Optional[float] = None
    short_range_high: Optional[float] = None
    outer_range_low: Optional[float] = None
    outer_range_high: Optional[float] = None
    distance_to_nearest_short: Optional[float] = None
    outside_short_range: bool = False
    outside_outer_range: bool = False
    spot_window_span: Optional[float] = None
    spot_velocity_pts_per_min: Optional[float] = None
    pnl_peak: Optional[float] = None
    pnl_drawdown: Optional[float] = None
    loss_ratio_to_sl: Optional[float] = None
    drawdown_ratio_to_tp: Optional[float] = None
    samples_in_window: int = 0
    day_mode: Optional[str] = None
    position_risk_side: Optional[str] = None
    market_bias: Optional[str] = None
    market_context: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    last_action_changed_at: Optional[str] = None
    updated_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if payload.get("reasons") is None:
            payload["reasons"] = []
        if payload.get("market_context") is None:
            payload["market_context"] = {}
        if payload.get("plan") is None:
            payload["plan"] = {}
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "BatmanBKMAIState":
        data = dict(payload or {})
        if not isinstance(data.get("reasons"), list):
            data["reasons"] = []
        if not isinstance(data.get("market_context"), dict):
            data["market_context"] = {}
        if not isinstance(data.get("plan"), dict):
            data["plan"] = {}
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__.keys()})  # type: ignore[arg-type]


class BatmanBKMAIManager:
    def __init__(self, *, config: BatmanBKMAIConfig, status_path: Path, events_path: Path, logger: Any = None) -> None:
        self.config = config
        self.status_path = status_path
        self.events_path = events_path
        self.log = logger
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.events_path.exists():
            try:
                self.events_path.write_text("")
            except Exception:
                pass
        self.state = self._load_state()
        self._samples: Deque[Tuple[float, float, float]] = deque()
        self._last_event_emit_ts: Optional[float] = None
        self._peak_pnl: float = float(self.state.pnl_peak or 0.0 or 0.0)

    def _log(self, level: str, msg: str, *args: Any) -> None:
        if self.log and hasattr(self.log, level):
            getattr(self.log, level)(msg, *args)

    def _load_state(self) -> BatmanBKMAIState:
        if not self.status_path.exists():
            st = BatmanBKMAIState()
            st.reasons = []
            return st
        try:
            payload = json.loads(self.status_path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("invalid ai state payload")
            st = BatmanBKMAIState.from_payload(payload)
            if st.reasons is None:
                st.reasons = []
            return st
        except Exception:
            st = BatmanBKMAIState()
            st.reasons = []
            return st

    def _persist(self, when: Optional[datetime] = None) -> None:
        now = when or _now_ist()
        self.state.updated_at = now.isoformat(timespec="seconds")
        try:
            self.status_path.write_text(json.dumps(self.state.as_dict(), indent=2, default=str))
        except Exception:
            self._log("exception", "Failed to write BKM AI status")

    def _append_event(self, row: Dict[str, Any]) -> None:
        try:
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except Exception:
            self._log("exception", "Failed to append BKM AI event")

    def snapshot(self) -> Dict[str, Any]:
        return self.state.as_dict()

    def reset_idle(self, *, when: Optional[datetime] = None, reason: str = "NO_BASKET") -> Dict[str, Any]:
        now = when or _now_ist()
        prior_status = self.state.status
        prior_action = self.state.action
        if prior_status == "IDLE" and prior_action == "HOLD":
            return self.snapshot()
        self._samples.clear()
        self._peak_pnl = 0.0
        self.state.status = "IDLE"
        self.state.action = "HOLD"
        self.state.severity = "INFO"
        self.state.score = 0.0
        self.state.confidence = 0.0
        self.state.reasons = [reason]
        self.state.basket_expiry = None
        self.state.spot = None
        self.state.pnl = None
        self.state.tp = None
        self.state.sl = None
        self.state.dte = None
        self.state.short_range_low = None
        self.state.short_range_high = None
        self.state.outer_range_low = None
        self.state.outer_range_high = None
        self.state.distance_to_nearest_short = None
        self.state.outside_short_range = False
        self.state.outside_outer_range = False
        self.state.spot_window_span = None
        self.state.spot_velocity_pts_per_min = None
        self.state.pnl_peak = None
        self.state.pnl_drawdown = None
        self.state.loss_ratio_to_sl = None
        self.state.drawdown_ratio_to_tp = None
        self.state.samples_in_window = 0
        self.state.day_mode = None
        self.state.position_risk_side = None
        self.state.market_bias = None
        self.state.market_context = {}
        self.state.plan = {}
        self.state.current_session_date = now.date().isoformat()
        self.state.last_action_changed_at = now.isoformat(timespec="seconds")
        self._persist(now)
        self._emit_change_event(now, changed=True, force=True)
        return self.snapshot()

    def _emit_change_event(self, when: datetime, *, changed: bool, force: bool = False) -> None:
        if not changed and not force:
            return
        now_ts = when.timestamp()
        if not force and self._last_event_emit_ts is not None:
            if (now_ts - self._last_event_emit_ts) < float(self.config.event_emit_min_interval_sec):
                return
        row = {
            "timestamp": when.isoformat(timespec="seconds"),
            "status": self.state.status,
            "mode": self.state.mode,
            "action": self.state.action,
            "severity": self.state.severity,
            "score": round(float(self.state.score or 0.0), 2),
            "confidence": round(float(self.state.confidence or 0.0), 3),
            "reasons": list(self.state.reasons or []),
            "basket_expiry": self.state.basket_expiry,
            "spot": self.state.spot,
            "pnl": self.state.pnl,
            "tp": self.state.tp,
            "sl": self.state.sl,
            "dte": self.state.dte,
            "day_mode": self.state.day_mode,
            "position_risk_side": self.state.position_risk_side,
            "market_bias": self.state.market_bias,
        }
        self._append_event(row)
        self._last_event_emit_ts = now_ts

    def _prune_samples(self, now_ts: float) -> None:
        cutoff = now_ts - float(self.config.sample_window_sec)
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _extract_ranges(self, basket: Any) -> Dict[str, Optional[float]]:
        legs = list(getattr(basket, "legs", []) or [])
        ce_buys = sorted(
            [float(getattr(l, "strike", 0.0) or 0.0) for l in legs if str(getattr(l, "option_type", "")).upper() == "CE" and str(getattr(l, "side", "")).upper() == "BUY"]
        )
        ce_sells = sorted(
            [float(getattr(l, "strike", 0.0) or 0.0) for l in legs if str(getattr(l, "option_type", "")).upper() == "CE" and str(getattr(l, "side", "")).upper() == "SELL"]
        )
        pe_buys = sorted(
            [float(getattr(l, "strike", 0.0) or 0.0) for l in legs if str(getattr(l, "option_type", "")).upper() == "PE" and str(getattr(l, "side", "")).upper() == "BUY"]
        )
        pe_sells = sorted(
            [float(getattr(l, "strike", 0.0) or 0.0) for l in legs if str(getattr(l, "option_type", "")).upper() == "PE" and str(getattr(l, "side", "")).upper() == "SELL"]
        )
        ce_short = ce_sells[0] if len(ce_sells) == 1 else None
        pe_short = pe_sells[0] if len(pe_sells) == 1 else None
        ce_outer = max(ce_buys) if ce_buys else None
        pe_outer = min(pe_buys) if pe_buys else None
        ce_inner = min(ce_buys) if ce_buys else None
        pe_inner = max(pe_buys) if pe_buys else None
        return {
            "ce_short": ce_short,
            "pe_short": pe_short,
            "ce_outer": ce_outer,
            "pe_outer": pe_outer,
            "ce_inner": ce_inner,
            "pe_inner": pe_inner,
        }

    def _score(
        self,
        *,
        now: datetime,
        basket: Any,
        spot: float,
        pnl: float,
        tp: float,
        sl: float,
        day_mode: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rng = self._extract_ranges(basket)
        ce_short = rng.get("ce_short")
        pe_short = rng.get("pe_short")
        ce_outer = rng.get("ce_outer")
        pe_outer = rng.get("pe_outer")
        if ce_short is None or pe_short is None:
            return {
                "score": 0.0,
                "action": "HOLD",
                "severity": "INFO",
                "reasons": ["AI_PARSE_RANGE_FAILED"],
                "features": {},
                "confidence": 0.2,
            }

        now_ts = now.timestamp()
        self._samples.append((now_ts, float(spot), float(pnl)))
        self._prune_samples(now_ts)

        samples = list(self._samples)
        spot_span: Optional[float] = None
        spot_vel: Optional[float] = None
        if len(samples) >= 2:
            spot_vals = [s[1] for s in samples]
            spot_span = max(spot_vals) - min(spot_vals)
            dt_sec = max(1e-6, samples[-1][0] - samples[0][0])
            spot_vel = abs(samples[-1][1] - samples[0][1]) / (dt_sec / 60.0)

        self._peak_pnl = max(float(self._peak_pnl or 0.0), float(pnl))
        pnl_drawdown = max(0.0, float(self._peak_pnl or 0.0) - float(pnl))
        loss_ratio_to_sl: Optional[float] = None
        if float(sl or 0.0) < 0 and float(pnl) < 0:
            loss_ratio_to_sl = abs(float(pnl)) / max(1.0, abs(float(sl)))
        drawdown_ratio_to_tp: Optional[float] = None
        if float(tp or 0.0) > 0 and pnl_drawdown > 0:
            drawdown_ratio_to_tp = pnl_drawdown / max(1.0, float(tp))

        outside_short = bool(spot < pe_short or spot > ce_short)
        outside_outer = False
        if ce_outer is not None and pe_outer is not None:
            outside_outer = bool(spot < pe_outer or spot > ce_outer)
        dist_near_short = min(abs(float(spot) - pe_short), abs(ce_short - float(spot)))
        short_width = max(1.0, ce_short - pe_short)
        dist_to_put_short = abs(float(spot) - pe_short)
        dist_to_call_short = abs(ce_short - float(spot))
        risk_side = "CENTER"
        if abs(dist_to_call_short - dist_to_put_short) <= 50:
            risk_side = "CENTER"
        elif dist_to_call_short < dist_to_put_short:
            risk_side = "CALL"
        else:
            risk_side = "PUT"
        reasons: List[str] = []
        score = 0.0

        # Early grace: keep advisory but suppress aggressive calls right after entry.
        entry_ts = _parse_dt(getattr(basket, "entry_ts", None))
        in_grace = False
        if entry_ts is not None:
            in_grace = (now - entry_ts).total_seconds() < float(self.config.open_grace_sec)

        if outside_outer:
            score += 90.0
            reasons.append("SPOT_OUTSIDE_OUTER_WINGS")
        elif outside_short:
            score += 48.0
            reasons.append("SPOT_OUTSIDE_SHORT_RANGE")
            if ce_outer is not None and spot > ce_short:
                beyond = max(0.0, float(spot) - ce_short)
                width = max(50.0, float(ce_outer) - ce_short)
                score += 20.0 * min(1.0, beyond / width)
            elif pe_outer is not None and spot < pe_short:
                beyond = max(0.0, pe_short - float(spot))
                width = max(50.0, pe_short - float(pe_outer))
                score += 20.0 * min(1.0, beyond / width)

        near_buf = max(10.0, float(self.config.near_short_buffer_points))
        if dist_near_short <= near_buf:
            score += 18.0 * (1.0 - (dist_near_short / near_buf))
            reasons.append("NEAR_SHORT_STRIKE")

        if spot_vel is not None and spot_vel >= float(self.config.velocity_warn_pts_per_min):
            score += 12.0
            reasons.append("FAST_SPOT_MOVE")
            if dist_near_short <= near_buf * 1.5:
                score += 8.0
                reasons.append("FAST_MOVE_NEAR_SHORT")

        if loss_ratio_to_sl is not None:
            if loss_ratio_to_sl >= float(self.config.loss_ratio_critical):
                score += 35.0
                reasons.append("LOSS_NEAR_SL")
            elif loss_ratio_to_sl >= float(self.config.loss_ratio_warn):
                score += 16.0
                reasons.append("LOSS_BUILDING")

        if drawdown_ratio_to_tp is not None:
            if drawdown_ratio_to_tp >= float(self.config.drawdown_ratio_critical):
                score += 22.0
                reasons.append("PNL_DRAWDOWN_HIGH")
            elif drawdown_ratio_to_tp >= float(self.config.drawdown_ratio_warn):
                score += 10.0
                reasons.append("Pnl_DRAWDOWN_WATCH")

        expiry_date = _parse_date(getattr(basket, "expiry", None))
        dte: Optional[int] = None
        if expiry_date is not None:
            dte = (expiry_date - now.date()).days
            if dte <= 3 and dist_near_short <= near_buf * 1.5:
                score += 10.0
                reasons.append("LOW_DTE_GAMMA_PROXIMITY")

        if str(day_mode).upper() == "LOCKED_RED":
            reasons.append("DAY_LOCKED")

        # Phase 1 market context signals (option-chain OI/PCR + multi-timeframe trend).
        ctx = context if isinstance(context, dict) else {}
        oc_ctx = ctx.get("option_chain") if isinstance(ctx.get("option_chain"), dict) else {}
        trend_ctx = ctx.get("trend") if isinstance(ctx.get("trend"), dict) else {}
        structure_ctx = ctx.get("structure") if isinstance(ctx.get("structure"), dict) else {}
        trend_bias = str(trend_ctx.get("bias") or "NEUTRAL").upper()
        trend_bias_score = 0.0
        try:
            trend_bias_score = float(trend_ctx.get("bias_score") or 0.0)
        except Exception:
            trend_bias_score = 0.0
        pcr_bias = str(oc_ctx.get("pcr_bias") or "NEUTRAL").upper()
        pcr_total = oc_ctx.get("pcr_total")
        pcr_near = oc_ctx.get("pcr_near_atm")
        oi_build = oc_ctx.get("oi_build") if isinstance(oc_ctx.get("oi_build"), dict) else {}
        oi_build_bias = str(oi_build.get("bias") or "UNKNOWN").upper()
        call_wall_above = oc_ctx.get("call_wall_above") if isinstance(oc_ctx.get("call_wall_above"), dict) else {}
        put_wall_below = oc_ctx.get("put_wall_below") if isinstance(oc_ctx.get("put_wall_below"), dict) else {}
        call_wall_above_dist = call_wall_above.get("distance_from_spot")
        put_wall_below_dist = put_wall_below.get("distance_from_spot")
        trend_vol_regime = str(trend_ctx.get("volatility_regime") or "NORMAL").upper()
        breakout_confirmation = str(trend_ctx.get("breakout_confirmation") or "NONE").upper()
        trend_confidence = 0.0
        signal_conflict_score = 0.0
        pcr_unbalanced = False
        pcr_unbalanced_side = "NEUTRAL"
        dominant_signal_bias = "NEUTRAL"
        intraday_support = structure_ctx.get("intraday_support") if isinstance(structure_ctx.get("intraday_support"), dict) else {}
        intraday_resistance = structure_ctx.get("intraday_resistance") if isinstance(structure_ctx.get("intraday_resistance"), dict) else {}
        weekly_support = structure_ctx.get("weekly_support")
        weekly_resistance = structure_ctx.get("weekly_resistance")
        try:
            trend_confidence = float(structure_ctx.get("trend_confidence") or 0.0)
        except Exception:
            trend_confidence = 0.0
        try:
            signal_conflict_score = float(structure_ctx.get("signal_conflict_score") or 0.0)
        except Exception:
            signal_conflict_score = 0.0
        pcr_unbalanced = bool(structure_ctx.get("pcr_unbalanced", False))
        pcr_unbalanced_side = str(structure_ctx.get("pcr_unbalanced_side") or "NEUTRAL").upper()
        dominant_signal_bias = str(structure_ctx.get("dominant_signal_bias") or "NEUTRAL").upper()
        bullish_signals = 0
        bearish_signals = 0
        context_score_adjust = 0.0

        if trend_bias_score >= float(self.config.trend_bias_threshold):
            bullish_signals += 1
        elif trend_bias_score <= -float(self.config.trend_bias_threshold):
            bearish_signals += 1

        try:
            if pcr_near is not None:
                pcr_near_f = float(pcr_near)
                if pcr_near_f >= float(self.config.pcr_bullish_threshold):
                    bullish_signals += 1
                elif pcr_near_f <= float(self.config.pcr_bearish_threshold):
                    bearish_signals += 1
        except Exception:
            pass

        if oi_build_bias.startswith("BULLISH"):
            bullish_signals += 1
        elif oi_build_bias.startswith("BEARISH"):
            bearish_signals += 1

        if breakout_confirmation == "UP_CONFIRMED":
            bullish_signals += 1
        elif breakout_confirmation == "DOWN_CONFIRMED":
            bearish_signals += 1

        if risk_side == "CALL":
            if bullish_signals >= 2:
                context_score_adjust += 10.0 if outside_short else 7.0
                reasons.append("BULLISH_CONTEXT_NEAR_CALL_SHORT")
                if trend_confidence >= 0.70:
                    context_score_adjust += 3.0
                    reasons.append("HIGH_CONF_BULLISH_CONTEXT")
            elif bearish_signals >= 2:
                context_score_adjust -= 4.0
                reasons.append("BEARISH_CONTEXT_OFFSETS_CALL_RISK")
                if trend_confidence >= 0.70:
                    context_score_adjust -= 1.5
            try:
                if call_wall_above_dist is not None:
                    wall_dist = float(call_wall_above_dist)
                    if wall_dist > max(near_buf * 1.25, 250.0):
                        context_score_adjust += 4.0
                        reasons.append("CALL_RESISTANCE_WALL_FAR")
                    elif 0 <= wall_dist <= near_buf * 0.75:
                        context_score_adjust -= 2.5
                        reasons.append("CALL_WALL_NEARBY")
            except Exception:
                pass
        elif risk_side == "PUT":
            if bearish_signals >= 2:
                context_score_adjust += 10.0 if outside_short else 7.0
                reasons.append("BEARISH_CONTEXT_NEAR_PUT_SHORT")
                if trend_confidence >= 0.70:
                    context_score_adjust += 3.0
                    reasons.append("HIGH_CONF_BEARISH_CONTEXT")
            elif bullish_signals >= 2:
                context_score_adjust -= 4.0
                reasons.append("BULLISH_CONTEXT_OFFSETS_PUT_RISK")
                if trend_confidence >= 0.70:
                    context_score_adjust -= 1.5
            try:
                if put_wall_below_dist is not None:
                    wall_dist = float(put_wall_below_dist)
                    if wall_dist > max(near_buf * 1.25, 250.0):
                        context_score_adjust += 4.0
                        reasons.append("PUT_SUPPORT_WALL_FAR")
                    elif 0 <= wall_dist <= near_buf * 0.75:
                        context_score_adjust -= 2.5
                        reasons.append("PUT_WALL_NEARBY")
            except Exception:
                pass
        else:
            if bullish_signals >= 2 and bearish_signals == 0:
                reasons.append("BULLISH_MARKET_CONTEXT")
            elif bearish_signals >= 2 and bullish_signals == 0:
                reasons.append("BEARISH_MARKET_CONTEXT")

        if pcr_unbalanced and pcr_unbalanced_side in {"BULLISH", "BEARISH"}:
            if (risk_side == "CALL" and pcr_unbalanced_side == "BULLISH") or (risk_side == "PUT" and pcr_unbalanced_side == "BEARISH"):
                context_score_adjust += 3.0
                reasons.append("PCR_UNBALANCED_AGAINST_POSITION")
            elif (risk_side == "CALL" and pcr_unbalanced_side == "BEARISH") or (risk_side == "PUT" and pcr_unbalanced_side == "BULLISH"):
                context_score_adjust -= 2.0
                reasons.append("PCR_UNBALANCED_SUPPORTIVE")

        if signal_conflict_score >= 60.0:
            # Hysteresis against overreacting when signals disagree heavily.
            context_score_adjust -= 3.0
            reasons.append("SIGNAL_CONFLICT_HIGH")
        elif signal_conflict_score <= 20.0 and trend_confidence >= 0.75 and dominant_signal_bias != "NEUTRAL":
            context_score_adjust += 2.0 if risk_side in {"CALL", "PUT"} else 0.0
            reasons.append("SIGNAL_ALIGNMENT_HIGH")

        if context_score_adjust != 0.0:
            cap = max(0.0, float(self.config.context_risk_score_cap))
            context_score_adjust = max(-cap, min(cap, context_score_adjust))
            score += context_score_adjust

        # Volatility / breakout modifiers (Phase 3 chart features).
        if trend_vol_regime == "HIGH" and risk_side in {"CALL", "PUT"} and dist_near_short <= near_buf * 1.5:
            score += 5.0
            reasons.append("HIGH_VOL_NEAR_SHORT")
        elif trend_vol_regime == "LOW" and not outside_short and dist_near_short > near_buf:
            score = max(0.0, score - 2.0)
            reasons.append("LOW_VOL_RANGE_SUPPORTIVE")

        if risk_side == "CALL" and breakout_confirmation == "UP_CONFIRMED":
            score += 8.0
            reasons.append("UP_BREAKOUT_CONFIRMED_NEAR_CALL_RISK")
        elif risk_side == "PUT" and breakout_confirmation == "DOWN_CONFIRMED":
            score += 8.0
            reasons.append("DOWN_BREAKOUT_CONFIRMED_NEAR_PUT_RISK")

        if in_grace and score < float(self.config.flatten_score):
            score = max(0.0, score - 12.0)
            reasons.append("OPEN_GRACE_ACTIVE")

        score = round(float(score), 2)
        action = "HOLD"
        severity = "INFO"
        if outside_outer or score >= float(self.config.flatten_score):
            action = "FLATTEN_RECOMMEND"
            severity = "CRITICAL"
        elif score >= float(self.config.reduce_score):
            action = "REDUCE_RISK"
            severity = "WARN"
        elif score >= float(self.config.warn_score):
            action = "WATCH_TIGHT"
            severity = "INFO"

        confidence = 0.35
        confidence += min(0.35, 0.05 * len(samples))
        if spot_vel is not None:
            confidence += 0.10
        if dte is not None:
            confidence += 0.05
        if trend_confidence > 0:
            confidence += min(0.10, 0.10 * float(trend_confidence))
        if signal_conflict_score >= 60.0:
            confidence -= 0.10
        elif signal_conflict_score >= 40.0:
            confidence -= 0.05
        confidence = round(min(0.95, confidence), 3)

        features = {
            "basket_expiry": expiry_date.isoformat() if expiry_date else None,
            "dte": dte,
            "short_range_low": pe_short,
            "short_range_high": ce_short,
            "outer_range_low": pe_outer,
            "outer_range_high": ce_outer,
            "distance_to_nearest_short": round(float(dist_near_short), 2),
            "outside_short_range": outside_short,
            "outside_outer_range": outside_outer,
            "spot_window_span": None if spot_span is None else round(float(spot_span), 2),
            "spot_velocity_pts_per_min": None if spot_vel is None else round(float(spot_vel), 2),
            "pnl_peak": round(float(self._peak_pnl), 2),
            "pnl_drawdown": round(float(pnl_drawdown), 2),
            "loss_ratio_to_sl": None if loss_ratio_to_sl is None else round(float(loss_ratio_to_sl), 3),
            "drawdown_ratio_to_tp": None if drawdown_ratio_to_tp is None else round(float(drawdown_ratio_to_tp), 3),
            "samples_in_window": len(samples),
            "in_open_grace": in_grace,
            "position_risk_side": risk_side,
            "context_score_adjust": round(float(context_score_adjust), 2),
            "market_context": {
                "trend_bias": trend_bias,
                "trend_bias_score": round(float(trend_bias_score), 3),
                "pcr_bias": pcr_bias,
                "pcr_total": None if pcr_total is None else round(float(pcr_total), 3),
                "pcr_near_atm": None if pcr_near is None else round(float(pcr_near), 3),
                "oi_build_bias": oi_build_bias,
                "bullish_signals": bullish_signals,
                "bearish_signals": bearish_signals,
                "volatility_regime": trend_vol_regime,
                "breakout_confirmation": breakout_confirmation,
                "call_wall_above_distance": call_wall_above_dist,
                "put_wall_below_distance": put_wall_below_dist,
                "trend_confidence": round(float(trend_confidence), 3),
                "signal_conflict_score": round(float(signal_conflict_score), 1),
                "dominant_signal_bias": dominant_signal_bias,
                "pcr_unbalanced": bool(pcr_unbalanced),
                "pcr_unbalanced_side": pcr_unbalanced_side,
                "structure": structure_ctx,
                "trend": trend_ctx,
                "option_chain": oc_ctx,
            },
        }
        if not reasons:
            reasons = ["NORMAL_MONITOR"]
        market_bias = "NEUTRAL"
        if bullish_signals >= 2 and bullish_signals > bearish_signals:
            market_bias = "BULLISH"
        elif bearish_signals >= 2 and bearish_signals > bullish_signals:
            market_bias = "BEARISH"
        if action == "FLATTEN_RECOMMEND":
            next_course = "High risk. Plan exit/flatten immediately."
        elif action == "REDUCE_RISK":
            next_course = "Risk rising. Reduce risk or monitor closely."
        elif action == "WATCH_TIGHT":
            next_course = "Watch closely. No immediate change recommended."
        else:
            next_course = "Hold and monitor. No action needed."
        if signal_conflict_score >= 60.0:
            next_course = f"{next_course} Signals are conflicting; avoid aggressive decisions."
        elif trend_confidence >= 0.75 and dominant_signal_bias in {"BULLISH", "BEARISH"}:
            next_course = f"{next_course} Trend context confidence is high ({int(round(trend_confidence*100))}%)."
        mode_name = str(self.config.mode or "ADVISOR").upper()
        protective_lock_order = {"HOLD": 0, "WATCH_TIGHT": 1, "REDUCE_RISK": 2, "FLATTEN_RECOMMEND": 3}
        min_lock_action = str(self.config.protective_lock_min_action or "REDUCE_RISK").upper()
        entry_lock_active = (
            mode_name in {"AUTO_PROTECT", "PROTECTIVE"}
            and protective_lock_order.get(action, 0) >= protective_lock_order.get(min_lock_action, 2)
        )
        alert_severity_override = severity
        if mode_name in {"AUTO_PROTECT", "PROTECTIVE"} and action == "WATCH_TIGHT" and bool(self.config.protective_watch_escalates_alert):
            alert_severity_override = "WARN"
        plan = {
            "mode": mode_name,
            "will_auto_execute_ai": False,
            "recommended_action": action,
            "priority": severity,
            "alert_severity_override": alert_severity_override,
            "next_course": next_course,
            "position_risk_side": risk_side,
            "market_bias": market_bias,
            "entry_lock_active": bool(entry_lock_active),
            "entry_lock_reason": ("AI_PROTECTIVE_ENTRY_LOCK" if entry_lock_active else None),
            "signal_summary": {
                "bullish_signals": bullish_signals,
                "bearish_signals": bearish_signals,
                "trend_bias": trend_bias,
                "pcr_bias": pcr_bias,
                "oi_build_bias": oi_build_bias,
                "volatility_regime": trend_vol_regime,
                "breakout_confirmation": breakout_confirmation,
                "trend_confidence": round(float(trend_confidence), 3),
                "signal_conflict_score": round(float(signal_conflict_score), 1),
                "pcr_unbalanced": bool(pcr_unbalanced),
                "pcr_unbalanced_side": pcr_unbalanced_side,
                "dominant_signal_bias": dominant_signal_bias,
            },
            "support_resistance": {
                "intraday_support": intraday_support,
                "intraday_resistance": intraday_resistance,
                "weekly_support": weekly_support,
                "weekly_resistance": weekly_resistance,
            },
        }
        return {
            "score": score,
            "action": action,
            "severity": severity,
            "reasons": reasons,
            "features": features,
            "confidence": confidence,
            "market_bias": market_bias,
            "position_risk_side": risk_side,
            "market_context": features.get("market_context"),
            "plan": plan,
            "alert_severity_override": alert_severity_override,
        }

    def update_open(
        self,
        *,
        basket: Any,
        spot: float,
        pnl: float,
        tp: float,
        sl: float,
        day_mode: str,
        context: Optional[Dict[str, Any]] = None,
        when: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = when or _now_ist()
        out = self._score(
            now=now,
            basket=basket,
            spot=float(spot),
            pnl=float(pnl),
            tp=float(tp),
            sl=float(sl),
            day_mode=str(day_mode),
            context=context,
        )
        prev_action = str(self.state.action or "HOLD")
        prev_severity = str(self.state.severity or "INFO")
        prev_score = float(self.state.score or 0.0)

        feats = out.get("features") or {}
        self.state.status = "ACTIVE"
        self.state.mode = str(self.config.mode or "ADVISOR").upper()
        self.state.action = str(out.get("action") or "HOLD")
        self.state.severity = str(out.get("severity") or "INFO")
        self.state.score = float(out.get("score") or 0.0)
        self.state.confidence = float(out.get("confidence") or 0.0)
        self.state.reasons = list(out.get("reasons") or [])
        self.state.current_session_date = now.date().isoformat()
        self.state.basket_expiry = feats.get("basket_expiry")
        self.state.spot = round(float(spot), 2)
        self.state.pnl = round(float(pnl), 2)
        self.state.tp = round(float(tp), 2)
        self.state.sl = round(float(sl), 2)
        self.state.dte = feats.get("dte")
        self.state.short_range_low = feats.get("short_range_low")
        self.state.short_range_high = feats.get("short_range_high")
        self.state.outer_range_low = feats.get("outer_range_low")
        self.state.outer_range_high = feats.get("outer_range_high")
        self.state.distance_to_nearest_short = feats.get("distance_to_nearest_short")
        self.state.outside_short_range = bool(feats.get("outside_short_range", False))
        self.state.outside_outer_range = bool(feats.get("outside_outer_range", False))
        self.state.spot_window_span = feats.get("spot_window_span")
        self.state.spot_velocity_pts_per_min = feats.get("spot_velocity_pts_per_min")
        self.state.pnl_peak = feats.get("pnl_peak")
        self.state.pnl_drawdown = feats.get("pnl_drawdown")
        self.state.loss_ratio_to_sl = feats.get("loss_ratio_to_sl")
        self.state.drawdown_ratio_to_tp = feats.get("drawdown_ratio_to_tp")
        self.state.samples_in_window = int(feats.get("samples_in_window") or 0)
        self.state.day_mode = str(day_mode)
        self.state.position_risk_side = str(out.get("position_risk_side") or feats.get("position_risk_side") or "")
        self.state.market_bias = str(out.get("market_bias") or "NEUTRAL")
        self.state.market_context = out.get("market_context") if isinstance(out.get("market_context"), dict) else {}
        self.state.plan = out.get("plan") if isinstance(out.get("plan"), dict) else {}

        action_changed = (self.state.action != prev_action) or (self.state.severity != prev_severity)
        material_score_change = abs(self.state.score - prev_score) >= 15.0
        if action_changed:
            self.state.last_action_changed_at = now.isoformat(timespec="seconds")

        self._persist(now)
        self._emit_change_event(now, changed=(action_changed or material_score_change))

        snap = self.snapshot()
        snap["action_changed"] = action_changed
        snap["material_score_change"] = material_score_change
        return snap
