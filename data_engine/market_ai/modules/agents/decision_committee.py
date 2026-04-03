from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _now_ist() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


def _clamp01(value: Any, fallback: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return fallback


def _safe_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
    if value in (None, "", "-", "--"):
        return fallback
    try:
        return float(value)
    except Exception:
        return fallback


@dataclass
class CommitteeComponent:
    name: str
    stance: str = "NEUTRAL"
    score: float = 0.0
    confidence: float = 0.0
    veto: bool = False
    reasons: List[str] = field(default_factory=list)


@dataclass
class DecisionCommitteeState:
    status: str = "IDLE"  # IDLE|ACTIVE|DISABLED
    focus: str = "INTRADAY"
    verdict: str = "IDLE"  # IDLE|WAIT|READY|VETO
    consensus_bias: str = "NEUTRAL"
    ensemble_confidence: float = 0.0
    advisor_signal: str = "NO_TRADE"
    advisor_strategy: Optional[str] = None
    expiry: Optional[str] = None
    spot: Optional[float] = None
    has_open_bkm: bool = False
    current_session_date: str = ""
    updated_at: str = ""
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["components"] = payload.get("components") if isinstance(payload.get("components"), dict) else {}
        payload["reasons"] = payload.get("reasons") if isinstance(payload.get("reasons"), list) else []
        return payload


class DecisionCommittee:
    def __init__(
        self,
        *,
        status_path: Path,
        history_path: Optional[Path] = None,
        outcomes_path: Optional[Path] = None,
        logger: Any = None,
    ) -> None:
        self.status_path = status_path
        self.history_path = history_path
        self.outcomes_path = outcomes_path
        self.log = logger
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        if self.history_path is not None:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.history_path.exists():
                try:
                    self.history_path.write_text("")
                except Exception:
                    pass
        if self.outcomes_path is not None:
            self.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.outcomes_path.exists():
                try:
                    self.outcomes_path.write_text("")
                except Exception:
                    pass
        self.state = self._load_state()

    def _log(self, level: str, msg: str, *args: Any) -> None:
        if self.log and hasattr(self.log, level):
            getattr(self.log, level)(msg, *args)

    def _load_state(self) -> DecisionCommitteeState:
        if not self.status_path.exists():
            st = DecisionCommitteeState()
            st.current_session_date = _now_ist().date().isoformat()
            return st
        try:
            payload = json.loads(self.status_path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("invalid decision committee payload")
            st = DecisionCommitteeState()
            for key in DecisionCommitteeState.__dataclass_fields__.keys():
                if key in payload:
                    setattr(st, key, payload[key])
            if not isinstance(st.components, dict):
                st.components = {}
            if not isinstance(st.reasons, list):
                st.reasons = []
            return st
        except Exception:
            st = DecisionCommitteeState()
            st.current_session_date = _now_ist().date().isoformat()
            return st

    def _persist(self, when: Optional[datetime] = None) -> None:
        ts = when or _now_ist()
        self.state.updated_at = ts.isoformat(timespec="seconds")
        if not self.state.current_session_date:
            self.state.current_session_date = ts.date().isoformat()
        try:
            self.status_path.write_text(json.dumps(self.state.as_dict(), indent=2, default=str))
        except Exception:
            self._log("exception", "Failed to persist decision committee status")

    def _append_history(self, row: Dict[str, Any]) -> None:
        if self.history_path is None:
            return
        try:
            with self.history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except Exception:
            self._log("exception", "Failed to append decision committee history")

    def _append_outcome(self, row: Dict[str, Any]) -> None:
        if self.outcomes_path is None:
            return
        try:
            with self.outcomes_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except Exception:
            self._log("exception", "Failed to append decision committee outcome")

    def _history_signature(self, state: DecisionCommitteeState) -> str:
        payload = {
            "status": state.status,
            "verdict": state.verdict,
            "consensus_bias": state.consensus_bias,
            "advisor_signal": state.advisor_signal,
            "advisor_strategy": state.advisor_strategy,
            "expiry": state.expiry,
            "has_open_bkm": state.has_open_bkm,
            "reasons": state.reasons,
            "components": state.components,
        }
        try:
            return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            return str(payload)

    def _persist_and_log_if_changed(self, prior_state: Optional[DecisionCommitteeState], when: Optional[datetime] = None) -> None:
        ts = when or _now_ist()
        self._persist(ts)
        prior_sig = self._history_signature(prior_state) if prior_state is not None else None
        next_sig = self._history_signature(self.state)
        if prior_sig == next_sig:
            return
        self._append_history(
            {
                "timestamp": ts.isoformat(timespec="seconds"),
                "status": self.state.status,
                "focus": self.state.focus,
                "verdict": self.state.verdict,
                "consensus_bias": self.state.consensus_bias,
                "ensemble_confidence": self.state.ensemble_confidence,
                "advisor_signal": self.state.advisor_signal,
                "advisor_strategy": self.state.advisor_strategy,
                "expiry": self.state.expiry,
                "spot": self.state.spot,
                "has_open_bkm": self.state.has_open_bkm,
                "reasons": list(self.state.reasons or []),
                "components": dict(self.state.components or {}),
            }
        )

    def snapshot(self) -> Dict[str, Any]:
        return self.state.as_dict()

    def record_outcome(self, *, trade_row: Dict[str, Any], now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        if self.outcomes_path is None or not isinstance(trade_row, dict) or not trade_row:
            return None
        ts = now or _now_ist()
        entry_features = trade_row.get("entry_features") if isinstance(trade_row.get("entry_features"), dict) else {}
        committee_snapshot = (
            entry_features.get("committee_snapshot")
            if isinstance(entry_features.get("committee_snapshot"), dict)
            else {}
        )
        outcome = {
            "timestamp": ts.isoformat(timespec="seconds"),
            "position_id": trade_row.get("position_id"),
            "trade_mode": trade_row.get("trade_mode"),
            "strategy_type": trade_row.get("strategy_type"),
            "strategy_label": trade_row.get("strategy_label"),
            "expiry": trade_row.get("expiry"),
            "opened_at": trade_row.get("opened_at"),
            "closed_at": trade_row.get("closed_at"),
            "current_session_date": trade_row.get("current_session_date"),
            "hold_minutes": _safe_float(trade_row.get("hold_minutes"), None),
            "pnl_rs": round(float(_safe_float(trade_row.get("pnl_rs"), 0.0) or 0.0), 2),
            "result": str(trade_row.get("result") or "UNKNOWN").upper(),
            "exit_reason": str(trade_row.get("reason") or "UNKNOWN").upper(),
            "entry_market_bias": str(entry_features.get("market_bias") or "NEUTRAL").upper(),
            "entry_price_action_bias": str(entry_features.get("price_action_bias") or "NEUTRAL").upper(),
            "entry_retest_status": str(entry_features.get("retest_status") or "NONE").upper(),
            "committee_status_at_entry": committee_snapshot.get("status"),
            "committee_verdict_at_entry": committee_snapshot.get("verdict"),
            "committee_bias_at_entry": committee_snapshot.get("consensus_bias"),
            "committee_confidence_at_entry": _safe_float(committee_snapshot.get("ensemble_confidence"), None),
            "committee_signal_at_entry": committee_snapshot.get("advisor_signal"),
            "committee_strategy_at_entry": committee_snapshot.get("advisor_strategy"),
            "committee_reasons_at_entry": (
                list(committee_snapshot.get("reasons") or [])
                if isinstance(committee_snapshot.get("reasons"), list)
                else []
            ),
            "committee_components_at_entry": (
                dict(committee_snapshot.get("components") or {})
                if isinstance(committee_snapshot.get("components"), dict)
                else {}
            ),
            "entry_features": dict(entry_features or {}),
        }
        self._append_outcome(outcome)
        return outcome

    def set_inactive(
        self,
        *,
        now: Optional[datetime] = None,
        status: str = "IDLE",
        reason: str,
        has_open_bkm: bool = False,
        expiry: Optional[str] = None,
    ) -> Dict[str, Any]:
        ts = now or _now_ist()
        prior_state = DecisionCommitteeState(**self.state.as_dict())
        self.state = DecisionCommitteeState(
            status=str(status or "IDLE").upper(),
            focus="INTRADAY",
            verdict="IDLE",
            consensus_bias="NEUTRAL",
            ensemble_confidence=0.0,
            advisor_signal="NO_TRADE",
            advisor_strategy=None,
            expiry=str(expiry or "").strip() or None,
            spot=None,
            has_open_bkm=bool(has_open_bkm),
            current_session_date=ts.date().isoformat(),
            updated_at=ts.isoformat(timespec="seconds"),
            components={},
            reasons=[str(reason or "IDLE")],
        )
        self._persist_and_log_if_changed(prior_state, ts)
        return self.snapshot()

    def evaluate_intraday(
        self,
        *,
        now: Optional[datetime],
        signal_payload: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        has_open_bkm: bool,
        allow_parallel_with_bkm: bool,
    ) -> Dict[str, Any]:
        ts = now or _now_ist()
        prior_state = DecisionCommitteeState(**self.state.as_dict())
        signal_payload = signal_payload if isinstance(signal_payload, dict) else {}
        context = context if isinstance(context, dict) else {}
        structure = context.get("structure") if isinstance(context.get("structure"), dict) else {}
        option_chain = context.get("option_chain") if isinstance(context.get("option_chain"), dict) else {}
        trend = context.get("trend") if isinstance(context.get("trend"), dict) else {}
        recommendation = signal_payload.get("recommendation") if isinstance(signal_payload.get("recommendation"), dict) else {}

        market_structure = self._score_market_structure(structure=structure, trend=trend)
        chain_component = self._score_option_chain(option_chain=option_chain, structure=structure)
        construction = self._score_trade_construction(signal_payload=signal_payload, recommendation=recommendation)
        critic = self._score_risk_critic(
            signal_payload=signal_payload,
            structure=structure,
            has_open_bkm=has_open_bkm,
            allow_parallel_with_bkm=allow_parallel_with_bkm,
        )

        components = {
            "market_structure": asdict(market_structure),
            "option_chain": asdict(chain_component),
            "trade_construction": asdict(construction),
            "risk_critic": asdict(critic),
        }
        reasons: List[str] = []
        for comp in (market_structure, chain_component, construction, critic):
            reasons.extend([r for r in comp.reasons if r])

        consensus_bias = self._consensus_bias(market_structure.stance, chain_component.stance, construction.stance)
        confidence = self._ensemble_confidence(market_structure, chain_component, construction, critic)
        advisor_signal = str(signal_payload.get("signal") or "NO_TRADE").upper()
        verdict = "WAIT"
        if critic.veto:
            verdict = "VETO"
        elif advisor_signal == "ENTER_NOW" and confidence >= 0.68:
            verdict = "READY"
        elif advisor_signal == "NO_TRADE":
            verdict = "WAIT"

        self.state = DecisionCommitteeState(
            status="ACTIVE",
            focus="INTRADAY",
            verdict=verdict,
            consensus_bias=consensus_bias,
            ensemble_confidence=round(float(confidence), 3),
            advisor_signal=advisor_signal,
            advisor_strategy=str(signal_payload.get("strategy") or recommendation.get("strategy_type") or "").strip() or None,
            expiry=str(signal_payload.get("expiry") or recommendation.get("expiry") or "").strip() or None,
            spot=_safe_float(signal_payload.get("spot"), None),
            has_open_bkm=bool(has_open_bkm),
            current_session_date=ts.date().isoformat(),
            updated_at=ts.isoformat(timespec="seconds"),
            components=components,
            reasons=reasons[:12],
        )
        self._persist_and_log_if_changed(prior_state, ts)
        return self.snapshot()

    def _score_market_structure(self, *, structure: Dict[str, Any], trend: Dict[str, Any]) -> CommitteeComponent:
        stance = str(structure.get("dominant_signal_bias") or trend.get("bias") or "NEUTRAL").upper()
        conf = _clamp01(structure.get("trend_confidence"), 0.0)
        breakout = str(trend.get("breakout_confirmation") or structure.get("breakout_confirmation") or "NONE").upper()
        pa_conf = str(structure.get("price_action_confirmation") or "NONE").upper()
        retest = str(structure.get("retest_status") or "NONE").upper()
        score = conf * 0.6
        reasons: List[str] = []
        if stance in {"BULLISH", "BEARISH"}:
            reasons.append(f"STRUCTURE_{stance}")
            score += 0.15
        if breakout != "NONE":
            reasons.append(f"BREAKOUT_{breakout}")
            score += 0.10
        if pa_conf != "NONE":
            reasons.append(f"PRICE_ACTION_{pa_conf}")
            score += 0.10
        if retest not in {"NONE", "RETEST_FAILED"}:
            reasons.append(f"RETEST_{retest}")
            score += 0.10
        return CommitteeComponent(
            name="market_structure",
            stance=stance,
            score=round(min(1.0, score), 3),
            confidence=round(conf, 3),
            veto=False,
            reasons=reasons,
        )

    def _score_option_chain(self, *, option_chain: Dict[str, Any], structure: Dict[str, Any]) -> CommitteeComponent:
        pcr_bias = str(option_chain.get("pcr_bias") or "NEUTRAL").upper()
        oi_bias = str((option_chain.get("oi_build") or {}).get("bias") or structure.get("oi_build_bias") or "UNKNOWN").upper()
        unbalanced = bool(structure.get("pcr_unbalanced", False))
        unbalanced_side = str(structure.get("pcr_unbalanced_side") or "NEUTRAL").upper()
        stance = "NEUTRAL"
        reasons: List[str] = []
        score = 0.15
        if pcr_bias in {"BULLISH", "BEARISH"}:
            stance = pcr_bias
            reasons.append(f"PCR_{pcr_bias}")
            score += 0.20
        if "BULLISH" in oi_bias:
            stance = "BULLISH" if stance == "NEUTRAL" else stance
            reasons.append("OI_BUILD_BULLISH")
            score += 0.20
        elif "BEARISH" in oi_bias:
            stance = "BEARISH" if stance == "NEUTRAL" else stance
            reasons.append("OI_BUILD_BEARISH")
            score += 0.20
        if unbalanced and unbalanced_side in {"BULLISH", "BEARISH"}:
            reasons.append(f"PCR_UNBALANCED_{unbalanced_side}")
            if stance == unbalanced_side or stance == "NEUTRAL":
                stance = unbalanced_side
                score += 0.10
            else:
                score -= 0.12
        return CommitteeComponent(
            name="option_chain",
            stance=stance,
            score=round(max(0.0, min(1.0, score)), 3),
            confidence=round(max(0.25, min(1.0, score)), 3),
            veto=False,
            reasons=reasons,
        )

    def _score_trade_construction(self, *, signal_payload: Dict[str, Any], recommendation: Dict[str, Any]) -> CommitteeComponent:
        signal = str(signal_payload.get("signal") or "NO_TRADE").upper()
        strategy = str(signal_payload.get("strategy") or recommendation.get("strategy_type") or "").upper()
        est_credit = _safe_float(recommendation.get("est_credit_rs_per_set"), 0.0) or 0.0
        score = 0.0
        reasons: List[str] = []
        stance = "NEUTRAL"
        if strategy == "PUT_CREDIT_SPREAD":
            stance = "BULLISH"
        elif strategy == "CALL_CREDIT_SPREAD":
            stance = "BEARISH"
        elif strategy == "IRON_CONDOR":
            stance = "NEUTRAL"
        if signal == "ENTER_NOW":
            reasons.append("ADVISOR_ENTER_NOW")
            score += 0.45
        elif signal == "WAIT":
            reasons.append("ADVISOR_WAIT")
            score += 0.20
        else:
            reasons.append("ADVISOR_NO_TRADE")
        if strategy:
            reasons.append(f"STRATEGY_{strategy}")
            score += 0.15
        if est_credit >= 500.0:
            reasons.append("CREDIT_OK")
            score += 0.20
        elif est_credit > 0:
            reasons.append("CREDIT_LIGHT")
            score += 0.08
        return CommitteeComponent(
            name="trade_construction",
            stance=stance,
            score=round(max(0.0, min(1.0, score)), 3),
            confidence=round(max(0.0, min(1.0, score)), 3),
            veto=False,
            reasons=reasons,
        )

    def _score_risk_critic(
        self,
        *,
        signal_payload: Dict[str, Any],
        structure: Dict[str, Any],
        has_open_bkm: bool,
        allow_parallel_with_bkm: bool,
    ) -> CommitteeComponent:
        conflict = _safe_float(signal_payload.get("signal_conflict_score"), 0.0) or 0.0
        vol = str(signal_payload.get("volatility_regime") or structure.get("volatility_regime") or "UNKNOWN").upper()
        pa_conf = str(signal_payload.get("price_action_confirmation") or structure.get("price_action_confirmation") or "NONE").upper()
        reasons: List[str] = []
        veto = False
        score = 0.7
        if has_open_bkm and not allow_parallel_with_bkm:
            veto = True
            reasons.append("OPEN_BKM_BLOCK")
            score = 0.0
        elif conflict >= 60.0:
            veto = True
            reasons.append("CONFLICT_TOO_HIGH")
            score = 0.0
        else:
            if conflict >= 45.0:
                reasons.append("CONFLICT_ELEVATED")
                score -= 0.25
            if vol == "HIGH":
                reasons.append("HIGH_VOL")
                score -= 0.10
            if pa_conf == "NONE":
                reasons.append("PRICE_ACTION_UNCONFIRMED")
                score -= 0.10
        return CommitteeComponent(
            name="risk_critic",
            stance="NEUTRAL",
            score=round(max(0.0, min(1.0, score)), 3),
            confidence=round(max(0.0, min(1.0, score)), 3),
            veto=veto,
            reasons=reasons,
        )

    def _consensus_bias(self, *stances: str) -> str:
        counts: Dict[str, int] = {}
        for stance in stances:
            s = str(stance or "NEUTRAL").upper()
            if s not in {"BULLISH", "BEARISH", "NEUTRAL"}:
                s = "NEUTRAL"
            counts[s] = counts.get(s, 0) + 1
        bullish = counts.get("BULLISH", 0)
        bearish = counts.get("BEARISH", 0)
        if bullish > bearish:
            return "BULLISH"
        if bearish > bullish:
            return "BEARISH"
        return "NEUTRAL"

    def _ensemble_confidence(
        self,
        market_structure: CommitteeComponent,
        option_chain: CommitteeComponent,
        construction: CommitteeComponent,
        critic: CommitteeComponent,
    ) -> float:
        weighted = (
            market_structure.score * 0.35
            + option_chain.score * 0.30
            + construction.score * 0.20
            + critic.score * 0.15
        )
        return max(0.0, min(1.0, weighted))
