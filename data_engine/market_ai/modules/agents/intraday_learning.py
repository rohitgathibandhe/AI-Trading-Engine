from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _now_ist() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_label(value: Any, default: str = "UNKNOWN") -> str:
    raw = str(value or "").strip().upper()
    return raw or default


def _result_label(pnl: float) -> str:
    if pnl > 0:
        return "WIN"
    if pnl < 0:
        return "LOSS"
    return "FLAT"


@dataclass
class IntradayLearningConfig:
    enabled: bool = True
    min_bucket_samples: int = 3
    negative_bucket_win_rate_max: float = 0.40
    negative_bucket_avg_pnl_max: float = -50.0
    positive_bucket_win_rate_min: float = 0.58
    positive_bucket_avg_pnl_min: float = 20.0
    confidence_penalty_per_bucket: float = 0.06
    confidence_bonus_per_bucket: float = 0.03
    confidence_adjust_cap: float = 0.18
    entry_block_negative_bucket_hits: int = 3
    review_window: int = 300

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "IntradayLearningConfig":
        return cls(
            enabled=bool(settings.get("intraday_ai_learning_enabled", True))
            and bool(settings.get("intraday_ai_enabled", True)),
            min_bucket_samples=max(1, int(float(settings.get("intraday_ai_learning_min_bucket_samples", 3) or 3))),
            negative_bucket_win_rate_max=max(
                0.0, min(1.0, float(settings.get("intraday_ai_learning_negative_bucket_win_rate_max", 0.40) or 0.40))
            ),
            negative_bucket_avg_pnl_max=float(
                settings.get("intraday_ai_learning_negative_bucket_avg_pnl_max", -50.0) or -50.0
            ),
            positive_bucket_win_rate_min=max(
                0.0, min(1.0, float(settings.get("intraday_ai_learning_positive_bucket_win_rate_min", 0.58) or 0.58))
            ),
            positive_bucket_avg_pnl_min=float(
                settings.get("intraday_ai_learning_positive_bucket_avg_pnl_min", 20.0) or 20.0
            ),
            confidence_penalty_per_bucket=max(
                0.0, float(settings.get("intraday_ai_learning_confidence_penalty_per_bucket", 0.06) or 0.06)
            ),
            confidence_bonus_per_bucket=max(
                0.0, float(settings.get("intraday_ai_learning_confidence_bonus_per_bucket", 0.03) or 0.03)
            ),
            confidence_adjust_cap=max(
                0.01, float(settings.get("intraday_ai_learning_confidence_adjust_cap", 0.18) or 0.18)
            ),
            entry_block_negative_bucket_hits=max(
                1, int(float(settings.get("intraday_ai_learning_entry_block_negative_bucket_hits", 3) or 3))
            ),
            review_window=max(30, int(float(settings.get("intraday_ai_learning_review_window", 300) or 300))),
        )


@dataclass
class IntradayLearningState:
    status: str = "IDLE"
    total_outcomes: int = 0
    wins: int = 0
    losses: int = 0
    flats: int = 0
    realized_pnl_rs: float = 0.0
    last_outcome_at: Optional[str] = None
    last_assessment: Optional[Dict[str, Any]] = None
    top_negative_buckets: List[Dict[str, Any]] = None  # type: ignore[assignment]
    top_positive_buckets: List[Dict[str, Any]] = None  # type: ignore[assignment]
    updated_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["last_assessment"] = dict(payload.get("last_assessment") or {})
        payload["top_negative_buckets"] = list(payload.get("top_negative_buckets") or [])
        payload["top_positive_buckets"] = list(payload.get("top_positive_buckets") or [])
        return payload


class IntradayLearningManager:
    _BUCKET_KEYS: Tuple[str, ...] = (
        "strategy_type",
        "committee_verdict",
        "committee_bias",
        "market_bias",
        "price_action_bias",
        "retest_status",
        "volatility_regime",
        "breakout_confirmation",
        "pcr_bias",
        "oi_build_bias",
        "strategy_regime_combo",
        "strategy_bias_combo",
    )

    def __init__(self, *, config: IntradayLearningConfig, status_path: Path, outcomes_path: Path, logger: Any = None) -> None:
        self.config = config
        self.status_path = status_path
        self.outcomes_path = outcomes_path
        self.log = logger
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.outcomes_path.exists():
            try:
                self.outcomes_path.write_text("")
            except Exception:
                pass
        self.state = self._load_state()
        self.refresh_summary()

    def _log(self, level: str, msg: str, *args: Any) -> None:
        if self.log and hasattr(self.log, level):
            getattr(self.log, level)(msg, *args)

    def _load_state(self) -> IntradayLearningState:
        if not self.status_path.exists():
            return IntradayLearningState()
        try:
            payload = json.loads(self.status_path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("invalid intraday learning payload")
            state = IntradayLearningState(
                **{key: payload.get(key) for key in IntradayLearningState.__dataclass_fields__.keys()}  # type: ignore[arg-type]
            )
            state.last_assessment = dict(state.last_assessment or {})
            state.top_negative_buckets = list(state.top_negative_buckets or [])
            state.top_positive_buckets = list(state.top_positive_buckets or [])
            return state
        except Exception:
            return IntradayLearningState()

    def _persist(self, when: Optional[datetime] = None) -> None:
        ts = when or _now_ist()
        self.state.updated_at = ts.isoformat(timespec="seconds")
        try:
            self.status_path.write_text(json.dumps(self.state.as_dict(), indent=2))
        except Exception:
            self._log("exception", "Failed to persist intraday learning status")

    def snapshot(self) -> Dict[str, Any]:
        return self.state.as_dict()

    def _load_outcomes(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not self.outcomes_path.exists():
            return rows
        try:
            with self.outcomes_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        rows.append(payload)
        except Exception:
            return []
        return rows

    def _normalize_outcome(self, row: Dict[str, Any]) -> Dict[str, Any]:
        entry_features = row.get("entry_features") if isinstance(row.get("entry_features"), dict) else {}
        strategy_type = _normalize_label(row.get("strategy_type") or row.get("committee_strategy_at_entry"))
        committee_verdict = _normalize_label(row.get("committee_verdict_at_entry"))
        committee_bias = _normalize_label(row.get("committee_bias_at_entry"))
        market_bias = _normalize_label(row.get("entry_market_bias") or entry_features.get("market_bias"))
        price_action_bias = _normalize_label(row.get("entry_price_action_bias") or entry_features.get("price_action_bias"))
        retest_status = _normalize_label(row.get("entry_retest_status") or entry_features.get("retest_status"))
        volatility_regime = _normalize_label(entry_features.get("volatility_regime"))
        breakout_confirmation = _normalize_label(entry_features.get("breakout_confirmation"))
        pcr_bias = _normalize_label(entry_features.get("pcr_bias"))
        oi_build_bias = _normalize_label(entry_features.get("oi_build_bias"))
        pnl = float(_safe_float(row.get("pnl_rs"), 0.0) or 0.0)
        return {
            "strategy_type": strategy_type,
            "committee_verdict": committee_verdict,
            "committee_bias": committee_bias,
            "market_bias": market_bias,
            "price_action_bias": price_action_bias,
            "retest_status": retest_status,
            "volatility_regime": volatility_regime,
            "breakout_confirmation": breakout_confirmation,
            "pcr_bias": pcr_bias,
            "oi_build_bias": oi_build_bias,
            "strategy_regime_combo": f"{strategy_type}|{volatility_regime}",
            "strategy_bias_combo": f"{strategy_type}|{committee_bias if committee_bias != 'UNKNOWN' else market_bias}",
            "pnl_rs": round(pnl, 2),
            "result": _normalize_label(row.get("result") or _result_label(pnl)),
            "closed_at": row.get("closed_at") or row.get("timestamp"),
        }

    def _stats_for_bucket(self, rows: Iterable[Dict[str, Any]], bucket_name: str, label: str) -> Dict[str, Any]:
        scoped = [row for row in rows if _normalize_label(row.get(bucket_name)) == label]
        count = len(scoped)
        if count == 0:
            return {"count": 0, "wins": 0, "losses": 0, "flats": 0, "avg_pnl_rs": 0.0, "win_rate": None}
        wins = sum(1 for row in scoped if str(row.get("result") or "").upper() == "WIN")
        losses = sum(1 for row in scoped if str(row.get("result") or "").upper() == "LOSS")
        flats = sum(1 for row in scoped if str(row.get("result") or "").upper() == "FLAT")
        pnl = sum(float(_safe_float(row.get("pnl_rs"), 0.0) or 0.0) for row in scoped)
        return {
            "count": count,
            "wins": wins,
            "losses": losses,
            "flats": flats,
            "avg_pnl_rs": round(pnl / max(1, count), 2),
            "win_rate": round(wins / count, 4),
        }

    def refresh_summary(self) -> Dict[str, Any]:
        rows = [self._normalize_outcome(row) for row in self._load_outcomes()][-self.config.review_window :]
        wins = sum(1 for row in rows if str(row.get("result") or "").upper() == "WIN")
        losses = sum(1 for row in rows if str(row.get("result") or "").upper() == "LOSS")
        flats = sum(1 for row in rows if str(row.get("result") or "").upper() == "FLAT")
        pnl = round(sum(float(_safe_float(row.get("pnl_rs"), 0.0) or 0.0) for row in rows), 2)
        negatives: List[Dict[str, Any]] = []
        positives: List[Dict[str, Any]] = []
        for bucket_name in self._BUCKET_KEYS:
            labels = sorted(
                {
                    _normalize_label(row.get(bucket_name))
                    for row in rows
                    if _normalize_label(row.get(bucket_name)) not in {"UNKNOWN", "NONE"}
                }
            )
            for label in labels:
                stats = self._stats_for_bucket(rows, bucket_name, label)
                if stats["count"] < self.config.min_bucket_samples:
                    continue
                payload = {"bucket": bucket_name, "label": label, **stats}
                if (
                    stats["avg_pnl_rs"] <= self.config.negative_bucket_avg_pnl_max
                    and (stats["win_rate"] or 0.0) <= self.config.negative_bucket_win_rate_max
                ):
                    negatives.append(payload)
                if (
                    stats["avg_pnl_rs"] >= self.config.positive_bucket_avg_pnl_min
                    and (stats["win_rate"] or 0.0) >= self.config.positive_bucket_win_rate_min
                ):
                    positives.append(payload)
        negatives.sort(key=lambda item: (float(item["avg_pnl_rs"]), float(item.get("win_rate") or 0.0), -int(item["count"])))
        positives.sort(key=lambda item: (-float(item["avg_pnl_rs"]), -float(item.get("win_rate") or 0.0), -int(item["count"])))

        self.state.status = "ACTIVE" if self.config.enabled else "DISABLED"
        self.state.total_outcomes = len(rows)
        self.state.wins = wins
        self.state.losses = losses
        self.state.flats = flats
        self.state.realized_pnl_rs = pnl
        self.state.last_outcome_at = str(rows[-1].get("closed_at") or "") if rows else None
        self.state.top_negative_buckets = negatives[:10]
        self.state.top_positive_buckets = positives[:10]
        self._persist()
        return self.snapshot()

    def ingest_outcome(self, outcome_row: Dict[str, Any], *, now: Optional[datetime] = None) -> Dict[str, Any]:
        del outcome_row  # committee already persists the shared outcomes file
        self.refresh_summary()
        self._persist(now or _now_ist())
        return self.snapshot()

    def _matched_buckets(self, snapshot: Dict[str, Any], bucket_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        matched: List[Dict[str, Any]] = []
        for row in bucket_rows or []:
            bucket_name = str(row.get("bucket") or "")
            label = str(row.get("label") or "")
            if bucket_name and _normalize_label(snapshot.get(bucket_name)) == label:
                matched.append(dict(row))
        return matched

    def assess_signal(
        self,
        *,
        signal_payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        consensus_bias: Optional[str] = None,
        base_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            assessment = {
                "status": "DISABLED",
                "block_entry": False,
                "confidence_adjust": 0.0,
                "component": {
                    "name": "adaptive_learning",
                    "stance": "NEUTRAL",
                    "score": 0.5,
                    "confidence": 0.0,
                    "veto": False,
                    "reasons": ["LEARNING_DISABLED"],
                },
                "matched_negative_buckets": [],
                "matched_positive_buckets": [],
                "reasons": ["LEARNING_DISABLED"],
            }
            self.state.last_assessment = assessment
            self._persist()
            return assessment

        signal_payload = signal_payload if isinstance(signal_payload, dict) else {}
        context = context if isinstance(context, dict) else {}
        recommendation = signal_payload.get("recommendation") if isinstance(signal_payload.get("recommendation"), dict) else {}
        structure = context.get("structure") if isinstance(context.get("structure"), dict) else {}
        option_chain = context.get("option_chain") if isinstance(context.get("option_chain"), dict) else {}
        snapshot = {
            "strategy_type": _normalize_label(signal_payload.get("strategy") or recommendation.get("strategy_type")),
            "committee_verdict": _normalize_label("READY" if str(signal_payload.get("signal") or "").upper() == "ENTER_NOW" else "WAIT"),
            "committee_bias": _normalize_label(consensus_bias),
            "market_bias": _normalize_label(signal_payload.get("market_bias")),
            "price_action_bias": _normalize_label(signal_payload.get("price_action_bias") or structure.get("price_action_bias")),
            "retest_status": _normalize_label(signal_payload.get("retest_status") or structure.get("retest_status")),
            "volatility_regime": _normalize_label(signal_payload.get("volatility_regime") or structure.get("volatility_regime")),
            "breakout_confirmation": _normalize_label(signal_payload.get("breakout_confirmation")),
            "pcr_bias": _normalize_label(signal_payload.get("pcr_bias") or option_chain.get("pcr_bias")),
            "oi_build_bias": _normalize_label(signal_payload.get("oi_build_bias") or ((option_chain.get("oi_build") or {}) if isinstance(option_chain.get("oi_build"), dict) else {}).get("bias")),
        }
        snapshot["strategy_regime_combo"] = f"{snapshot['strategy_type']}|{snapshot['volatility_regime']}"
        snapshot["strategy_bias_combo"] = f"{snapshot['strategy_type']}|{snapshot['committee_bias'] if snapshot['committee_bias'] != 'UNKNOWN' else snapshot['market_bias']}"

        negatives = self._matched_buckets(snapshot, list(self.state.top_negative_buckets or []))
        positives = self._matched_buckets(snapshot, list(self.state.top_positive_buckets or []))
        confidence_adjust = (
            -len(negatives) * float(self.config.confidence_penalty_per_bucket)
            + len(positives) * float(self.config.confidence_bonus_per_bucket)
        )
        cap = float(self.config.confidence_adjust_cap)
        confidence_adjust = max(-cap, min(cap, confidence_adjust))

        severe_negative = any(
            str(row.get("bucket") or "") in {"strategy_type", "strategy_regime_combo", "strategy_bias_combo"}
            and int(row.get("count") or 0) >= int(self.config.min_bucket_samples)
            and float(row.get("avg_pnl_rs") or 0.0) <= float(self.config.negative_bucket_avg_pnl_max)
            and float(row.get("win_rate") or 0.0) <= float(self.config.negative_bucket_win_rate_max)
            for row in negatives
        )
        block_entry = (
            str(signal_payload.get("signal") or "").upper() == "ENTER_NOW"
            and (
                len(negatives) >= int(self.config.entry_block_negative_bucket_hits)
                or severe_negative
            )
        )
        stance = "NEUTRAL"
        if len(positives) > len(negatives):
            stance = "SUPPORTIVE"
        elif negatives:
            stance = "CAUTION"

        reasons: List[str] = []
        for row in negatives[:4]:
            reasons.append(f"LEARNING_NEGATIVE_{row['bucket'].upper()}_{row['label']}")
        for row in positives[:3]:
            reasons.append(f"LEARNING_POSITIVE_{row['bucket'].upper()}_{row['label']}")
        if block_entry:
            reasons.append("LEARNING_BLOCK_ENTRY")

        base = float(base_confidence or 0.5)
        component_score = max(0.0, min(1.0, base + confidence_adjust))
        assessment = {
            "status": "ACTIVE",
            "block_entry": block_entry,
            "confidence_adjust": round(float(confidence_adjust), 3),
            "matched_negative_buckets": negatives[:6],
            "matched_positive_buckets": positives[:6],
            "reasons": reasons[:10] or ["LEARNING_NEUTRAL"],
            "component": {
                "name": "adaptive_learning",
                "stance": stance,
                "score": round(component_score, 3),
                "confidence": round(min(1.0, abs(confidence_adjust) + 0.25 if (negatives or positives) else 0.2), 3),
                "veto": bool(block_entry),
                "reasons": reasons[:10] or ["LEARNING_NEUTRAL"],
            },
        }
        self.state.last_assessment = {
            "strategy_type": snapshot["strategy_type"],
            "market_bias": snapshot["market_bias"],
            "committee_bias": snapshot["committee_bias"],
            "confidence_adjust": assessment["confidence_adjust"],
            "block_entry": assessment["block_entry"],
            "negative_hits": len(negatives),
            "positive_hits": len(positives),
            "reasons": list(assessment["reasons"]),
        }
        self._persist()
        return assessment
