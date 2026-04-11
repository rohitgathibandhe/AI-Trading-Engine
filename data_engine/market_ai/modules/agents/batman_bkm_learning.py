from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _now_ist() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


def _parse_dt(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_label(value: Any, default: str = "UNKNOWN") -> str:
    raw = str(value or "").strip().upper()
    return raw or default


def _result_from_pnl(pnl: float) -> str:
    if pnl > 0:
        return "WIN"
    if pnl < 0:
        return "LOSS"
    return "FLAT"


def _signal_conflict_bucket(value: Any) -> str:
    val = _safe_float(value, None)
    if val is None:
        return "UNKNOWN"
    if val >= 60.0:
        return "HIGH"
    if val >= 35.0:
        return "MEDIUM"
    return "LOW"


def _trend_confidence_bucket(value: Any) -> str:
    val = _safe_float(value, None)
    if val is None:
        return "UNKNOWN"
    if val >= 0.80:
        return "HIGH"
    if val >= 0.60:
        return "MEDIUM"
    return "LOW"


@dataclass
class BatmanBKMLearningConfig:
    enabled: bool = True
    allow_premarket_entry: bool = False
    opening_range_block_minutes: int = 15
    min_bucket_samples: int = 3
    negative_bucket_win_rate_max: float = 0.45
    negative_bucket_avg_pnl_max: float = -2000.0
    positive_bucket_win_rate_min: float = 0.60
    positive_bucket_avg_pnl_min: float = 1000.0
    entry_block_penalty_score: float = 12.0
    risk_score_penalty_per_bucket: float = 4.0
    risk_score_bonus_per_bucket: float = 1.5
    risk_score_adjust_cap: float = 15.0
    review_window: int = 100

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "BatmanBKMLearningConfig":
        return cls(
            enabled=bool(settings.get("batman_bkm_learning_enabled", True)),
            allow_premarket_entry=bool(settings.get("batman_bkm_learning_allow_premarket_entry", False)),
            opening_range_block_minutes=max(0, int(float(settings.get("batman_bkm_learning_opening_range_block_minutes", 15) or 15))),
            min_bucket_samples=max(1, int(float(settings.get("batman_bkm_learning_min_bucket_samples", 3) or 3))),
            negative_bucket_win_rate_max=max(0.0, min(1.0, float(settings.get("batman_bkm_learning_negative_bucket_win_rate_max", 0.45) or 0.45))),
            negative_bucket_avg_pnl_max=float(settings.get("batman_bkm_learning_negative_bucket_avg_pnl_max", -2000.0) or -2000.0),
            positive_bucket_win_rate_min=max(0.0, min(1.0, float(settings.get("batman_bkm_learning_positive_bucket_win_rate_min", 0.60) or 0.60))),
            positive_bucket_avg_pnl_min=float(settings.get("batman_bkm_learning_positive_bucket_avg_pnl_min", 1000.0) or 1000.0),
            entry_block_penalty_score=max(1.0, float(settings.get("batman_bkm_learning_entry_block_penalty_score", 12.0) or 12.0)),
            risk_score_penalty_per_bucket=max(0.5, float(settings.get("batman_bkm_learning_risk_score_penalty_per_bucket", 4.0) or 4.0)),
            risk_score_bonus_per_bucket=max(0.0, float(settings.get("batman_bkm_learning_risk_score_bonus_per_bucket", 1.5) or 1.5)),
            risk_score_adjust_cap=max(1.0, float(settings.get("batman_bkm_learning_risk_score_adjust_cap", 15.0) or 15.0)),
            review_window=max(20, int(float(settings.get("batman_bkm_learning_review_window", 100) or 100))),
        )


@dataclass
class BatmanBKMLearningState:
    status: str = "IDLE"
    total_outcomes: int = 0
    wins: int = 0
    losses: int = 0
    flats: int = 0
    realized_pnl_rs: float = 0.0
    last_outcome_at: Optional[str] = None
    last_outcome_id: Optional[str] = None
    last_entry_assessment: Optional[Dict[str, Any]] = None
    top_negative_buckets: List[Dict[str, Any]] = None  # type: ignore[assignment]
    top_positive_buckets: List[Dict[str, Any]] = None  # type: ignore[assignment]
    updated_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["top_negative_buckets"] = list(payload.get("top_negative_buckets") or [])
        payload["top_positive_buckets"] = list(payload.get("top_positive_buckets") or [])
        payload["last_entry_assessment"] = dict(payload.get("last_entry_assessment") or {})
        return payload


class BatmanBKMLearningManager:
    _BUCKET_KEYS: Tuple[str, ...] = (
        "entry_phase",
        "entry_mode",
        "market_bias",
        "pcr_bias",
        "oi_build_bias",
        "volatility_regime",
        "breakout_confirmation",
        "dominant_signal_bias",
        "signal_conflict_bucket",
        "trend_confidence_bucket",
        "quality_status",
    )

    def __init__(self, *, config: BatmanBKMLearningConfig, status_path: Path, outcomes_path: Path, logger: Any = None) -> None:
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

    def _load_state(self) -> BatmanBKMLearningState:
        if not self.status_path.exists():
            return BatmanBKMLearningState()
        try:
            payload = json.loads(self.status_path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("invalid payload")
            state = BatmanBKMLearningState(**{k: payload.get(k) for k in BatmanBKMLearningState.__dataclass_fields__.keys()})  # type: ignore[arg-type]
            state.top_negative_buckets = list(state.top_negative_buckets or [])
            state.top_positive_buckets = list(state.top_positive_buckets or [])
            state.last_entry_assessment = dict(state.last_entry_assessment or {})
            return state
        except Exception:
            return BatmanBKMLearningState()

    def _persist(self, when: Optional[datetime] = None) -> None:
        now = when or _now_ist()
        self.state.updated_at = now.isoformat(timespec="seconds")
        try:
            self.status_path.write_text(json.dumps(self.state.as_dict(), indent=2))
        except Exception:
            self._log("exception", "Failed to write Batman BKM learning status")

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

    def _append_outcome(self, payload: Dict[str, Any]) -> None:
        try:
            with self.outcomes_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            self._log("exception", "Failed to append Batman BKM learning outcome")

    def snapshot(self) -> Dict[str, Any]:
        return self.state.as_dict()

    def _stats_for_bucket(self, rows: Iterable[Dict[str, Any]], bucket_name: str, label: str) -> Dict[str, Any]:
        scoped = [row for row in rows if _normalize_label(row.get(bucket_name)) == label]
        count = len(scoped)
        if count == 0:
            return {"count": 0, "wins": 0, "losses": 0, "flats": 0, "avg_pnl_rs": 0.0, "win_rate": None, "loss_rate": None}
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
            "loss_rate": round(losses / count, 4),
        }

    def refresh_summary(self) -> Dict[str, Any]:
        rows = self._load_outcomes()[-self.config.review_window :]
        wins = sum(1 for row in rows if str(row.get("result") or "").upper() == "WIN")
        losses = sum(1 for row in rows if str(row.get("result") or "").upper() == "LOSS")
        flats = sum(1 for row in rows if str(row.get("result") or "").upper() == "FLAT")
        pnl = round(sum(float(_safe_float(row.get("pnl_rs"), 0.0) or 0.0) for row in rows), 2)
        negatives: List[Dict[str, Any]] = []
        positives: List[Dict[str, Any]] = []
        for bucket_name in self._BUCKET_KEYS:
            labels = sorted({_normalize_label(row.get(bucket_name)) for row in rows if row.get(bucket_name) not in (None, "", "UNKNOWN")})
            for label in labels:
                stats = self._stats_for_bucket(rows, bucket_name, label)
                if stats["count"] < self.config.min_bucket_samples:
                    continue
                row = {
                    "bucket": bucket_name,
                    "label": label,
                    **stats,
                }
                if stats["avg_pnl_rs"] <= self.config.negative_bucket_avg_pnl_max and (stats["win_rate"] or 0.0) <= self.config.negative_bucket_win_rate_max:
                    negatives.append(row)
                if stats["avg_pnl_rs"] >= self.config.positive_bucket_avg_pnl_min and (stats["win_rate"] or 0.0) >= self.config.positive_bucket_win_rate_min:
                    positives.append(row)
        negatives.sort(key=lambda item: (float(item["avg_pnl_rs"]), float(item.get("win_rate") or 0.0), -int(item["count"])))
        positives.sort(key=lambda item: (-float(item["avg_pnl_rs"]), -float(item.get("win_rate") or 0.0), -int(item["count"])))

        self.state.status = "ACTIVE" if self.config.enabled else "DISABLED"
        self.state.total_outcomes = len(rows)
        self.state.wins = wins
        self.state.losses = losses
        self.state.flats = flats
        self.state.realized_pnl_rs = pnl
        self.state.last_outcome_at = str(rows[-1].get("closed_at") or rows[-1].get("timestamp") or "") if rows else None
        self.state.last_outcome_id = str(rows[-1].get("trade_id") or "") if rows else None
        self.state.top_negative_buckets = negatives[:8]
        self.state.top_positive_buckets = positives[:8]
        self._persist()
        return self.snapshot()

    def _entry_phase(self, when: Optional[datetime], *, market_open_time: dtime = dtime(9, 15)) -> str:
        if when is None:
            return "UNKNOWN"
        if (when.hour, when.minute) < (market_open_time.hour, market_open_time.minute):
            return "PRE_MARKET"
        block_until_minute = market_open_time.minute + int(self.config.opening_range_block_minutes)
        open_buffer_end = dtime(market_open_time.hour + (block_until_minute // 60), block_until_minute % 60)
        if (when.hour, when.minute) < (open_buffer_end.hour, open_buffer_end.minute):
            return "OPENING_RANGE"
        return "LIVE_SESSION"

    def normalize_entry_snapshot(self, snapshot: Dict[str, Any], *, when: Optional[datetime] = None) -> Dict[str, Any]:
        now = when or _parse_dt(snapshot.get("timestamp")) or _now_ist()
        payload = dict(snapshot or {})
        payload["timestamp"] = now.isoformat(timespec="seconds")
        payload["entry_phase"] = _normalize_label(payload.get("entry_phase") or self._entry_phase(now))
        payload["entry_mode"] = _normalize_label(payload.get("entry_mode") or "UNKNOWN")
        payload["market_bias"] = _normalize_label(payload.get("market_bias"))
        payload["pcr_bias"] = _normalize_label(payload.get("pcr_bias"))
        payload["oi_build_bias"] = _normalize_label(payload.get("oi_build_bias"))
        payload["volatility_regime"] = _normalize_label(payload.get("volatility_regime"))
        payload["breakout_confirmation"] = _normalize_label(payload.get("breakout_confirmation"))
        payload["dominant_signal_bias"] = _normalize_label(payload.get("dominant_signal_bias"))
        payload["quality_status"] = _normalize_label(payload.get("quality_status"))
        payload["signal_conflict_bucket"] = _signal_conflict_bucket(payload.get("signal_conflict_score"))
        payload["trend_confidence_bucket"] = _trend_confidence_bucket(payload.get("trend_confidence"))
        payload["opened_pre_market"] = payload["entry_phase"] == "PRE_MARKET"
        return payload

    def assess_entry(self, snapshot: Dict[str, Any], *, when: Optional[datetime] = None) -> Dict[str, Any]:
        normalized = self.normalize_entry_snapshot(snapshot, when=when)
        assessment = self._assess_snapshot(normalized, mode="ENTRY")
        self.state.last_entry_assessment = assessment
        self._persist()
        return assessment

    def assess_open_context(self, snapshot: Dict[str, Any], *, when: Optional[datetime] = None) -> Dict[str, Any]:
        normalized = self.normalize_entry_snapshot(snapshot, when=when)
        assessment = self._assess_snapshot(normalized, mode="MONITOR")
        self.state.last_entry_assessment = assessment
        self._persist()
        return assessment

    def _assess_snapshot(self, normalized: Dict[str, Any], *, mode: str) -> Dict[str, Any]:
        assessment = {
            "mode": mode,
            "status": "ALLOW",
            "block_entry": False,
            "risk_score_adjust": 0.0,
            "reasons": [],
            "supportive_reasons": [],
            "matched_negative_buckets": [],
            "matched_positive_buckets": [],
            "timestamp": normalized.get("timestamp"),
        }
        if not self.config.enabled:
            assessment["status"] = "DISABLED"
            assessment["reasons"] = ["LEARNING_DISABLED"]
            return assessment

        if mode == "ENTRY" and not self.config.allow_premarket_entry and normalized.get("entry_phase") == "PRE_MARKET":
            assessment["status"] = "BLOCK"
            assessment["block_entry"] = True
            assessment["reasons"] = ["LEARNING_BLOCK_PRE_MARKET_ENTRY"]
            return assessment

        if mode == "ENTRY" and normalized.get("entry_mode") == "FORCE" and normalized.get("entry_phase") == "OPENING_RANGE":
            assessment["status"] = "BLOCK"
            assessment["block_entry"] = True
            assessment["reasons"] = ["LEARNING_BLOCK_FORCE_ENTRY_IN_OPENING_RANGE"]
            return assessment

        rows = self._load_outcomes()[-self.config.review_window :]
        if not rows:
            assessment["status"] = "COLD_START"
            assessment["reasons"] = ["LEARNING_COLD_START"]
            return assessment

        penalty = 0.0
        bonus = 0.0
        negative_hits: List[Dict[str, Any]] = []
        positive_hits: List[Dict[str, Any]] = []

        for bucket_name in self._BUCKET_KEYS:
            label = _normalize_label(normalized.get(bucket_name))
            if label in {"UNKNOWN", ""}:
                continue
            stats = self._stats_for_bucket(rows, bucket_name, label)
            if stats["count"] < self.config.min_bucket_samples:
                continue
            hit = {"bucket": bucket_name, "label": label, **stats}
            if stats["avg_pnl_rs"] <= self.config.negative_bucket_avg_pnl_max and (stats["win_rate"] or 0.0) <= self.config.negative_bucket_win_rate_max:
                penalty += float(self.config.risk_score_penalty_per_bucket)
                negative_hits.append(hit)
            elif stats["avg_pnl_rs"] >= self.config.positive_bucket_avg_pnl_min and (stats["win_rate"] or 0.0) >= self.config.positive_bucket_win_rate_min:
                bonus += float(self.config.risk_score_bonus_per_bucket)
                positive_hits.append(hit)

        raw_adjust = penalty - bonus
        cap = float(self.config.risk_score_adjust_cap)
        assessment["risk_score_adjust"] = round(max(-cap, min(cap, raw_adjust)), 2)
        assessment["matched_negative_buckets"] = negative_hits
        assessment["matched_positive_buckets"] = positive_hits

        if negative_hits:
            assessment["reasons"].extend([f"LEARNING_NEGATIVE_{hit['bucket']}_{hit['label']}" for hit in negative_hits[:4]])
        if positive_hits:
            assessment["supportive_reasons"].extend([f"LEARNING_POSITIVE_{hit['bucket']}_{hit['label']}" for hit in positive_hits[:4]])

        if mode == "ENTRY" and assessment["risk_score_adjust"] >= float(self.config.entry_block_penalty_score):
            assessment["status"] = "BLOCK"
            assessment["block_entry"] = True
            assessment["reasons"].insert(0, "LEARNING_BLOCK_HIGH_RISK_MATCH")
        elif assessment["risk_score_adjust"] > 0:
            assessment["status"] = "CAUTION"
        elif assessment["risk_score_adjust"] < 0:
            assessment["status"] = "SUPPORTIVE"
        else:
            assessment["status"] = "ALLOW"
        if not assessment["reasons"]:
            assessment["reasons"] = ["LEARNING_NEUTRAL"]
        return assessment

    def record_outcome(
        self,
        *,
        expiry: str,
        trade_mode: str,
        opened_at: Any,
        closed_at: Any,
        close_reason: str,
        pnl_rs: float,
        entry_snapshot: Optional[Dict[str, Any]] = None,
        close_context: Optional[Dict[str, Any]] = None,
        quality_status: Optional[str] = None,
        quality_score: Optional[float] = None,
        net_credit_rs: Optional[float] = None,
    ) -> Dict[str, Any]:
        opened_dt = _parse_dt(opened_at)
        closed_dt = _parse_dt(closed_at)
        held_minutes = None
        if opened_dt and closed_dt:
            held_minutes = round(max((closed_dt - opened_dt).total_seconds(), 0.0) / 60.0, 2)
        normalized_entry = self.normalize_entry_snapshot(
            {
                **(entry_snapshot or {}),
                "quality_status": quality_status if quality_status is not None else (entry_snapshot or {}).get("quality_status"),
                "quality_score": quality_score if quality_score is not None else (entry_snapshot or {}).get("quality_score"),
            },
            when=opened_dt,
        )
        trade_id = f"{expiry}|{opened_dt.isoformat() if opened_dt else opened_at}|{closed_dt.isoformat() if closed_dt else closed_at}"
        existing_ids = {str(row.get("trade_id") or "") for row in self._load_outcomes()}
        if trade_id in existing_ids:
            return {"ok": True, "duplicate": True, "trade_id": trade_id}
        row = {
            "trade_id": trade_id,
            "trade_mode": str(trade_mode or "").strip().lower(),
            "expiry": str(expiry or ""),
            "opened_at": opened_dt.isoformat(timespec="seconds") if opened_dt else str(opened_at or ""),
            "closed_at": closed_dt.isoformat(timespec="seconds") if closed_dt else str(closed_at or ""),
            "held_minutes": held_minutes,
            "close_reason": _normalize_label(close_reason, default="UNKNOWN"),
            "pnl_rs": round(float(pnl_rs or 0.0), 2),
            "result": _result_from_pnl(float(pnl_rs or 0.0)),
            "entry_phase": normalized_entry.get("entry_phase"),
            "entry_mode": normalized_entry.get("entry_mode"),
            "opened_pre_market": bool(normalized_entry.get("opened_pre_market", False)),
            "market_bias": normalized_entry.get("market_bias"),
            "pcr_bias": normalized_entry.get("pcr_bias"),
            "oi_build_bias": normalized_entry.get("oi_build_bias"),
            "volatility_regime": normalized_entry.get("volatility_regime"),
            "breakout_confirmation": normalized_entry.get("breakout_confirmation"),
            "dominant_signal_bias": normalized_entry.get("dominant_signal_bias"),
            "signal_conflict_bucket": normalized_entry.get("signal_conflict_bucket"),
            "trend_confidence_bucket": normalized_entry.get("trend_confidence_bucket"),
            "quality_status": normalized_entry.get("quality_status"),
            "quality_score": _safe_float(quality_score if quality_score is not None else normalized_entry.get("quality_score"), None),
            "net_credit_rs": _safe_float(net_credit_rs, None),
            "entry_snapshot": normalized_entry,
            "close_context": close_context or {},
            "timestamp": _now_ist().isoformat(timespec="seconds"),
        }
        self._append_outcome(row)
        self.refresh_summary()
        return {"ok": True, "duplicate": False, "trade_id": trade_id, "row": row}

    def backfill_closed_baskets(self, *, strategy_state: Dict[str, Any], ai_events: Optional[List[Dict[str, Any]]] = None, trade_mode: str = "paper") -> int:
        bucket = (strategy_state.get("BATMAN_BKM") if isinstance(strategy_state, dict) else {}) or {}
        if not isinstance(bucket, dict):
            return 0
        ai_rows = list(ai_events or [])
        appended = 0
        for expiry, payload in bucket.items():
            if not isinstance(payload, dict):
                continue
            if _normalize_label(payload.get("status")) != "CLOSED":
                continue
            opened_at = payload.get("opened_at")
            closed_at = payload.get("closed_at")
            if not opened_at or not closed_at:
                continue
            close_ctx = {}
            for row in reversed(ai_rows):
                if str(row.get("basket_expiry") or "") != str(expiry):
                    continue
                ts = _parse_dt(row.get("timestamp"))
                closed_dt = _parse_dt(closed_at)
                if closed_dt is not None and ts is not None and ts > closed_dt:
                    continue
                close_ctx = {
                    "market_bias": row.get("market_bias"),
                    "position_risk_side": row.get("position_risk_side"),
                    "ai_reasons": list(row.get("reasons") or []),
                    "ai_confidence": _safe_float(row.get("confidence"), None),
                    "ai_score": _safe_float(row.get("score"), None),
                }
                break
            entry_snapshot = payload.get("learning_entry") if isinstance(payload.get("learning_entry"), dict) else {}
            res = self.record_outcome(
                expiry=str(expiry),
                trade_mode=str(trade_mode or "paper"),
                opened_at=opened_at,
                closed_at=closed_at,
                close_reason=str(payload.get("reason") or ""),
                pnl_rs=float(_safe_float(payload.get("pnl"), 0.0) or 0.0),
                entry_snapshot=entry_snapshot,
                close_context=close_ctx,
                quality_status=str(payload.get("quality_status") or ""),
                quality_score=_safe_float(payload.get("quality_score"), None),
                net_credit_rs=_safe_float(payload.get("net_credit"), None),
            )
            if bool(res.get("ok")) and not bool(res.get("duplicate")):
                appended += 1
        if appended:
            self.refresh_summary()
        return appended
