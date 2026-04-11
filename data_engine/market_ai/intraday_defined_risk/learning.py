from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import math
import sqlite3
from pathlib import Path
from statistics import median
from typing import Any

from .data_models import AdaptiveParameters, CalibrationResult, DecisionOutput


DEFAULT_DB_PATH = Path("/tmp/intraday_defined_risk_learning.sqlite3")


class LearningStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    playbook TEXT NOT NULL DEFAULT 'UNKNOWN',
                    strategy TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    action TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    feature_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date TEXT NOT NULL,
                    playbook TEXT NOT NULL DEFAULT 'UNKNOWN',
                    strategy TEXT NOT NULL,
                    pnl_rupees REAL NOT NULL,
                    max_drawdown_rupees REAL NOT NULL,
                    margin_utilisation REAL NOT NULL,
                    feature_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS parameter_state (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    active_json TEXT NOT NULL,
                    shadow_json TEXT,
                    shadow_started_at TEXT,
                    shadow_sessions_seen INTEGER NOT NULL DEFAULT 0,
                    shadow_objective REAL,
                    shadow_drawdown REAL
                );
                CREATE TABLE IF NOT EXISTS parameter_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_json TEXT NOT NULL,
                    candidate_json TEXT,
                    objective_before REAL,
                    objective_after REAL,
                    notes TEXT NOT NULL
                );
                """
            )
            existing = conn.execute("SELECT singleton_id FROM parameter_state WHERE singleton_id = 1").fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO parameter_state (singleton_id, active_json) VALUES (1, ?)",
                    (json.dumps(asdict(AdaptiveParameters())),),
                )
            decision_cols = {row["name"] for row in conn.execute("PRAGMA table_info(decisions)").fetchall()}
            if "playbook" not in decision_cols:
                conn.execute("ALTER TABLE decisions ADD COLUMN playbook TEXT NOT NULL DEFAULT 'UNKNOWN'")
            outcome_cols = {row["name"] for row in conn.execute("PRAGMA table_info(outcomes)").fetchall()}
            if "playbook" not in outcome_cols:
                conn.execute("ALTER TABLE outcomes ADD COLUMN playbook TEXT NOT NULL DEFAULT 'UNKNOWN'")
            conn.commit()

    def active_parameters(self) -> AdaptiveParameters:
        with self._connect() as conn:
            row = conn.execute("SELECT active_json FROM parameter_state WHERE singleton_id = 1").fetchone()
        if not row:
            return AdaptiveParameters()
        return AdaptiveParameters(**json.loads(row["active_json"]))

    def log_decision(self, decision: DecisionOutput, features: dict[str, Any], session_date: str | None = None) -> None:
        ts = decision.entry.get("timestamp") or datetime.utcnow().isoformat()
        session = session_date or ts[:10]
        playbook = str(features.get("playbook") or "UNKNOWN")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions (ts, session_date, playbook, strategy, regime, confidence, action, payload_json, feature_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    session,
                    playbook,
                    decision.strategy.value,
                    decision.regime.value,
                    decision.confidence_score,
                    decision.action,
                    decision.to_json(),
                    json.dumps(features, default=str, sort_keys=True),
                ),
            )
            conn.commit()

    def log_outcome(
        self,
        session_date: str,
        strategy: str,
        pnl_rupees: float,
        max_drawdown_rupees: float,
        margin_utilisation: float,
        features: dict[str, Any],
    ) -> None:
        created_at = datetime.utcnow().isoformat()
        playbook = str(features.get("playbook") or "UNKNOWN")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO outcomes (session_date, playbook, strategy, pnl_rupees, max_drawdown_rupees, margin_utilisation, feature_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_date,
                    playbook,
                    strategy,
                    pnl_rupees,
                    max_drawdown_rupees,
                    margin_utilisation,
                    json.dumps(features, default=str, sort_keys=True),
                    created_at,
                ),
            )
            conn.commit()

    def playbook_summary(self, window: int = 120) -> dict[str, dict[str, float]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT playbook, pnl_rupees, max_drawdown_rupees, margin_utilisation
                FROM outcomes
                ORDER BY session_date DESC, created_at DESC
                LIMIT ?
                """,
                (window,),
            ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["playbook"] or "UNKNOWN", []).append(row)
        summary: dict[str, dict[str, float]] = {}
        for playbook, grouped_rows in grouped.items():
            wins = sum(1 for row in grouped_rows if row["pnl_rupees"] > 0)
            gross_profit = sum(max(row["pnl_rupees"], 0.0) for row in grouped_rows)
            gross_loss = abs(sum(min(row["pnl_rupees"], 0.0) for row in grouped_rows))
            summary[playbook] = {
                "samples": float(len(grouped_rows)),
                "win_rate": wins / len(grouped_rows) if grouped_rows else 0.0,
                "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else gross_profit,
                "objective": _objective(grouped_rows) if grouped_rows else -math.inf,
                "max_drawdown": _max_drawdown(grouped_rows) if grouped_rows else 0.0,
            }
        return summary

    def calibrate(self, window: int = 60) -> CalibrationResult:
        current = self.active_parameters().clamped()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM outcomes
                ORDER BY session_date DESC, created_at DESC
                LIMIT ?
                """,
                (window,),
            ).fetchall()
        if len(rows) < 30:
            return CalibrationResult(
                status="REJECTED",
                reasons=["At least 30 outcome samples are required before calibration."],
                current_parameters=current,
            )

        objective_before = _objective(rows)
        candidate = _heuristic_candidate(current, rows).clamped()
        candidate_objective = _estimate_candidate_objective(candidate, rows)
        candidate_drawdown = _max_drawdown(rows)
        current_drawdown = _max_drawdown(rows)
        if candidate_objective <= objective_before or candidate_drawdown > current_drawdown:
            self._record_history(
                status="REJECTED",
                current=current,
                candidate=candidate,
                objective_before=objective_before,
                objective_after=candidate_objective,
                notes="Candidate did not improve objective or increased drawdown.",
            )
            return CalibrationResult(
                status="REJECTED",
                reasons=["Candidate parameters did not improve the guarded objective."],
                current_parameters=current,
                candidate_parameters=candidate,
                objective_before=objective_before,
                objective_after=candidate_objective,
            )

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE parameter_state
                SET shadow_json = ?, shadow_started_at = ?, shadow_sessions_seen = 0, shadow_objective = ?, shadow_drawdown = ?
                WHERE singleton_id = 1
                """,
                (
                    json.dumps(asdict(candidate)),
                    datetime.utcnow().isoformat(),
                    candidate_objective,
                    candidate_drawdown,
                ),
            )
            conn.commit()
        self._record_history(
            status="SHADOW",
            current=current,
            candidate=candidate,
            objective_before=objective_before,
            objective_after=candidate_objective,
            notes="Candidate entered five-session shadow mode.",
        )
        return CalibrationResult(
            status="SHADOW",
            reasons=["Candidate parameters entered guarded shadow mode for five sessions."],
            current_parameters=current,
            candidate_parameters=candidate,
            objective_before=objective_before,
            objective_after=candidate_objective,
        )

    def record_shadow_session(self, session_objective: float, session_drawdown: float) -> CalibrationResult:
        with self._connect() as conn:
            state = conn.execute(
                "SELECT active_json, shadow_json, shadow_sessions_seen, shadow_objective, shadow_drawdown FROM parameter_state WHERE singleton_id = 1"
            ).fetchone()
            if not state or not state["shadow_json"]:
                return CalibrationResult(
                    status="NO_SHADOW",
                    reasons=["No shadow candidate is currently active."],
                    current_parameters=self.active_parameters(),
                )

            current = AdaptiveParameters(**json.loads(state["active_json"]))
            candidate = AdaptiveParameters(**json.loads(state["shadow_json"]))
            seen = int(state["shadow_sessions_seen"]) + 1
            candidate_objective = float(state["shadow_objective"])
            candidate_drawdown = float(state["shadow_drawdown"])

            if session_drawdown > candidate_drawdown:
                conn.execute(
                    "UPDATE parameter_state SET shadow_json = NULL, shadow_started_at = NULL, shadow_sessions_seen = 0, shadow_objective = NULL, shadow_drawdown = NULL WHERE singleton_id = 1"
                )
                conn.commit()
                self._record_history(
                    status="ROLLED_BACK",
                    current=current,
                    candidate=candidate,
                    objective_before=session_objective,
                    objective_after=candidate_objective,
                    notes="Shadow candidate breached its drawdown guardrail.",
                )
                return CalibrationResult(
                    status="ROLLED_BACK",
                    reasons=["Shadow candidate drawdown breached the safety threshold."],
                    current_parameters=current,
                    candidate_parameters=candidate,
                    objective_before=session_objective,
                    objective_after=candidate_objective,
                )

            if seen >= candidate.shadow_sessions and session_objective >= candidate_objective:
                conn.execute(
                    """
                    UPDATE parameter_state
                    SET active_json = ?, shadow_json = NULL, shadow_started_at = NULL, shadow_sessions_seen = 0, shadow_objective = NULL, shadow_drawdown = NULL
                    WHERE singleton_id = 1
                    """,
                    (json.dumps(asdict(candidate)),),
                )
                conn.commit()
                self._record_history(
                    status="PROMOTED",
                    current=current,
                    candidate=candidate,
                    objective_before=session_objective,
                    objective_after=candidate_objective,
                    notes="Shadow candidate promoted after guarded trial.",
                )
                return CalibrationResult(
                    status="PROMOTED",
                    reasons=["Shadow candidate promoted to active parameters."],
                    current_parameters=candidate,
                    candidate_parameters=candidate,
                    objective_before=session_objective,
                    objective_after=candidate_objective,
                )

            conn.execute(
                "UPDATE parameter_state SET shadow_sessions_seen = ? WHERE singleton_id = 1",
                (seen,),
            )
            conn.commit()
            return CalibrationResult(
                status="SHADOW",
                reasons=[f"Shadow candidate session count advanced to {seen}."],
                current_parameters=current,
                candidate_parameters=candidate,
                objective_before=session_objective,
                objective_after=candidate_objective,
            )

    def _record_history(
        self,
        status: str,
        current: AdaptiveParameters,
        candidate: AdaptiveParameters | None,
        objective_before: float | None,
        objective_after: float | None,
        notes: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO parameter_history (created_at, status, current_json, candidate_json, objective_before, objective_after, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    status,
                    json.dumps(asdict(current)),
                    json.dumps(asdict(candidate)) if candidate else None,
                    objective_before,
                    objective_after,
                    notes,
                ),
            )
            conn.commit()


def _objective(rows: list[sqlite3.Row]) -> float:
    gross_profit = sum(max(row["pnl_rupees"], 0.0) for row in rows)
    gross_loss = abs(sum(min(row["pnl_rupees"], 0.0) for row in rows))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
    drawdown_ratio = _max_drawdown(rows) / max(gross_profit, 1.0)
    margin_penalty = median(float(row["margin_utilisation"]) for row in rows)
    return profit_factor - (0.5 * drawdown_ratio) - (0.3 * margin_penalty)


def _estimate_candidate_objective(candidate: AdaptiveParameters, rows: list[sqlite3.Row]) -> float:
    filtered: list[sqlite3.Row] = []
    for row in rows:
        features = json.loads(row["feature_json"])
        rv30 = float(features.get("rv30_pct", 0.0))
        credit_ratio = float(features.get("credit_width_ratio", 0.0))
        is_condor = row["strategy"] == "IRON_CONDOR"
        threshold = candidate.condor_credit_width_ratio if is_condor else candidate.directional_credit_width_ratio
        if credit_ratio < threshold:
            continue
        if rv30 > candidate.rv_high_cutoff and row["strategy"] == "IRON_CONDOR":
            continue
        filtered.append(row)
    if not filtered:
        return -math.inf
    return _objective(filtered)


def _heuristic_candidate(current: AdaptiveParameters, rows: list[sqlite3.Row]) -> AdaptiveParameters:
    features = [json.loads(row["feature_json"]) for row in rows]
    high_rv_losses = [row["pnl_rupees"] for row, feature in zip(rows, features) if float(feature.get("rv30_pct", 0.0)) > current.rv_high_cutoff]
    low_credit_losses = [
        row["pnl_rupees"]
        for row, feature in zip(rows, features)
        if float(feature.get("credit_width_ratio", 0.0)) < current.directional_credit_width_ratio
    ]
    candidate = current
    if high_rv_losses and sum(high_rv_losses) < 0:
        candidate = AdaptiveParameters(
            rv_high_cutoff=min(current.rv_high_cutoff + 0.05, 0.75),
            rv_mid_cutoff=current.rv_mid_cutoff,
            directional_credit_width_ratio=current.directional_credit_width_ratio,
            condor_credit_width_ratio=current.condor_credit_width_ratio,
            shadow_sessions=current.shadow_sessions,
        )
    if low_credit_losses and sum(low_credit_losses) < 0:
        candidate = AdaptiveParameters(
            rv_high_cutoff=candidate.rv_high_cutoff,
            rv_mid_cutoff=candidate.rv_mid_cutoff,
            directional_credit_width_ratio=min(candidate.directional_credit_width_ratio + 0.02, 0.40),
            condor_credit_width_ratio=min(candidate.condor_credit_width_ratio + 0.02, 0.40),
            shadow_sessions=current.shadow_sessions,
        )
    return candidate


def _max_drawdown(rows: list[sqlite3.Row]) -> float:
    peak = 0.0
    equity = 0.0
    drawdown = 0.0
    for row in reversed(rows):
        equity += row["pnl_rupees"]
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown
