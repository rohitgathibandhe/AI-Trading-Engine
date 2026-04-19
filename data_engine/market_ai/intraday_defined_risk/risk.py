from __future__ import annotations

import math

from .data_models import AccountRiskLimits, AccountState, RiskAssessment, StrategyType, TradeStructure


def kill_switch_triggered(account_state: AccountState, risk_limits: AccountRiskLimits) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if account_state.realised_pnl_rupees <= -risk_limits.max_daily_loss_rupees:
        reasons.append("Realised daily PnL breached the daily loss cap.")
    margin_utilisation = 0.0
    if risk_limits.max_margin_rupees > 0:
        margin_utilisation = account_state.margin_used_rupees / risk_limits.max_margin_rupees
    if margin_utilisation > 0.90:
        reasons.append("Margin utilisation breached 90%.")
    return bool(reasons), reasons


def compute_max_loss_rupees_per_lot(structure: TradeStructure, lot_size: int) -> float:
    if structure.strategy in {StrategyType.BEAR_CALL_CREDIT_SPREAD, StrategyType.BULL_PUT_CREDIT_SPREAD}:
        return max((structure.width_points - structure.credit_points) * lot_size, 0.0)
    if structure.strategy == StrategyType.CALL_DEBIT_SPREAD:
        debit_points = float(structure.metadata.get("debit_points") or 0.0)
        return max(debit_points * lot_size, 0.0)
    if structure.strategy == StrategyType.IRON_CONDOR:
        call_side = max(structure.call_width_points - structure.credit_points, 0.0)
        put_side = max(structure.put_width_points - structure.credit_points, 0.0)
        return max(call_side, put_side) * lot_size
    return 0.0


def assess_trade_risk(
    structure: TradeStructure,
    lot_size: int,
    risk_limits: AccountRiskLimits,
    account_state: AccountState,
    margin_estimate_per_lot: float | None = None,
) -> RiskAssessment:
    max_loss_per_lot = compute_max_loss_rupees_per_lot(structure, lot_size=lot_size)
    if max_loss_per_lot <= 0:
        return RiskAssessment(
            allowed=False,
            reasons=["Computed max loss per lot is non-positive."],
            max_loss_rupees_per_lot=0.0,
            lots=0,
            projected_margin_rupees=account_state.margin_used_rupees,
            projected_margin_utilisation=account_state.margin_used_rupees / risk_limits.max_margin_rupees,
        )

    effective_margin = margin_estimate_per_lot or structure.margin_estimate_per_lot
    if effective_margin is None:
        return RiskAssessment(
            allowed=False,
            reasons=["Margin estimate per lot is required to validate the trade."],
            max_loss_rupees_per_lot=max_loss_per_lot,
            lots=0,
            projected_margin_rupees=account_state.margin_used_rupees,
            projected_margin_utilisation=account_state.margin_used_rupees / risk_limits.max_margin_rupees,
        )

    max_lots_by_risk = math.floor(risk_limits.max_risk_rupees_per_trade / max_loss_per_lot)
    available_margin = max(risk_limits.max_margin_rupees - account_state.margin_used_rupees, 0.0)
    max_lots_by_margin = math.floor(available_margin / effective_margin)
    lots = max(min(max_lots_by_risk, max_lots_by_margin), 0)
    projected_margin = account_state.margin_used_rupees + (lots * effective_margin)
    projected_utilisation = projected_margin / risk_limits.max_margin_rupees if risk_limits.max_margin_rupees > 0 else 1.0

    reasons: list[str] = []
    allowed = True
    if lots < 1:
        allowed = False
        reasons.append("Risk or margin limits do not permit even one lot.")
    if projected_utilisation > 0.90:
        allowed = False
        reasons.append("Projected margin utilisation would exceed 90%.")

    return RiskAssessment(
        allowed=allowed,
        reasons=reasons,
        max_loss_rupees_per_lot=max_loss_per_lot,
        lots=lots,
        projected_margin_rupees=projected_margin,
        projected_margin_utilisation=projected_utilisation,
    )
