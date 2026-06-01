from __future__ import annotations

import math

from .data_models import AccountRiskLimits, AccountState, RiskAssessment, StrategyType, TradeStructure


def _confidence_lot_cap(
    confidence: float,
    daily_bias_score: int,
    strategy: StrategyType,
    max_lots: int,
) -> int:
    """Scale allowed lots by regime confidence and daily trend alignment.

    Higher conviction (confidence + matching daily trend) → more lots, up to max.
    Counter-trend intraday trades get reduced sizing.

    Scaling table (with aligned daily):
      confidence >= 0.90 → full max_lots
      confidence >= 0.80 → 80% of max_lots
      confidence >= 0.70 → 60% of max_lots
      confidence >= 0.60 → 40% of max_lots
      confidence <  0.60 → 1 lot (minimum conviction)

    daily_bias_score adjustment (bearish for Bear Call, bullish for Bull Put):
      score >= 2 (aligned):      +1 extra lot up to max
      score == 0 (neutral):       no change
      score <= -1 (counter-trend): halve the lot cap
    """
    if confidence >= 0.90:
        base_fraction = 1.0
    elif confidence >= 0.80:
        base_fraction = 0.80
    elif confidence >= 0.70:
        base_fraction = 0.60
    elif confidence >= 0.60:
        base_fraction = 0.40
    else:
        return 1

    cap = max(1, round(max_lots * base_fraction))

    # Daily alignment modifier
    is_bearish_strategy = strategy in {StrategyType.BEAR_CALL_CREDIT_SPREAD}
    is_bullish_strategy = strategy in {StrategyType.BULL_PUT_CREDIT_SPREAD}
    aligned_bearish = is_bearish_strategy and daily_bias_score <= -2
    aligned_bullish = is_bullish_strategy and daily_bias_score >= 2
    counter_trend = (is_bearish_strategy and daily_bias_score >= 1) or (is_bullish_strategy and daily_bias_score <= -1)

    if aligned_bearish or aligned_bullish:
        cap = min(cap + 1, max_lots)  # daily trend confirms intraday → bonus lot
    elif counter_trend:
        cap = max(1, cap // 2)  # going against daily trend → cut size

    return max(1, cap)


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
    # Apply confidence + daily-alignment scaling: high-conviction trades get more lots
    confidence_cap = _confidence_lot_cap(
        confidence=float(getattr(structure, "confidence", 1.0)),
        daily_bias_score=int(getattr(structure, "daily_bias_score", 0)),
        strategy=structure.strategy,
        max_lots=lots,
    )
    lots = min(lots, confidence_cap)
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
