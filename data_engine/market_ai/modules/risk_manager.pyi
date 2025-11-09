from __future__ import annotations

from typing import Optional


class RiskLimits:
    max_daily_loss: float
    per_leg_sl_mult: float
    max_rolls_per_day: int
    hard_kill: bool
    equity: float
    max_exposure_pct: float
    overall_sl_pct: float
    max_portfolio_delta: float

    def __init__(
        self,
        max_daily_loss: float,
        per_leg_sl_mult: float,
        max_rolls_per_day: int,
        hard_kill: bool = ...,
        equity: float = ...,
        max_exposure_pct: float = ...,
        overall_sl_pct: float = ...,
        max_portfolio_delta: float = ...,
    ) -> None: ...


class RiskManager:
    l: RiskLimits
    roll_count: int
    day_pnl: float
    last_block_reason: Optional[str]

    def __init__(self, limits: RiskLimits) -> None: ...
    def on_roll(self) -> None: ...
    def on_mtm(self, day_pnl: float) -> None: ...
    def allow_trade(self) -> bool: ...
    def can_allocate(self, notional: float) -> bool: ...
    def per_leg_stop_hit(self, entry_leg_px: float, current_leg_ltp: float) -> bool: ...
    def portfolio_stop_hit(self, entry_credit: float, realized_pnl: float) -> bool: ...
    def delta_within_bounds(self, net_delta: float) -> bool: ...
