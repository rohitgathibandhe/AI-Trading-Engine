from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time


@dataclass
class RiskConfig:
    max_loss_pct: float = 0.01
    daily_target_pct: float = 0.015
    time_exit: time = time(15, 10)
    per_leg_sl_mult: float = 1.6


@dataclass
class RiskState:
    capital: float
    day_start: datetime
    realized_pnl: float = 0.0
    open_exposure: float = 0.0
    entries: int = 0


class RiskManager:
    def __init__(self, config: RiskConfig, state: RiskState):
        self.config = config
        self.state = state

    def update_pnl(self, realized: float):
        self.state.realized_pnl += realized

    def can_trade(self, now: datetime) -> bool:
        if now.time() >= self.config.time_exit:
            return False
        loss_limit = -self.config.max_loss_pct * self.state.capital
        if self.state.realized_pnl <= loss_limit:
            return False
        target = self.config.daily_target_pct * self.state.capital
        if self.state.realized_pnl >= target:
            return False
        return True

    def leg_stop_triggered(self, entry_price: float, current_price: float) -> bool:
        return current_price >= entry_price * self.config.per_leg_sl_mult
