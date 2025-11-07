# market_ai/modules/risk_manager.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

LOG = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    max_daily_loss: float        # absolute daily loss threshold
    per_leg_sl_mult: float       # e.g., 2.0 => roll/exit if leg LTP >= 2× entry leg premium
    max_rolls_per_day: int       # e.g., 3
    hard_kill: bool = False      # external kill toggle
    equity: float = 0.0          # account equity reference
    max_exposure_pct: float = 0.1  # cap per trade notional / equity
    overall_sl_pct: float = 2.0     # overall stop as ratio of entry credit
    max_portfolio_delta: float = 0.25  # allowable net delta magnitude post hedging

class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.l = limits
        self.roll_count = 0
        self.day_pnl = 0.0
        self.last_block_reason: Optional[str] = None

    def on_roll(self):
        self.roll_count += 1

    def on_mtm(self, day_pnl: float):
        self.day_pnl = day_pnl

    def allow_trade(self) -> bool:
        self.last_block_reason = None
        if self.l.hard_kill:
            self.last_block_reason = "hard_kill"
            LOG.warning("Hard kill active; trading blocked.")
            return False
        if self.day_pnl <= -abs(self.l.max_daily_loss):
            self.last_block_reason = "daily_loss"
            LOG.warning("Max daily loss breached; trading blocked.")
            return False
        if self.roll_count >= self.l.max_rolls_per_day:
            self.last_block_reason = "max_rolls"
            LOG.warning("Max rolls per day reached; trading blocked.")
            return False
        return True

    def can_allocate(self, notional: float) -> bool:
        equity = self.l.equity
        if equity <= 0 or self.l.max_exposure_pct <= 0:
            return True
        exposure_pct = notional / equity
        if exposure_pct > self.l.max_exposure_pct:
            LOG.warning("Exposure %.2f%% exceeds %.2f%% cap", exposure_pct * 100, self.l.max_exposure_pct * 100)
            return False
        return True

    def per_leg_stop_hit(self, entry_leg_px: float, current_leg_ltp: float) -> bool:
        if entry_leg_px <= 0:
            return False
        return current_leg_ltp >= (self.l.per_leg_sl_mult * entry_leg_px)

    def portfolio_stop_hit(self, entry_credit: float, realized_pnl: float) -> bool:
        if entry_credit <= 0 or self.l.overall_sl_pct <= 0:
            return False
        loss = -realized_pnl if realized_pnl < 0 else 0.0
        return loss >= (self.l.overall_sl_pct * entry_credit)

    def delta_within_bounds(self, net_delta: float) -> bool:
        limit = abs(self.l.max_portfolio_delta)
        if limit <= 0:
            return True
        return abs(net_delta) <= limit
