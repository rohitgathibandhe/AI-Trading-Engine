# market_ai/modules/risk_manager.py
from __future__ import annotations

import logging
from dataclasses import dataclass

LOG = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    max_daily_loss: float        # e.g., 1.5 * entry credit, or an absolute ₹ amount
    per_leg_sl_mult: float       # e.g., 2.0 => roll/exit if leg LTP >= 2× entry leg premium
    max_rolls_per_day: int       # e.g., 3
    hard_kill: bool = False      # external kill toggle

class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.l = limits
        self.roll_count = 0
        self.day_pnl = 0.0

    def on_roll(self):
        self.roll_count += 1

    def on_mtm(self, day_pnl: float):
        self.day_pnl = day_pnl

    def allow_trade(self) -> bool:
        if self.l.hard_kill:
            LOG.warning("Hard kill active; trading blocked.")
            return False
        if self.day_pnl <= -abs(self.l.max_daily_loss):
            LOG.warning("Max daily loss breached; trading blocked.")
            return False
        if self.roll_count >= self.l.max_rolls_per_day:
            LOG.warning("Max rolls per day reached; trading blocked.")
            return False
        return True

    def per_leg_stop_hit(self, entry_leg_px: float, current_leg_ltp: float) -> bool:
        if entry_leg_px <= 0:
            return False
        return current_leg_ltp >= (self.l.per_leg_sl_mult * entry_leg_px)
