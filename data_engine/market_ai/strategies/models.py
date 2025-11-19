from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import List, Optional


class TrendSide(str, Enum):
    SIDEWAYS = "SIDEWAYS"
    UP = "UP"
    DOWN = "DOWN"
    NO_TRADE = "NO_TRADE"


class LegSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


class StrategyType(str, Enum):
    NONE = "NONE"
    STRANGLE = "STRANGLE"
    TREND_SPREAD = "TREND_SPREAD"


@dataclass
class OptionLeg:
    symbol: str
    expiry: date
    strike: float
    option_type: OptionType
    side: LegSide
    quantity: int
    entry_price: float = 0.0
    security_id: Optional[str] = None
    current_ltp: Optional[float] = None


@dataclass
class TradeAction:
    action_type: str  # HOLD | OPEN_STRANGLE | OPEN_SPREAD | CLOSE_ALL | CLOSE_LEGS
    strategy_type: StrategyType
    legs_to_open: List[OptionLeg]
    legs_to_close: List[OptionLeg]
    reason: str


@dataclass
class TrendContext:
    trend_side: TrendSide
    rsi: float
    atr: float
    breakout_up: bool
    breakdown_down: bool


@dataclass
class MarketSnapshot:
    symbol: str
    spot: float
    candles_15m: List[dict]
    yesterday_high: float
    yesterday_low: float
    india_vix: float
    now: datetime


@dataclass
class RiskConfig:
    max_intraday_loss: float
    intraday_target: float
    allow_carry_forward: bool
    max_carry_days: int
    vix_carry_threshold: float
    last_entry_time: time
    force_exit_time: time
