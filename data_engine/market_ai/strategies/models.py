from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum, auto
from typing import List, Optional


class TrendSide(Enum):
    SIDEWAYS = auto()
    BULL = auto()
    BEAR = auto()
    NO_TRADE = auto()


class LegSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OptionType(str, Enum):
    CALL = "CE"
    PUT = "PE"


class StrategyType(Enum):
    NONE = auto()
    STRANGLE = auto()
    BULL_PUT_SPREAD = auto()
    BEAR_CALL_SPREAD = auto()
    BIASED_STRANGLE = auto()


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
    strategy_type: StrategyType = StrategyType.NONE
    opened_at: Optional[datetime] = None


@dataclass
class TradeAction:
    action_type: str
    strategy_type: StrategyType
    legs_to_open: List[OptionLeg]
    legs_to_close: List[OptionLeg]
    reason: str


@dataclass
class MarketSnapshot:
    symbol: str
    spot: float
    candles_5m: List[dict]
    yesterday_high: float
    yesterday_low: float
    yesterday_close: float
    india_vix: float
    now: datetime


@dataclass
class TrendContext:
    trend_side: TrendSide
    bull_score: int
    bear_score: int
    confidence: int


@dataclass
class RiskConfig:
    max_intraday_loss: float
    intraday_target: float
    allow_carry_forward: bool
    max_carry_days: int
    vix_carry_threshold: float
    last_entry_time: time
    force_exit_time: time
    max_daily_loss_pct: float
    max_daily_trades: int
    per_leg_sl_mult: float = 0.0
    per_leg_tp_mult: float = 0.0
    partial_target_pct: float = 0.0
    vix_adaptive_low: float = 12.0
    vix_adaptive_high: float = 18.0
    strangle_delta_low: float = 0.12
    strangle_delta_high: float = 0.22
    strangle_offset_low: float = 200.0
    strangle_offset_high: float = 120.0
    spread_short_delta_low: float = 0.22
    spread_short_delta_high: float = 0.35
    daily_target: float = 4000.0
    daily_max_loss: float = -6000.0
    max_entries_per_day: int = 3
    max_legs_per_day: int = 12
    day_mode: str = "NORMAL"


@dataclass
class ORBState:
    orb_levels: Optional["ORBLevels"] = None
    breakout_signal: Optional["ORBBreakoutSignal"] = None
    losing_side_exited: bool = False
    last_reentry_side: Optional[OptionType] = None
    last_reentry_time: Optional[datetime] = None
