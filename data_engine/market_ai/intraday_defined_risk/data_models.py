from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from enum import Enum
from typing import Any, Mapping
import json


class ValidationError(ValueError):
    """Raised when a market snapshot is incomplete or internally inconsistent."""


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


class StrategyType(str, Enum):
    BEAR_CALL_CREDIT_SPREAD = "BEAR_CALL_CREDIT_SPREAD"
    BULL_PUT_CREDIT_SPREAD = "BULL_PUT_CREDIT_SPREAD"
    CALL_DEBIT_SPREAD = "CALL_DEBIT_SPREAD"
    IRON_CONDOR = "IRON_CONDOR"
    SHORT_STRANGLE = "SHORT_STRANGLE"
    NO_TRADE = "NO_TRADE"


class RegimeLabel(str, Enum):
    DOWN_TREND = "DOWN_TREND"
    UP_TREND = "UP_TREND"
    RANGE = "RANGE"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class AdaptiveParameters:
    rv_high_cutoff: float = 0.60
    rv_mid_cutoff: float = 0.35
    directional_credit_width_ratio: float = 0.26
    relaxed_directional_credit_width_ratio: float = 0.22
    strict_directional_credit_width_ratio: float = 0.30
    condor_credit_width_ratio: float = 0.30
    candidate_widths: tuple[float, ...] = (50.0, 75.0, 100.0, 150.0)
    liquidity_spread_ratio_cap: float = 0.15
    minimum_anchor_distance_points: float = 0.0
    strong_setup_quality_threshold: float = 10.5
    weak_setup_quality_threshold: float = 7.0
    hedge_cost_ratio_cap: float = 0.90
    relaxed_hedge_cost_ratio_cap: float = 1.05
    strict_hedge_cost_ratio_cap: float = 0.75
    width_preference_penalty: float = 1.25
    shadow_sessions: int = 5
    bearish_trade_score_threshold: float = 5.8
    bullish_trade_score_threshold: float = 5.8
    bearish_trade_margin: float = 1.0
    bullish_trade_margin: float = 1.0
    failed_breakout_transition_shadow_buffer: float = 0.25

    def clamped(self) -> "AdaptiveParameters":
        mid = min(max(self.rv_mid_cutoff, 0.25), 0.75)
        high = min(max(self.rv_high_cutoff, 0.25), 0.75)
        if high < mid:
            high = mid
        return AdaptiveParameters(
            rv_high_cutoff=high,
            rv_mid_cutoff=mid,
            directional_credit_width_ratio=min(max(self.directional_credit_width_ratio, 0.15), 0.40),
            relaxed_directional_credit_width_ratio=min(max(self.relaxed_directional_credit_width_ratio, 0.15), 0.40),
            strict_directional_credit_width_ratio=min(max(self.strict_directional_credit_width_ratio, 0.15), 0.45),
            condor_credit_width_ratio=min(max(self.condor_credit_width_ratio, 0.15), 0.40),
            candidate_widths=tuple(sorted({float(width) for width in self.candidate_widths if width > 0} or {50.0, 75.0, 100.0, 150.0})),
            liquidity_spread_ratio_cap=min(max(self.liquidity_spread_ratio_cap, 0.05), 0.30),
            minimum_anchor_distance_points=max(0.0, self.minimum_anchor_distance_points),
            strong_setup_quality_threshold=max(1.0, self.strong_setup_quality_threshold),
            weak_setup_quality_threshold=max(1.0, min(self.weak_setup_quality_threshold, self.strong_setup_quality_threshold)),
            hedge_cost_ratio_cap=min(max(self.hedge_cost_ratio_cap, 0.30), 1.50),
            relaxed_hedge_cost_ratio_cap=min(max(self.relaxed_hedge_cost_ratio_cap, 0.30), 1.75),
            strict_hedge_cost_ratio_cap=min(max(self.strict_hedge_cost_ratio_cap, 0.20), 1.25),
            width_preference_penalty=min(max(self.width_preference_penalty, 0.0), 2.0),
            shadow_sessions=max(1, self.shadow_sessions),
            bearish_trade_score_threshold=min(max(self.bearish_trade_score_threshold, 2.0), 12.0),
            bullish_trade_score_threshold=min(max(self.bullish_trade_score_threshold, 2.0), 12.0),
            bearish_trade_margin=min(max(self.bearish_trade_margin, 0.5), 5.0),
            bullish_trade_margin=min(max(self.bullish_trade_margin, 0.5), 5.0),
            failed_breakout_transition_shadow_buffer=min(max(self.failed_breakout_transition_shadow_buffer, 0.0), 1.0),
        )


@dataclass(frozen=True)
class OhlcvBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def validate(self) -> None:
        if self.high <= 0 or self.low <= 0 or self.open <= 0 or self.close <= 0:
            raise ValidationError("OHLC prices must be positive.")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValidationError("OHLC bar is internally inconsistent.")
        if self.low > self.high:
            raise ValidationError("OHLC low cannot be above high.")
        if self.volume < 0:
            raise ValidationError("OHLC volume cannot be negative.")


@dataclass(frozen=True)
class OhlcvSeries:
    timeframe_minutes: int
    bars: list[OhlcvBar]

    def validate(self) -> None:
        if self.timeframe_minutes not in {5, 15}:
            raise ValidationError("Only 5-minute and 15-minute bars are supported.")
        if not self.bars:
            raise ValidationError("OHLCV series cannot be empty.")
        previous: datetime | None = None
        for bar in self.bars:
            bar.validate()
            if previous and bar.timestamp <= previous:
                raise ValidationError("OHLCV series must be strictly increasing by timestamp.")
            previous = bar.timestamp

    @property
    def latest(self) -> OhlcvBar:
        return self.bars[-1]

    @property
    def session_date(self) -> date:
        return self.latest.timestamp.date()


@dataclass(frozen=True)
class ORLevels:
    length_minutes: int
    high: float
    low: float

    def validate(self) -> None:
        if self.length_minutes not in {15, 30, 45}:
            raise ValidationError("Opening range length must be 15, 30, or 45 minutes.")
        if self.low <= 0 or self.high <= 0 or self.low >= self.high:
            raise ValidationError("Opening range bounds are invalid.")


@dataclass(frozen=True)
class OptionsContractQuote:
    strike: float
    option_type: OptionType
    bid: float
    ask: float
    ltp: float
    delta: float | None = None
    iv: float | None = None
    oi: int | None = None
    symbol: str | None = None

    def validate(self) -> None:
        if self.strike <= 0:
            raise ValidationError("Strike must be positive.")
        if self.bid < 0 or self.ask < 0 or self.ltp < 0:
            raise ValidationError("Option prices cannot be negative.")
        if self.ask and self.bid and self.ask < self.bid:
            raise ValidationError("Option ask cannot be lower than bid.")
        if self.delta is not None and abs(self.delta) > 1.0:
            raise ValidationError("Absolute option delta cannot exceed 1.0.")
        if self.iv is not None and self.iv < 0:
            raise ValidationError("Implied volatility cannot be negative.")
        if self.oi is not None and self.oi < 0:
            raise ValidationError("Open interest cannot be negative.")

    @property
    def mid_price(self) -> float | None:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        if self.ltp > 0:
            return self.ltp
        return None

    @property
    def spread_ratio(self) -> float:
        mid = self.mid_price
        if mid is None or mid <= 0:
            return float("inf")
        if self.ask <= 0 and self.bid <= 0:
            return float("inf")
        return (self.ask - self.bid) / mid


@dataclass(frozen=True)
class OptionsChainSnapshot:
    timestamp: datetime
    expiry: date
    spot: float
    quotes: list[OptionsContractQuote]
    margin_estimate_per_lot: float | None = None

    def validate(self) -> None:
        if self.spot <= 0:
            raise ValidationError("Spot must be positive.")
        if not self.quotes:
            raise ValidationError("Option chain snapshot cannot be empty.")
        for quote in self.quotes:
            quote.validate()
        if self.margin_estimate_per_lot is not None and self.margin_estimate_per_lot <= 0:
            raise ValidationError("Margin estimate per lot must be positive when supplied.")

    def quotes_for(self, option_type: OptionType) -> list[OptionsContractQuote]:
        return [quote for quote in self.quotes if quote.option_type == option_type]

    def find_quote(self, strike: float, option_type: OptionType) -> OptionsContractQuote | None:
        for quote in self.quotes:
            if quote.option_type == option_type and quote.strike == strike:
                return quote
        return None


@dataclass(frozen=True)
class AccountRiskLimits:
    max_risk_rupees_per_trade: float
    max_margin_rupees: float
    max_daily_loss_rupees: float

    def validate(self) -> None:
        if self.max_risk_rupees_per_trade <= 0:
            raise ValidationError("Per-trade risk limit must be positive.")
        if self.max_margin_rupees <= 0:
            raise ValidationError("Margin limit must be positive.")
        if self.max_daily_loss_rupees <= 0:
            raise ValidationError("Daily loss limit must be positive.")


@dataclass(frozen=True)
class AccountState:
    realised_pnl_rupees: float = 0.0
    margin_used_rupees: float = 0.0

    def validate(self) -> None:
        if self.margin_used_rupees < 0:
            raise ValidationError("Margin used cannot be negative.")


@dataclass(frozen=True)
class MarketSnapshot:
    nifty_5m: OhlcvSeries
    nifty_15m: OhlcvSeries
    option_chain: OptionsChainSnapshot
    risk_limits: AccountRiskLimits
    account_state: AccountState
    live_vwap: float | None = None
    or_levels: Mapping[int, ORLevels] | None = None
    lot_size: int = 65
    slippage_points: float = 0.0
    previous_option_chain: OptionsChainSnapshot | None = None
    previous_session_close: float | None = None
    nifty_daily: OhlcvSeries | None = None  # Last N daily bars for multi-timeframe trend context

    def validate(self) -> None:
        self.nifty_5m.validate()
        self.nifty_15m.validate()
        self.option_chain.validate()
        self.risk_limits.validate()
        self.account_state.validate()
        if self.nifty_5m.session_date != self.nifty_15m.session_date:
            raise ValidationError("5m and 15m inputs must belong to the same trading session.")
        if self.option_chain.timestamp.date() != self.nifty_5m.session_date:
            raise ValidationError("Option chain timestamp must match the index session date.")
        if self.lot_size <= 0:
            raise ValidationError("Lot size must be positive.")
        if self.slippage_points < 0:
            raise ValidationError("Slippage cannot be negative.")
        if self.live_vwap is not None and self.live_vwap <= 0:
            raise ValidationError("VWAP must be positive when supplied.")
        if self.or_levels:
            for level in self.or_levels.values():
                level.validate()
        if self.previous_option_chain is not None:
            self.previous_option_chain.validate()
            if self.previous_option_chain.timestamp.date() != self.nifty_5m.session_date:
                raise ValidationError("Previous option chain timestamp must match the index session date.")
            if self.previous_option_chain.expiry != self.option_chain.expiry:
                raise ValidationError("Previous option chain expiry must match the current chain expiry.")
            if self.previous_option_chain.timestamp >= self.option_chain.timestamp:
                raise ValidationError("Previous option chain timestamp must be older than the current chain timestamp.")
        if self.previous_session_close is not None and self.previous_session_close <= 0:
            raise ValidationError("Previous session close must be positive when supplied.")

    @property
    def timestamp(self) -> datetime:
        return self.option_chain.timestamp


@dataclass(frozen=True)
class RegimeState:
    regime: RegimeLabel
    trend_15m: str
    execution_5m: str
    ema20_15m: float | None
    ema20_slope_15m: float | None
    rv30_pct: float
    or_length_minutes: int | None
    opening_range_high: float | None
    opening_range_low: float | None
    vwap: float | None
    confidence: float
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyLeg:
    action: str
    option_type: OptionType
    strike: float
    quote: OptionsContractQuote


@dataclass(frozen=True)
class TradeStructure:
    strategy: StrategyType
    legs: list[StrategyLeg]
    credit_points: float
    width_points: float
    call_width_points: float = 0.0
    put_width_points: float = 0.0
    margin_estimate_per_lot: float | None = None
    rationale: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskAssessment:
    allowed: bool
    reasons: list[str]
    max_loss_rupees_per_lot: float
    lots: int
    projected_margin_rupees: float
    projected_margin_utilisation: float


@dataclass(frozen=True)
class DecisionOutput:
    action: str
    strategy: StrategyType
    regime: RegimeLabel
    rationale: list[str]
    confidence_score: float
    entry: dict[str, Any]
    stop_loss: dict[str, Any]
    take_profit: dict[str, Any]
    time_exit: str
    max_loss_rupees_per_lot: float
    lots: int
    legs: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["strategy"] = self.strategy.value
        payload["regime"] = self.regime.value
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_default, sort_keys=True)


@dataclass
class OpenPosition:
    structure: TradeStructure
    lots: int
    lot_size: int
    entry_time: datetime
    entry_credit_points: float
    target_value_points: float
    stop_value_points: float
    max_loss_rupees_per_lot: float
    take_profit_capture_pct: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: str
    current_value_points: float
    pnl_rupees: float


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    reasons: list[str]
    current_parameters: AdaptiveParameters
    candidate_parameters: AdaptiveParameters | None = None
    objective_before: float | None = None
    objective_after: float | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
