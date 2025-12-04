"""
Monthly Strangle w/ Weekly Hedge – rule-based manager.

This module implements the entry/management spec you provided. It is intentionally
pure/side-effect free so it can be plugged into the live daemon or a backtest.
Integration into start_agent will be done in a follow‑up step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import List, Optional, Tuple

from .models import (
    LegSide,
    OptionLeg,
    OptionType,
    StrategyType,
    TradeAction,
    TrendSide,
)

# --------------------------------------------------------------------------- #
# Config dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class EntryFilters:
    adx_max: float = 25.0
    max_body_pct: float = 1.2  # last 3 days
    max_gap_pct: float = 0.8
    monthly_range_band: Tuple[float, float] = (0.3, 0.7)
    vix_min: float = 12.0
    vix_rising_ok: bool = True
    entry_window_start: time = time(9, 30)
    entry_window_end: time = time(13, 0)
    enforce_in_paper: bool = False
    cycle_day_min: int = 1
    cycle_day_max: int = 7
    allow_early_next_cycle: bool = True
    early_next_cycle_days: Tuple[int, int] = (5, 7)  # days before current expiry


@dataclass
class StrikeConfig:
    short_delta_band: Tuple[float, float] = (0.12, 0.18)
    roll_distance: float = 150.0
    roll_only_if_credit: bool = True
    winning_leg_ltp: float = 15.0


@dataclass
class HedgeConfig:
    enable_weekly_hedge: bool = True
    add_on_drawdown_pct: float = -0.003  # -0.3%
    min_distance: float = 300.0
    max_ltp: float = 12.0
    max_spend_pct: float = 0.30  # of total credit


@dataclass
class BasketPnLConfig:
    tp_pct: float = 0.008  # +0.8% of margin
    sl_pct: float = -0.005  # -0.5% of margin
    trail_start_pct: float = 0.006  # start trailing at +0.6%
    trail_lock_pct: float = 0.004  # lock at +0.4%
    hard_exit_days_to_expiry: int = 3
    max_hold_days: int = 21
    reentry_allowed: bool = True
    reentry_max_attempts: int = 1
    reentry_min_dte: int = 10


@dataclass
class LegRules:
    soft_tp_mult: float = 0.5
    sl_mult: float = 1.6
    delta_breach: float = 0.30


@dataclass
class MonthlyStrangleConfig:
    entry: EntryFilters = field(default_factory=EntryFilters)
    strikes: StrikeConfig = field(default_factory=StrikeConfig)
    hedges: HedgeConfig = field(default_factory=HedgeConfig)
    basket: BasketPnLConfig = field(default_factory=BasketPnLConfig)
    legs: LegRules = field(default_factory=LegRules)


# --------------------------------------------------------------------------- #
# Helpers and core logic
# --------------------------------------------------------------------------- #


def _in_entry_window(now: datetime, cfg: EntryFilters, trade_mode: str = "live") -> bool:
    if trade_mode == "paper" and not cfg.enforce_in_paper:
        return True
    return cfg.entry_window_start <= now.time() <= cfg.entry_window_end


def _within_cycle_day(entry_date: date, expiry: date, cfg: EntryFilters) -> bool:
    dte = (expiry - entry_date).days
    cycle_day = (expiry.replace(day=1) - entry_date.replace(day=1)).days  # fallback
    # Approximate: treat first 7 calendar days of the cycle as valid
    if cfg.cycle_day_min <= entry_date.day <= cfg.cycle_day_max:
        return True
    if cfg.allow_early_next_cycle and cfg.early_next_cycle_days[0] <= dte <= cfg.early_next_cycle_days[1]:
        return True
    return False


def _passes_filters(
    now: datetime,
    entry_date: date,
    expiry: date,
    adx: float,
    max_body_pct: float,
    gap_pct: float,
    monthly_range_frac: float,
    vix: float,
    vix_rising: bool,
    cfg: EntryFilters,
    trade_mode: str = "live",
) -> bool:
    if not _in_entry_window(now, cfg, trade_mode):
        return False
    if not _within_cycle_day(entry_date, expiry, cfg):
        return False
    if adx >= cfg.adx_max:
        return False
    if max_body_pct > cfg.max_body_pct:
        return False
    if abs(gap_pct) > cfg.max_gap_pct:
        return False
    if not (cfg.monthly_range_band[0] <= monthly_range_frac <= cfg.monthly_range_band[1]):
        return False
    if vix < cfg.vix_min and not (cfg.vix_rising_ok and vix_rising):
        return False
    return True


def _delta_in_band(delta: float, band: Tuple[float, float]) -> bool:
    lo, hi = band
    if delta is None:
        return False
    try:
        val = float(delta)
    except Exception:
        return False
    return lo <= abs(val) <= hi


def pick_strikes_from_chain(
    chain: List[dict],
    short_band: Tuple[float, float],
    spot: float,
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Pick CE/PE shorts by delta band. Expects chain entries with keys:
    { 'strike', 'option_type', 'delta', 'ltp', 'expiry', 'security_id' }
    Returns (ce_short, pe_short) dicts or (None, None) if not found.
    """
    def _ltp_ok(row: dict) -> bool:
        """Guard to ensure leg has a positive LTP before selection."""
        try:
            val = row.get("ltp", 0.0)
            val = 0.0 if val is None else float(val)
            return val > 0.0
        except Exception:
            return False

    def _strike_val(row: dict) -> Optional[float]:
        try:
            return float(row.get("strike"))
        except Exception:
            return None

    ce = None
    pe = None

    for row in chain:
        if row.get("option_type") == "CE" and _delta_in_band(row.get("delta", 0.0), short_band):
            if _ltp_ok(row):
                ce = row
                break
    for row in chain:
        if row.get("option_type") == "PE" and _delta_in_band(row.get("delta", 0.0), short_band):
            if _ltp_ok(row):
                pe = row
                break

    # Fallback: distance-based selection when greeks are missing
    if ce is None:
        ce_candidates = [
            r for r in chain
            if r.get("option_type") == "CE"
            and _ltp_ok(r)
            and _strike_val(r) is not None
            and _strike_val(r) >= spot + 250
        ]
        if ce_candidates:
            ce = sorted(
                ce_candidates,
                key=lambda r: abs(_strike_val(r) - (spot + 300)),
            )[0]
    if pe is None:
        pe_candidates = [
            r for r in chain
            if r.get("option_type") == "PE"
            and _ltp_ok(r)
            and _strike_val(r) is not None
            and _strike_val(r) <= spot - 250
        ]
        if pe_candidates:
            pe = sorted(
                pe_candidates,
                key=lambda r: abs(_strike_val(r) - (spot - 300)),
            )[0]
    return ce, pe


def build_option_leg(row: dict, side: LegSide, qty: int, strategy: StrategyType) -> OptionLeg:
    return OptionLeg(
        symbol=row.get("symbol", "NIFTY"),
        expiry=row.get("expiry"),
        strike=float(row.get("strike")),
        option_type=OptionType.CALL if row.get("option_type") == "CE" else OptionType.PUT,
        side=side,
        quantity=qty,
        entry_price=float(row.get("ltp", 0.0)),
        security_id=str(row.get("security_id") or ""),
        current_ltp=float(row.get("ltp", 0.0)),
        strategy_type=strategy,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def propose_entry(
    now: datetime,
    spot: float,
    adx: float,
    max_body_pct: float,
    gap_pct: float,
    monthly_range_frac: float,
    vix: float,
    vix_rising: bool,
    option_chain: List[dict],
    expiry: date,
    lot_size: int,
    cfg: MonthlyStrangleConfig,
    trade_mode: str = "live",
) -> TradeAction:
    """
    Decide whether to open a new monthly strangle.
    Returns a TradeAction OPEN with 4 legs, or HOLD with reason.
    """
    if not _passes_filters(
        now,
        now.date(),
        expiry,
        adx,
        max_body_pct,
        gap_pct,
        monthly_range_frac,
        vix,
        vix_rising,
        cfg.entry,
        trade_mode=trade_mode,
    ):
        return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Entry filters not satisfied")

    ce_row, pe_row = pick_strikes_from_chain(option_chain, cfg.strikes.short_delta_band, spot)
    if not ce_row or not pe_row:
        return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "No strikes in delta band")

    ce_leg = build_option_leg(ce_row, LegSide.SELL, lot_size, StrategyType.STRANGLE)
    pe_leg = build_option_leg(pe_row, LegSide.SELL, lot_size, StrategyType.STRANGLE)
    legs_to_open: List[OptionLeg] = [ce_leg, pe_leg]
    credit = ce_leg.entry_price * lot_size + pe_leg.entry_price * lot_size

    # Hedge candidates: further OTM by distance
    hedge_ce = {**ce_row, "strike": ce_leg.strike + cfg.hedges.min_distance, "ltp": None}
    hedge_pe = {**pe_row, "strike": pe_leg.strike - cfg.hedges.min_distance, "ltp": None}
    if cfg.hedges.enable_weekly_hedge:
        # attach hedges with placeholder prices (actual pricing happens upstream)
        legs_to_open.append(
            OptionLeg(
                symbol=ce_leg.symbol,
                expiry=ce_leg.expiry,
                strike=hedge_ce["strike"],
                option_type=OptionType.CALL,
                side=LegSide.BUY,
                quantity=lot_size,
                entry_price=0.0,
                strategy_type=StrategyType.STRANGLE,
            )
        )
        legs_to_open.append(
            OptionLeg(
                symbol=pe_leg.symbol,
                expiry=pe_leg.expiry,
                strike=hedge_pe["strike"],
                option_type=OptionType.PUT,
                side=LegSide.BUY,
                quantity=lot_size,
                entry_price=0.0,
                strategy_type=StrategyType.STRANGLE,
            )
        )

    return TradeAction(
        action_type="OPEN",
        strategy_type=StrategyType.STRANGLE,
        legs_to_open=legs_to_open,
        legs_to_close=[],
        reason=f"Monthly strangle entry: CE {ce_leg.strike}, PE {pe_leg.strike}, credit≈{credit:.2f}",
    )


def manage_basket(
    now: datetime,
    expiry: date,
    margin_deployed: float,
    basket_mtm: float,
    legs: List[OptionLeg],
    attempt_count: int,
    cfg: MonthlyStrangleConfig,
    leg_deltas: Optional[List[float]] = None,
    adx: Optional[float] = None,
) -> TradeAction:
    """
    Manage an open strangle basket.
    Returns TradeAction with CLOSE_ALL or HOLD; per-leg rolls are left to upstream
    hedge/delta logic, but we gate exits/trails here.
    """
    dte = (expiry - now.date()).days
    tp_level = margin_deployed * cfg.basket.tp_pct
    sl_level = margin_deployed * cfg.basket.sl_pct
    trail_start = margin_deployed * cfg.basket.trail_start_pct
    trail_lock = margin_deployed * cfg.basket.trail_lock_pct

    # Hard exits
    if leg_deltas:
        safe_leg_deltas = [d for d in leg_deltas if d is not None]
        high_delta_legs = [d for d in safe_leg_deltas if abs(d) > 0.25]
        if len(high_delta_legs) >= 2:
            return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], legs, "Both legs high delta; exiting basket")
    if adx is not None and adx > 30:
        return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], legs, "ADX>30 trend escape")
    if basket_mtm <= sl_level:
        return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], legs, "Basket SL hit")
    if basket_mtm >= tp_level:
        return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], legs, "Basket TP hit")
    if dte <= cfg.basket.hard_exit_days_to_expiry:
        return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], legs, "DTE threshold reached")
    # Max hold
    if dte < 0 or dte > cfg.basket.max_hold_days:
        return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], legs, "Max hold exceeded")
    # Re-entry guard
    if attempt_count > cfg.basket.reentry_max_attempts:
        return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Re-entry attempts exhausted")
    # Trail suggestion (upstream can enforce)
    if basket_mtm >= trail_start:
        # We signal HOLD; upstream can tighten SL to trail_lock
        return TradeAction("HOLD", StrategyType.STRANGLE, [], [], f"Trail suggested lock>{trail_lock:.0f}")
    return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Hold")


def should_roll_winning_leg(now: datetime, expiry: date) -> bool:
    """Guard to block rolling the winning leg in the last 5 days to expiry."""
    dte = (expiry - now.date()).days
    return dte > 5
