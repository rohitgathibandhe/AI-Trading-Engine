"""
Sizing utilities for options backtest.

Exports:
 - get_lot_size_from_db(default=75, db_path="market_data.db")
 - compute_margin_per_lot(entry_underly_price, lot_size, lot_cost_per_unit, margin_multiplier)
 - compute_lots_allowed(capital, margin_per_lot, max_pos_pct=None, max_margin_per_trade=None, allowed_by_cash=None)

Notes:
 - compute_margin_per_lot returns an approximate margin estimate used for short-premium trades.
 - compute_lots_allowed implements multiple safety checks:
      * capital / margin_per_lot floor
      * optional max_pos_pct (fraction of capital to risk by position)
      * optional max_margin_per_trade (absolute rupee cap)
 - DB access expects a sqlite database with table `lot_sizes(trading_symbol, lot_size, ...)`.
"""

from __future__ import annotations
import sqlite3
import os
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_DB_PATH = os.environ.get("MARKET_DATA_DB", "market_data.db")


def get_lot_size_from_db(default: int = 75, db_path: str = DEFAULT_DB_PATH, symbol: Optional[str] = "NIFTY") -> int:
    """
    Read lot size from sqlite DB table `lot_sizes` if present, otherwise return default.
    """
    try:
        if not os.path.exists(db_path):
            logger.debug("get_lot_size_from_db: db_path not found %s -> returning default %s", db_path, default)
            return default
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        # prefer trading_symbol match; try security_id fallback
        cur.execute("SELECT lot_size FROM lot_sizes WHERE trading_symbol = ? LIMIT 1", (symbol,))
        row = cur.fetchone()
        if row and row[0]:
            ls = int(row[0])
            logger.debug("get_lot_size_from_db: found lot_size %s for symbol %s", ls, symbol)
            return ls
        # fallback: try any lot_size
        cur.execute("SELECT lot_size FROM lot_sizes LIMIT 1")
        row = cur.fetchone()
        if row and row[0]:
            ls = int(row[0])
            logger.debug("get_lot_size_from_db: found fallback lot_size %s", ls)
            return ls
    except Exception as e:
        logger.debug("get_lot_size_from_db error: %s", e, exc_info=True)
    finally:
        try:
            con.close()
        except Exception:
            pass
    return default


def compute_margin_per_lot(entry_price: float,
                           lot_size: int,
                           lot_cost_per_unit: float,
                           margin_multiplier: float = 24.0) -> float:
    """
    Approximate margin required per lot for selling a 1 lot option position.
    This is a heuristic:
      margin_per_lot ~= max( lot_cost_per_unit * lot_size, entry_price * lot_size / margin_multiplier )

    Where:
      - lot_cost_per_unit is per-option premium (if you've bought or hedged)
      - entry_price is underlying index level (we use as proxy for futures/margin calc)
      - margin_multiplier is something like 24..40 depending on broker (higher -> lower margin)
    """
    try:
        premium_component = float(lot_cost_per_unit) * int(lot_size)
        notional_component = float(entry_price) * int(lot_size) / float(max(1.0, float(margin_multiplier)))
        margin_est = max(premium_component, notional_component)
        return float(margin_est)
    except Exception as e:
        logger.debug("compute_margin_per_lot error: %s", e, exc_info=True)
        # fallback safe large number to prevent overtrading
        return float(lot_cost_per_unit) * float(lot_size)


def compute_lots_allowed(capital: float,
                         margin_per_lot: float,
                         max_pos_pct: Optional[float] = None,
                         max_margin_per_trade: Optional[float] = None,
                         allow_by_cash: Optional[float] = None,
                         force_one_lot: bool = False) -> int:
    """
    Compute how many lots are allowed given capital and margin estimates.

    Parameters:
      capital: total equity available (float)
      margin_per_lot: margin required per lot (float)
      max_pos_pct: fraction (0..1) of capital allowed to use for position sizing (optional)
      max_margin_per_trade: absolute rupee cap per trade (optional)
      allow_by_cash: another cap in rupees (optional)
      force_one_lot: debugging override -> return 1 if minimum affordable

    Returns:
      integer number of lots (>=0)
    """
    try:
        if margin_per_lot <= 0:
            return 0
        # allowed by capital
        lots_by_capital = math.floor(float(capital) / float(margin_per_lot))
        # allowed by percent
        if max_pos_pct:
            allowed_by_pct = math.floor((float(capital) * float(max_pos_pct)) / float(margin_per_lot))
        else:
            allowed_by_pct = lots_by_capital

        # apply allow_by_cash cap if provided (convert rupees to lots)
        if allow_by_cash is not None:
            allowed_by_cash = math.floor(float(allow_by_cash) / float(margin_per_lot))
        else:
            allowed_by_cash = lots_by_capital

        # apply absolute margin_per_trade cap (in rupees)
        if max_margin_per_trade is not None:
            allowed_by_max_margin = math.floor(float(max_margin_per_trade) / float(margin_per_lot))
        else:
            allowed_by_max_margin = lots_by_capital

        candidates = [lots_by_capital, allowed_by_pct, allowed_by_cash, allowed_by_max_margin]
        lots = max(0, min(int(c) for c in candidates))
        if lots == 0 and force_one_lot:
            # if force_one_lot debugging mode requested and we can afford at least one lot by capital floor:
            if float(capital) >= float(margin_per_lot):
                return 1
        return int(lots)
    except Exception as e:
        logger.debug("compute_lots_allowed error: %s", e, exc_info=True)
        return 0
