from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from market_ai.utils.index_calendar import batman_entry_datetime, monthly_expiry_date

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

__all__ = ["BatmanConfig", "BatmanMonthlyStrategy", "run_backtest"]


@dataclass
class BatmanConfig:
    underlying_id: int = 13  # NIFTY
    underlying_seg: str = "NSE_FNO"
    lot_size: int = 50
    long_wing_distance: int = 300   # buy 1 lot ±300 from spot
    short_wing_distance: int = 500  # sell 3 lots ±500 from spot
    long_qty: int = 1
    short_qty: int = 3
    hedge_extra_distance: int = 400
    hedge_price_cap: float = 10.0
    add_extra_hedges: bool = True
    entry_time: Tuple[int, int] = (15, 15)
    tp_pct: float = 0.025
    sl_pct: float = 0.02
    min_hold_days: int = 15
    max_hold_days: int = 18
    deployed_capital_per_set: float = 500_000.0
    market: Optional[Any] = None
    dataset: Optional[pd.DataFrame] = None
    debug: bool = False


@dataclass
class BatmanLeg:
    strike: float
    option_type: str
    qty: int
    direction: str  # "LONG" or "SHORT"
    entry_price: float = 0.0
    exit_price: Optional[float] = None

    def value(self) -> float:
        price = self.exit_price if self.exit_price is not None else self.entry_price
        return price * self.qty * (-1 if self.direction == "SHORT" else 1)


@dataclass
class BatmanPosition:
    entry_time: datetime
    expiry: datetime
    deployed_capital: float
    legs: List[BatmanLeg]
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    realized_pnl: float = 0.0
    mtm_history: List[Tuple[datetime, float]] = field(default_factory=list)

    def mark_to_market(self, timestamp: datetime, leg_prices: Dict[str, float], lot_size: int) -> float:
        pnl = 0.0
        for leg in self.legs:
            price = leg_prices.get(f"{leg.option_type}:{leg.strike}", leg.entry_price)
            direction = 1 if leg.direction == "LONG" else -1
            pnl += direction * leg.qty * lot_size * (price - leg.entry_price)
        self.mtm_history.append((timestamp, pnl))
        return pnl

    def close(self, timestamp: datetime, reason: str, leg_prices: Dict[str, float], lot_size: int) -> None:
        pnl = 0.0
        for leg in self.legs:
            price = leg_prices.get(f"{leg.option_type}:{leg.strike}", leg.entry_price)
            leg.exit_price = price
            direction = 1 if leg.direction == "LONG" else -1
            pnl += direction * leg.qty * lot_size * (price - leg.entry_price)
        self.exit_time = timestamp
        self.exit_reason = reason
        self.realized_pnl = pnl


class BatmanMonthlyStrategy:
    def __init__(self, cfg: BatmanConfig | Dict[str, Any]):
        if isinstance(cfg, dict):
            data = {k: v for k, v in cfg.items() if hasattr(BatmanConfig, k)}
            cfg = BatmanConfig(**data)
        if not cfg.market and cfg.dataset is None:
            raise ValueError("Batman strategy requires market adapter or dataset.")
        self.cfg = cfg
        self.positions: List[BatmanPosition] = []

    def _round_strike(self, strike: float) -> float:
        return round(strike / 100.0) * 100.0

    def _build_strikes(self, spot: float) -> Dict[str, float]:
        atm = self._round_strike(spot)
        long_call = self._round_strike(atm + self.cfg.long_wing_distance)
        long_put = self._round_strike(atm - self.cfg.long_wing_distance)
        short_call = self._round_strike(atm + self.cfg.short_wing_distance)
        short_put = self._round_strike(atm - self.cfg.short_wing_distance)
        hedge_call = self._round_strike(short_call + self.cfg.hedge_extra_distance)
        hedge_put = self._round_strike(short_put - self.cfg.hedge_extra_distance)
        return {
            "atm": atm,
            "long_call": long_call,
            "long_put": long_put,
            "short_call": short_call,
            "short_put": short_put,
            "hedge_call": hedge_call,
            "hedge_put": hedge_put,
        }

    def _build_legs(self, strikes: Dict[str, float], prices: Dict[str, float]) -> List[BatmanLeg]:
        legs = [
            BatmanLeg(strike=strikes["long_call"], option_type="CALL", qty=self.cfg.long_qty, direction="LONG",
                      entry_price=prices.get(f"CALL:{strikes['long_call']}", 0.0)),
            BatmanLeg(strike=strikes["short_call"], option_type="CALL", qty=self.cfg.short_qty, direction="SHORT",
                      entry_price=prices.get(f"CALL:{strikes['short_call']}", 0.0)),
            BatmanLeg(strike=strikes["long_put"], option_type="PUT", qty=self.cfg.long_qty, direction="LONG",
                      entry_price=prices.get(f"PUT:{strikes['long_put']}", 0.0)),
            BatmanLeg(strike=strikes["short_put"], option_type="PUT", qty=self.cfg.short_qty, direction="SHORT",
                      entry_price=prices.get(f"PUT:{strikes['short_put']}", 0.0)),
        ]
        if self.cfg.add_extra_hedges:
            hc_price = prices.get(f"CALL:{strikes['hedge_call']}", None)
            hp_price = prices.get(f"PUT:{strikes['hedge_put']}", None)
            if hc_price is not None and hc_price <= self.cfg.hedge_price_cap:
                legs.append(
                    BatmanLeg(
                        strike=strikes["hedge_call"],
                        option_type="CALL",
                        qty=1,
                        direction="LONG",
                        entry_price=hc_price,
                    )
                )
            if hp_price is not None and hp_price <= self.cfg.hedge_price_cap:
                legs.append(
                    BatmanLeg(
                        strike=strikes["hedge_put"],
                        option_type="PUT",
                        qty=1,
                        direction="LONG",
                        entry_price=hp_price,
                    )
                )
        return legs

    def enter_position(self, entry_time: datetime, spot: float, prices: Dict[str, float]) -> BatmanPosition:
        strikes = self._build_strikes(spot)
        legs = self._build_legs(strikes, prices)
        expiry = monthly_expiry_date(entry_time.date())
        expiry_dt = datetime.combine(expiry, time(15, 30))
        position = BatmanPosition(
            entry_time=entry_time,
            expiry=expiry_dt,
            deployed_capital=self.cfg.deployed_capital_per_set,
            legs=legs,
        )
        self.positions.append(position)
        return position

    def evaluate_exit(self, position: BatmanPosition, timestamp: datetime, mtm_pnl: float) -> Optional[str]:
        if mtm_pnl >= position.deployed_capital * self.cfg.tp_pct:
            return "TP"
        if mtm_pnl <= -position.deployed_capital * self.cfg.sl_pct:
            return "SL"
        hold_days = (timestamp.date() - position.entry_time.date()).days
        if hold_days >= self.cfg.max_hold_days:
            return "TIME"
        if timestamp.date() >= position.expiry.date():
            return "EXPIRY"
        return None

    def _fetch_chain_prices(self, entry_date: date, expiry_date: date) -> Dict[str, Dict[str, float]]:
        if not self.cfg.market:
            raise RuntimeError("Market adapter required for option prices.")
        return self.cfg.market.get_option_chain(
            underlying_id=self.cfg.underlying_id,
            expiry_or_tag=expiry_date.isoformat(),
            as_of_date=entry_date.isoformat(),
            underlying_seg=getattr(self.cfg, "underlying_seg", "NSE_FNO"),
        ) or {}

    def _spot_on_date(self, df: pd.DataFrame, target_date: date) -> Optional[float]:
        rows = df.loc[df["date"].dt.date == target_date]
        if rows.empty:
            return None
        for col in ("close", "spot", "adj_close"):
            if col in rows.columns:
                try:
                    return float(rows[col].iloc[-1])
                except Exception:
                    continue
        try:
            return float(rows.iloc[-1]["price"])
        except Exception:
            return None

    def _closest_price(self, chain: Dict[str, Dict[str, Any]], strike: float, option_type: str) -> float:
        if not chain:
            return 0.0
        if str(strike) in chain:
            legs = chain[str(strike)]
        else:
            try:
                available = [float(k) for k in chain.keys()]
                nearest = min(available, key=lambda x: abs(x - strike))
                legs = chain[str(nearest)]
            except Exception:
                return 0.0
        node = legs.get("ce" if option_type == "CALL" else "pe") or {}
        return float(node.get("last_price") or node.get("close") or 0.0)

    def _fetch_leg_prices(self, entry_date: date, expiry: date, strikes: Dict[str, float]) -> Dict[str, float]:
        chain = self._fetch_chain_prices(entry_date, expiry)
        prices: Dict[str, float] = {}
        for key, strike in strikes.items():
            if key.startswith("long_call") or key.startswith("short_call"):
                prices[f"CALL:{strike}"] = self._closest_price(chain, strike, "CALL")
            else:
                prices[f"PUT:{strike}"] = self._closest_price(chain, strike, "PUT")
        return prices

    def _mtm_prices(self, entry_date: date, expiry: date, strikes: Dict[str, float], day_offset: int) -> Dict[str, float]:
        target_day = entry_date + timedelta(days=day_offset)
        if target_day > expiry:
            target_day = expiry
        chain = self._fetch_chain_prices(target_day, expiry)
        prices: Dict[str, float] = {}
        for key, strike in strikes.items():
            if key.startswith("long_call") or key.startswith("short_call"):
                prices[f"CALL:{strike}"] = self._closest_price(chain, strike, "CALL")
            else:
                prices[f"PUT:{strike}"] = self._closest_price(chain, strike, "PUT")
        return prices

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if df is None or df.empty:
            return pd.DataFrame(), {"entries": 0}
        price_df = df.copy()
        if "date" not in price_df.columns:
            raise ValueError("Input CSV must contain a 'date' column.")
        price_df["date"] = pd.to_datetime(price_df["date"])
        start_period = price_df["date"].min().to_period("M")
        end_period = price_df["date"].max().to_period("M")
        trades: List[Dict[str, Any]] = []
        timeline: List[Dict[str, Any]] = []

        period = start_period
        while period <= end_period:
            month_anchor = period.to_timestamp()
            expiry = monthly_expiry_date(month_anchor.date())
            entry_dt = batman_entry_datetime(expiry, time(self.cfg.entry_time[0], self.cfg.entry_time[1]))
            if entry_dt.date() < price_df["date"].min().date() or entry_dt.date() > price_df["date"].max().date():
                period += 1
                continue
            spot = self._spot_on_date(price_df, entry_dt.date())
            if spot is None:
                LOG.warning("Spot not available on %s; skipping Batman entry.", entry_dt.date())
                period += 1
                continue
            strike_map = self._build_strikes(spot)
            try:
                prices = self._fetch_leg_prices(entry_dt.date(), expiry, strike_map)
            except Exception as exc:
                LOG.warning("Failed to fetch option chain for %s: %s", entry_dt.date(), exc)
                period += 1
                continue
            position = self.enter_position(entry_dt, spot, prices)
            day = 0
            exit_reason = None
            while day <= self.cfg.max_hold_days:
                mtm_prices = self._mtm_prices(entry_dt.date(), expiry, strike_map, day)
                pnl = position.mark_to_market(entry_dt + timedelta(days=day), mtm_prices, self.cfg.lot_size)
                reason = self.evaluate_exit(position, entry_dt + timedelta(days=day), pnl)
                # capture timeline row
                timeline.append(
                    {
                        "timestamp": entry_dt + timedelta(days=day),
                        "pnl": pnl,
                        "long_call": strike_map["long_call"],
                        "short_call": strike_map["short_call"],
                        "long_put": strike_map["long_put"],
                        "short_put": strike_map["short_put"],
                        "long_call_ltp": mtm_prices.get(f"CALL:{strike_map['long_call']}"),
                        "short_call_ltp": mtm_prices.get(f"CALL:{strike_map['short_call']}"),
                        "long_put_ltp": mtm_prices.get(f"PUT:{strike_map['long_put']}"),
                        "short_put_ltp": mtm_prices.get(f"PUT:{strike_map['short_put']}"),
                        "long_call_entry": prices.get(f"CALL:{strike_map['long_call']}"),
                        "short_call_entry": prices.get(f"CALL:{strike_map['short_call']}"),
                        "long_put_entry": prices.get(f"PUT:{strike_map['long_put']}"),
                        "short_put_entry": prices.get(f"PUT:{strike_map['short_put']}"),
                        "qty_long": self.cfg.long_qty,
                        "qty_short": self.cfg.short_qty,
                    }
                )
                if reason:
                    position.close(entry_dt + timedelta(days=day), reason, mtm_prices, self.cfg.lot_size)
                    exit_reason = reason
                    break
                day += 1
            trades.append(
                {
                    "entry_time": position.entry_time,
                    "exit_time": position.exit_time,
                    "exit_reason": exit_reason,
                    "expiry": position.expiry.date(),
                    "entry_spot": spot,
                    "realized_pnl": position.realized_pnl,
                    "deployed_capital": position.deployed_capital,
                    "long_call": strike_map["long_call"],
                    "short_call": strike_map["short_call"],
                    "long_put": strike_map["long_put"],
                    "short_put": strike_map["short_put"],
                }
            )
            period += 1

        summary = {
            "entries": len(trades),
            "total_pnl": sum(t.get("realized_pnl", 0.0) for t in trades)
        }
        return pd.DataFrame(trades), pd.DataFrame(timeline), summary


def run_backtest(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    cfg_clean = dict(cfg)
    cfg_clean.pop("strategy", None)
    strategy = BatmanMonthlyStrategy(cfg_clean)
    trades_df, timeline_df, summary = strategy.run(df)
    return trades_df, timeline_df, summary
