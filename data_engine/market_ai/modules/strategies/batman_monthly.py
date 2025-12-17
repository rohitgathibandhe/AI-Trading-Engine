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
    long_wing_distance: int = 400    # far OTM buy (CALL above / PUT below)
    short_wing_distance: int = 200   # primary sell leg distance from spot
    hedge_extra_distance: int = 800  # ultra-far hedge distance from primary short
    long_qty: int = 1
    short_qty: int = 3
    hedge_qty: int = 2
    hedge_qty_cap: int = 8
    hedge_price_cap: float = 999.0
    add_extra_hedges: bool = True
    entry_time: Tuple[int, int] = (15, 16)  # 3:16 PM IST
    tp_pct: float = 0.02    # +2% of deployed capital
    sl_pct: float = 0.025   # -2.5% of deployed capital
    target_credit_mult: float = 0.4    # exit when PnL >= 40% of credit
    max_loss_credit_mult: float = -1.2 # exit when PnL <= -1.2x credit
    time_exit_days: int = 2            # exit when <= 2 days to expiry
    shock_enabled: bool = False
    shock_day_offset: int = 0          # 0 = entry day
    shock_pct: float = 0.025           # +2.5% move
    min_hold_days: int = 10
    max_hold_days: int = 22
    deployed_capital_per_set: float = 1_000_000.0
    credit_limit_pct: float = 0.06   # max 6% credit of spot * lot
    strike_step: int = 50
    max_shift_points: int = 600      # allow shifting strikes further OTM for credit filter
    balance_move_pct: float = 0.05   # ±5% move for T+0 balance simulation
    balance_tolerance_pct: float = 0.1
    balance_max_iter: int = 10
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
    net_credit: float = 0.0
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
        step = max(1, int(getattr(self.cfg, "strike_step", 50) or 50))
        return round(strike / step) * step

    def _build_strikes(self, spot: float, shift_points: int = 0) -> Dict[str, float]:
        """
        Build strikes respecting the 400:200:800 rule with an optional outward shift
        to satisfy the credit filter. Distances are rounded to strike_step granularity.
        """
        atm = self._round_strike(spot)
        base_long = self.cfg.long_wing_distance + shift_points
        base_short = self.cfg.short_wing_distance + shift_points
        long_call = self._round_strike(atm + base_long)
        long_put = self._round_strike(atm - base_long)
        short_call = self._round_strike(atm + base_short)
        short_put = self._round_strike(atm - base_short)
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
            "shift_points": shift_points,
        }

    def _build_legs(
        self,
        strikes: Dict[str, float],
        prices: Dict[str, float],
        hedge_qty: Tuple[int, int] | None = None,
    ) -> List[BatmanLeg]:
        hedge_put_qty, hedge_call_qty = hedge_qty or (self.cfg.hedge_qty, self.cfg.hedge_qty)
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
                        qty=hedge_call_qty,
                        direction="LONG",
                        entry_price=hc_price,
                    )
                )
            if hp_price is not None and hp_price <= self.cfg.hedge_price_cap:
                legs.append(
                    BatmanLeg(
                        strike=strikes["hedge_put"],
                        option_type="PUT",
                        qty=hedge_put_qty,
                        direction="LONG",
                        entry_price=hp_price,
                    )
                )
        return legs

    def _net_credit(self, legs: List[BatmanLeg]) -> float:
        """Return net credit in rupees (premium received - paid)."""
        credit = 0.0
        for leg in legs:
            direction = -1 if leg.direction == "LONG" else 1
            credit += direction * leg.qty * self.cfg.lot_size * leg.entry_price
        return credit

    @staticmethod
    def _intrinsic(spot: float, strike: float, option_type: str) -> float:
        if option_type == "CALL":
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)

    def _scenario_pnl(self, spot: float, legs: List[BatmanLeg]) -> float:
        total = 0.0
        for leg in legs:
            scenario_price = self._intrinsic(spot, leg.strike, leg.option_type)
            direction = 1 if leg.direction == "LONG" else -1
            total += direction * leg.qty * self.cfg.lot_size * (scenario_price - leg.entry_price)
        return total

    def _balance_hedges(
        self,
        spot: float,
        strikes: Dict[str, float],
        prices: Dict[str, float],
        base_hedge_qty: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Increase far OTM hedge quantities until the instantaneous (T+0) loss
        on +5% vs -5% moves are roughly balanced. Uses intrinsic repricing
        as a conservative proxy when IV data is unavailable.
        """
        put_qty, call_qty = base_hedge_qty
        move_pct = self.cfg.balance_move_pct
        tol_pct = self.cfg.balance_tolerance_pct
        max_iter = self.cfg.balance_max_iter
        for _ in range(max_iter):
            legs = self._build_legs(strikes, prices, hedge_qty=(put_qty, call_qty))
            up_pnl = self._scenario_pnl(spot * (1 + move_pct), legs)
            down_pnl = self._scenario_pnl(spot * (1 - move_pct), legs)
            up_loss = abs(min(0.0, up_pnl))
            down_loss = abs(min(0.0, down_pnl))
            max_loss = max(up_loss, down_loss, 1.0)
            if abs(up_loss - down_loss) <= max_loss * tol_pct:
                return put_qty, call_qty
            if up_loss > down_loss:
                call_qty = min(call_qty + 1, self.cfg.hedge_qty_cap)
            else:
                put_qty = min(put_qty + 1, self.cfg.hedge_qty_cap)
        return put_qty, call_qty

    def _prepare_entry(
        self,
        entry_time: datetime,
        spot: float,
        expiry: date,
    ) -> Tuple[BatmanPosition, Dict[str, float], Dict[str, float], float]:
        """
        Apply credit filter and hedge balancing, then create the BatmanPosition.
        Returns (position, strikes, prices, net_credit).
        """
        shift = 0
        strikes = self._build_strikes(spot, shift)
        prices = self._fetch_leg_prices(entry_time.date(), expiry, strikes)
        base_hedge = (self.cfg.hedge_qty, self.cfg.hedge_qty)
        legs = self._build_legs(strikes, prices, base_hedge)
        credit_cap = spot * self.cfg.lot_size * self.cfg.credit_limit_pct

        while self._net_credit(legs) > credit_cap and shift < self.cfg.max_shift_points:
            shift += self.cfg.strike_step
            strikes = self._build_strikes(spot, shift)
            prices = self._fetch_leg_prices(entry_time.date(), expiry, strikes)
            legs = self._build_legs(strikes, prices, base_hedge)

        hedge_qty = self._balance_hedges(spot, strikes, prices, base_hedge)
        legs = self._build_legs(strikes, prices, hedge_qty)
        net_credit = self._net_credit(legs)

        expiry_dt = datetime.combine(expiry, time(15, 30))
        position = BatmanPosition(
            entry_time=entry_time,
            expiry=expiry_dt,
            deployed_capital=self.cfg.deployed_capital_per_set,
            legs=legs,
            net_credit=net_credit,
        )
        self.positions.append(position)
        return position, strikes, prices, net_credit

    def evaluate_exit(self, position: BatmanPosition, timestamp: datetime, mtm_pnl: float) -> Optional[str]:
        """
        Exit hierarchy:
          1) Target hit: >= 40% of credit
          2) Tail loss: <= -1.2x credit
          3) Time exit: <= 2 days to expiry
          4) Legacy TP/SL by deployed capital
          5) Max hold / expiry
        """
        credit = max(position.net_credit, 0.0)
        if credit > 0:
            if mtm_pnl >= self.cfg.target_credit_mult * credit:
                return "TARGET_HIT"
            if mtm_pnl <= self.cfg.max_loss_credit_mult * credit:
                return "MAX_LOSS"

        days_to_expiry = (position.expiry.date() - timestamp.date()).days
        if days_to_expiry <= max(0, int(self.cfg.time_exit_days)):
            return "TIME_EXIT"

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
            if key == "shift_points":
                continue
            opt_type = "CALL" if "call" in key else "PUT"
            prices[f"{opt_type}:{strike}"] = self._closest_price(chain, strike, opt_type)
        return prices

    def _mtm_prices(self, entry_date: date, expiry: date, strikes: Dict[str, float], day_offset: int) -> Dict[str, float]:
        target_day = entry_date + timedelta(days=day_offset)
        if target_day > expiry:
            target_day = expiry
        chain = self._fetch_chain_prices(target_day, expiry)
        prices: Dict[str, float] = {}
        for key, strike in strikes.items():
            if key == "shift_points":
                continue
            opt_type = "CALL" if "call" in key else "PUT"
            prices[f"{opt_type}:{strike}"] = self._closest_price(chain, strike, opt_type)
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
            try:
                position, strike_map, prices, net_credit = self._prepare_entry(entry_dt, spot, expiry)
            except Exception as exc:
                LOG.warning("Failed to prepare Batman entry for %s: %s", entry_dt.date(), exc)
                period += 1
                continue
            hedge_call_qty = next(
                (l.qty for l in position.legs if l.strike == strike_map["hedge_call"] and l.option_type == "CALL"), 0
            )
            hedge_put_qty = next(
                (l.qty for l in position.legs if l.strike == strike_map["hedge_put"] and l.option_type == "PUT"), 0
            )
            day = 0
            exit_reason = None
            while day <= self.cfg.max_hold_days:
                mtm_prices = self._mtm_prices(entry_dt.date(), expiry, strike_map, day)
                # Optional synthetic shock to validate stop paths
                if self.cfg.shock_enabled and day == int(self.cfg.shock_day_offset):
                    shock_spot = spot * (1.0 + float(self.cfg.shock_pct))
                    shocked_prices: Dict[str, float] = {}
                    for key, price in mtm_prices.items():
                        try:
                            opt_type, strike_txt = key.split(":")
                            strike_val = float(strike_txt)
                            intrinsic = self._intrinsic(shock_spot, strike_val, opt_type)
                            shocked_prices[key] = max(float(price or 0.0), intrinsic)
                        except Exception:
                            shocked_prices[key] = price
                    mtm_prices = shocked_prices
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
                        "hedge_call": strike_map["hedge_call"],
                        "hedge_put": strike_map["hedge_put"],
                        "hedge_call_ltp": mtm_prices.get(f"CALL:{strike_map['hedge_call']}"),
                        "hedge_put_ltp": mtm_prices.get(f"PUT:{strike_map['hedge_put']}"),
                        "long_call_entry": prices.get(f"CALL:{strike_map['long_call']}"),
                        "short_call_entry": prices.get(f"CALL:{strike_map['short_call']}"),
                        "long_put_entry": prices.get(f"PUT:{strike_map['long_put']}"),
                        "short_put_entry": prices.get(f"PUT:{strike_map['short_put']}"),
                        "hedge_call_qty": hedge_call_qty,
                        "hedge_put_qty": hedge_put_qty,
                        "qty_long": self.cfg.long_qty,
                        "qty_short": self.cfg.short_qty,
                        "net_credit": net_credit,
                        "shift_points": strike_map.get("shift_points", 0),
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
                    "hedge_call": strike_map["hedge_call"],
                    "hedge_put": strike_map["hedge_put"],
                    "hedge_call_qty": hedge_call_qty,
                    "hedge_put_qty": hedge_put_qty,
                    "net_credit": net_credit,
                    "shift_points": strike_map.get("shift_points", 0),
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
