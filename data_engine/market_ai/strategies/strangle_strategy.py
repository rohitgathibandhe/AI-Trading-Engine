from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Optional

from .models import (
    LegSide,
    OptionLeg,
    OptionType,
    StrategyType,
    TradeAction,
)


@dataclass
class StrangleParams:
    delta_min: float = 0.12
    delta_max: float = 0.18
    hedge_gap: int = 300
    fallback_gap: int = 250
    hedge_cost_soft_cap: float = 4.0


class StrangleStrategyEngine:
    def __init__(self, symbol: str, params: Optional[StrangleParams] = None):
        self.symbol = symbol
        self.params = params or StrangleParams()

    def decide_strangle_trade(
        self,
        spot: float,
        option_chain: List[dict],
        expiry,
        lot_size: int,
        now: datetime,
        last_entry_time: time,
        current_positions: List[OptionLeg],
        basket_mtm: float,
        tp: float,
        sl: float,
    ) -> TradeAction:
        existing = self._find_existing_strangle(current_positions, expiry)
        if existing:
            if basket_mtm >= tp:
                return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], existing, "Basket TP hit")
            if basket_mtm <= sl:
                return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], existing, "Basket SL hit")
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Strangle open within TP/SL")

        if now.time() > last_entry_time:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Past entry cutoff")

        short_ce = self._pick_short(option_chain, expiry, option_type="CE", spot=spot)
        short_pe = self._pick_short(option_chain, expiry, option_type="PE", spot=spot)
        if not short_ce or not short_pe:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "No suitable strikes found")

        hedge_ce = self._pick_hedge(option_chain, expiry, option_type="CE", target_strike=short_ce["strike"] + self.params.hedge_gap)
        hedge_pe = self._pick_hedge(option_chain, expiry, option_type="PE", target_strike=short_pe["strike"] - self.params.hedge_gap)

        legs = [
            self._as_leg(short_ce, LegSide.SELL, lot_size),
            self._as_leg(short_pe, LegSide.SELL, lot_size),
        ]
        if hedge_ce:
            legs.append(self._as_leg(hedge_ce, LegSide.BUY, lot_size))
        if hedge_pe:
            legs.append(self._as_leg(hedge_pe, LegSide.BUY, lot_size))

        return TradeAction(
            "OPEN_STRANGLE",
            StrategyType.STRANGLE,
            legs_to_open=legs,
            legs_to_close=[],
            reason="SIDEWAYS regime: opening weekly hedged strangle",
        )

    def _find_existing_strangle(self, positions: List[OptionLeg], expiry) -> List[OptionLeg]:
        shorts = [p for p in positions if p.expiry == expiry and p.side == LegSide.SELL]
        has_ce = any(p.option_type == OptionType.CALL for p in shorts)
        has_pe = any(p.option_type == OptionType.PUT for p in shorts)
        return shorts if has_ce and has_pe else []

    def _pick_short(self, chain: List[dict], expiry, option_type: str, spot: float) -> Optional[dict]:
        # Try delta band
        filtered = []
        for row in chain:
            if row.get("expiry") != expiry:
                continue
            if str(row.get("option_type", "")).upper() not in {"CE", "PE"}:
                continue
            if option_type == "CE" and str(row.get("option_type")).upper() != "CE":
                continue
            if option_type == "PE" and str(row.get("option_type")).upper() != "PE":
                continue
            d = row.get("delta")
            if d is None:
                continue
            if option_type == "CE" and self.params.delta_min <= float(d) <= self.params.delta_max:
                filtered.append(row)
            if option_type == "PE" and (-self.params.delta_max) <= float(d) <= (-self.params.delta_min):
                filtered.append(row)
        if filtered:
            return sorted(filtered, key=lambda r: abs(abs(float(r.get("delta") or 0)) - ((self.params.delta_min + self.params.delta_max) / 2.0)))[0]
        # Fallback by distance
        target = spot + self.params.fallback_gap if option_type == "CE" else spot - self.params.fallback_gap
        return self._closest_strike(chain, expiry, option_type, target)

    def _pick_hedge(self, chain: List[dict], expiry, option_type: str, target_strike: float) -> Optional[dict]:
        return self._closest_strike(chain, expiry, option_type, target_strike)

    def _closest_strike(self, chain: List[dict], expiry, option_type: str, target: float) -> Optional[dict]:
        candidates = []
        for row in chain:
            if row.get("expiry") != expiry:
                continue
            if str(row.get("option_type", "")).upper() != ("CE" if option_type == "CE" else "PE"):
                continue
            strike = float(row.get("strike", 0))
            candidates.append((abs(strike - target), row))
        if not candidates:
            return None
        return sorted(candidates, key=lambda x: x[0])[0][1]

    def _as_leg(self, row: dict, side: LegSide, qty: int) -> OptionLeg:
        return OptionLeg(
            symbol=self.symbol,
            expiry=row.get("expiry"),
            strike=float(row.get("strike", 0)),
            option_type=OptionType.CALL if str(row.get("option_type", "")).upper() == "CE" else OptionType.PUT,
            side=side,
            quantity=qty,
            entry_price=float(row.get("ltp", 0) or 0.0),
            security_id=str(row.get("security_id") or row.get("securityId") or ""),
        )
