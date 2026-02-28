from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Optional, Tuple, Any
import math

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None


def _ist_now() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


def _round_strike(spot: float, step: int) -> float:
    step = max(1, int(step or 50))
    return round(spot / step) * step


@dataclass
class Leg:
    option_type: str  # "CE" / "PE"
    side: str         # "BUY" / "SELL"
    strike: float
    qty: int
    entry: float
    ltp: Optional[float] = None
    security_id: Optional[str] = None
    expiry: Optional[str] = None

    @property
    def is_long(self) -> bool:
        return self.side.upper() == "BUY"

    @property
    def entry_price(self) -> float:
        return float(self.entry or 0.0)

    @property
    def last_price(self) -> Optional[float]:
        return self.ltp

    @property
    def instrument_id(self) -> Optional[str]:
        return self.security_id

    @property
    def quantity(self) -> int:
        return int(self.qty or 0)


@dataclass
class BatmanBKMConfig:
    base_distance_points: int = 400
    inner_step_points: int = 200
    outer_step_points: int = 800
    strike_rounding: int = 50
    lot_size: int = 65
    lot_multiplier: int = 1
    max_credit_pct: float = 6.0
    credit_step_points: int = 100
    max_widen_iterations: int = 10
    balance_tolerance: float = 5000.0  # INR difference tolerance between up/down loss
    max_hedge_lots: int = 6
    tp_pct: float = 0.02
    sl_pct: float = 0.025
    entry_time: time = time(15, 16)
    exit_time: time = time(15, 10)
    payoff_range: int = 2500
    payoff_step: int = 50
    enable_balance: bool = True
    estimated_margin: float = 1_000_000.0
    min_credit_pct: float = 0.75
    min_short_distance_points: float = 450.0
    min_short_width_points: float = 1000.0
    max_center_offset_points: float = 250.0
    max_outer_distance_ratio: float = 1.80
    max_tail_loss_imbalance_abs: float = 35000.0
    max_tail_loss_imbalance_ratio: float = 0.35
    max_worst_loss_to_credit_ratio: float = 25.0
    quality_block_on_fail: bool = True


@dataclass
class BatmanBKMBasket:
    expiry: date
    legs: List[Leg]
    net_credit: float
    margin_required: float
    credit_pct: float
    entry_ts: datetime
    hedge_qty_call: int
    hedge_qty_put: int
    widened_iterations: int = 0
    exit_reason: Optional[str] = None
    exit_ts: Optional[datetime] = None
    pnl_exit: Optional[float] = None
    quality_status: str = "UNKNOWN"
    quality_score: float = 0.0
    quality_reasons: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)

    def mtm(self) -> float:
        total = 0.0
        for leg in self.legs:
            if leg.ltp is None:
                continue
            diff = (leg.ltp - leg.entry) if leg.side == "BUY" else (leg.entry - leg.ltp)
            total += diff * leg.qty * 1.0
        return total * 1.0


class BatmanBKMStrategy:
    def __init__(self, cfg: BatmanBKMConfig):
        self.cfg = cfg
        self.basket: Optional[BatmanBKMBasket] = None
        self.entered_expiries: set[date] = set()
        self.last_quality_report: Dict[str, Any] = {}

    # ── Strike and premium helpers ──────────────────────────────────────────
    def _build_strikes(self, spot: float, base_d: int) -> Dict[str, float]:
        atm = _round_strike(spot, self.cfg.strike_rounding)
        ce_buy = atm + base_d
        ce_sell = ce_buy + self.cfg.inner_step_points
        ce_hedge = ce_sell + self.cfg.outer_step_points
        pe_buy = atm - base_d
        pe_sell = pe_buy - self.cfg.inner_step_points
        pe_hedge = pe_sell - self.cfg.outer_step_points
        return {
            "ce_buy": ce_buy,
            "ce_sell": ce_sell,
            "ce_hedge": ce_hedge,
            "pe_buy": pe_buy,
            "pe_sell": pe_sell,
            "pe_hedge": pe_hedge,
            "atm": atm,
        }

    def _price_lookup(self, chain: List[Dict[str, Any]], strike: float, opt: str) -> Tuple[Optional[float], Optional[str]]:
        for row in chain:
            try:
                if str(row.get("option_type") or "").upper() != opt:
                    continue
                if abs(float(row.get("strike") or 0.0) - strike) < 1e-6:
                    ltp = row.get("ltp") or row.get("last_price") or row.get("close")
                    sec_id = row.get("security_id") or row.get("instrument_id")
                    return (None if ltp is None else float(ltp), sec_id)
            except Exception:
                continue
        return None, None

    def _build_legs(self, strikes: Dict[str, float], chain: List[Dict[str, Any]], hedge_q_ce: int, hedge_q_pe: int, expiry: Optional[date] = None) -> Optional[List[Leg]]:
        q_long = self.cfg.lot_multiplier * self.cfg.lot_size
        q_short = 3 * self.cfg.lot_multiplier * self.cfg.lot_size
        q_hedge_ce = hedge_q_ce * self.cfg.lot_size
        q_hedge_pe = hedge_q_pe * self.cfg.lot_size
        legs: List[Leg] = []
        specs = [
            ("PE", "BUY", strikes["pe_buy"], q_long),
            ("PE", "SELL", strikes["pe_sell"], q_short),
            ("PE", "BUY", strikes["pe_hedge"], q_hedge_pe),
            ("CE", "BUY", strikes["ce_buy"], q_long),
            ("CE", "SELL", strikes["ce_sell"], q_short),
            ("CE", "BUY", strikes["ce_hedge"], q_hedge_ce),
        ]
        for opt, side, strike, qty in specs:
            ltp, sec_id = self._price_lookup(chain, strike, opt)
            if ltp is None or ltp <= 0:
                return None
            legs.append(
                Leg(
                    option_type=opt,
                    side=side,
                    strike=strike,
                    qty=int(qty),
                    entry=ltp,
                    ltp=ltp,
                    security_id=sec_id,
                    expiry=expiry.isoformat() if expiry else None,
                )
            )
        return legs

    def _net_credit(self, legs: List[Leg]) -> float:
        total = 0.0
        for leg in legs:
            sign = 1 if leg.side == "SELL" else -1
            total += sign * leg.entry * leg.qty
        return total

    # ── Payoff and balancing ────────────────────────────────────────────────
    def _payoff(self, legs: List[Leg], spot: float) -> float:
        pnl = 0.0
        for leg in legs:
            intrinsic = 0.0
            if leg.option_type == "CE":
                intrinsic = max(0.0, spot - leg.strike)
            else:
                intrinsic = max(0.0, leg.strike - spot)
            value = intrinsic - leg.entry if leg.side == "BUY" else leg.entry - intrinsic
            pnl += value * leg.qty
        return pnl

    def _max_losses(self, legs: List[Leg], atm: float) -> Tuple[float, float]:
        rng = self.cfg.payoff_range
        step = self.cfg.payoff_step
        max_up = 0.0
        max_down = 0.0
        for i in range(-rng, rng + step, step):
            spot = atm + i
            pnl = self._payoff(legs, spot)
            if spot >= atm:
                max_up = min(max_up, pnl)
            else:
                max_down = min(max_down, pnl)
        return max_up, max_down

    def _extract_structure(self, legs: List[Leg]) -> Optional[Dict[str, float]]:
        ce_sells = sorted(
            [float(l.strike) for l in legs if str(l.option_type).upper() == "CE" and str(l.side).upper() == "SELL"]
        )
        pe_sells = sorted(
            [float(l.strike) for l in legs if str(l.option_type).upper() == "PE" and str(l.side).upper() == "SELL"]
        )
        ce_buys = sorted(
            [float(l.strike) for l in legs if str(l.option_type).upper() == "CE" and str(l.side).upper() == "BUY"]
        )
        pe_buys = sorted(
            [float(l.strike) for l in legs if str(l.option_type).upper() == "PE" and str(l.side).upper() == "BUY"]
        )
        if len(ce_sells) != 1 or len(pe_sells) != 1 or len(ce_buys) < 2 or len(pe_buys) < 2:
            return None
        short_call = ce_sells[0]
        short_put = pe_sells[0]
        near_call = min(ce_buys)
        far_call = max(ce_buys)
        near_put = max(pe_buys)
        far_put = min(pe_buys)
        return {
            "short_call": short_call,
            "short_put": short_put,
            "near_call": near_call,
            "far_call": far_call,
            "near_put": near_put,
            "far_put": far_put,
        }

    def assess_legs_quality(
        self,
        *,
        legs: List[Leg],
        spot: Optional[float],
        margin_required: Optional[float] = None,
        net_credit: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not legs:
            return {
                "ok": False,
                "status": "BLOCKED",
                "reason": "EMPTY_LEGS",
                "reasons": ["EMPTY_LEGS"],
                "score": 0.0,
                "metrics": {},
            }
        struct = self._extract_structure(legs)
        if not struct:
            return {
                "ok": False,
                "status": "BLOCKED",
                "reason": "INVALID_BKM_STRUCTURE",
                "reasons": ["INVALID_BKM_STRUCTURE"],
                "score": 0.0,
                "metrics": {},
            }

        short_put = float(struct["short_put"])
        short_call = float(struct["short_call"])
        center = (short_put + short_call) / 2.0
        spot_val = float(spot if spot is not None and float(spot or 0.0) > 0 else center)
        short_put_distance = max(0.0, spot_val - short_put)
        short_call_distance = max(0.0, short_call - spot_val)
        short_distance_min = min(short_put_distance, short_call_distance)
        short_width = max(0.0, short_call - short_put)
        center_offset = abs(center - spot_val)

        outer_call_distance = max(0.0, float(struct["far_call"]) - short_call)
        outer_put_distance = max(0.0, short_put - float(struct["far_put"]))
        outer_min = max(1.0, min(outer_call_distance, outer_put_distance))
        outer_max = max(outer_call_distance, outer_put_distance)
        outer_distance_ratio = outer_max / outer_min

        # Evaluate payoff tails around center to keep risk symmetry defendable.
        max_up, max_down = self._max_losses(legs, center)
        worst_loss_abs = max(abs(float(max_up)), abs(float(max_down)))
        tail_loss_imbalance_abs = abs(abs(float(max_up)) - abs(float(max_down)))
        tail_loss_imbalance_ratio = tail_loss_imbalance_abs / max(1.0, worst_loss_abs)

        margin = float(margin_required if margin_required is not None else self.cfg.estimated_margin or 0.0)
        credit = float(net_credit if net_credit is not None else self._net_credit(legs))
        credit_pct = (credit / margin) * 100.0 if margin > 0 else 0.0
        worst_loss_to_credit_ratio = worst_loss_abs / max(1.0, abs(credit))

        metrics = {
            "spot": round(spot_val, 2),
            "short_put": short_put,
            "short_call": short_call,
            "near_put": float(struct["near_put"]),
            "far_put": float(struct["far_put"]),
            "near_call": float(struct["near_call"]),
            "far_call": float(struct["far_call"]),
            "short_put_distance": round(short_put_distance, 2),
            "short_call_distance": round(short_call_distance, 2),
            "short_distance_min": round(short_distance_min, 2),
            "short_width": round(short_width, 2),
            "center": round(center, 2),
            "center_offset": round(center_offset, 2),
            "outer_put_distance": round(outer_put_distance, 2),
            "outer_call_distance": round(outer_call_distance, 2),
            "outer_distance_ratio": round(outer_distance_ratio, 3),
            "max_up_loss": round(float(max_up), 2),
            "max_down_loss": round(float(max_down), 2),
            "worst_loss_abs": round(float(worst_loss_abs), 2),
            "tail_loss_imbalance_abs": round(float(tail_loss_imbalance_abs), 2),
            "tail_loss_imbalance_ratio": round(float(tail_loss_imbalance_ratio), 3),
            "net_credit": round(float(credit), 2),
            "credit_pct": round(float(credit_pct), 3),
            "worst_loss_to_credit_ratio": round(float(worst_loss_to_credit_ratio), 2),
        }

        reasons: List[str] = []
        if credit_pct < float(self.cfg.min_credit_pct):
            reasons.append("CREDIT_TOO_LOW")
        if short_distance_min < float(self.cfg.min_short_distance_points):
            reasons.append("SHORT_DISTANCE_TOO_TIGHT")
        if short_width < float(self.cfg.min_short_width_points):
            reasons.append("SHORT_RANGE_TOO_NARROW")
        if center_offset > float(self.cfg.max_center_offset_points):
            reasons.append("CENTER_OFFSET_TOO_HIGH")
        if outer_distance_ratio > float(self.cfg.max_outer_distance_ratio):
            reasons.append("OUTER_WING_ASYMMETRY_HIGH")
        if (
            tail_loss_imbalance_abs > float(self.cfg.max_tail_loss_imbalance_abs)
            and tail_loss_imbalance_ratio > float(self.cfg.max_tail_loss_imbalance_ratio)
        ):
            reasons.append("TAIL_RISK_ASYMMETRY_HIGH")
        if worst_loss_to_credit_ratio > float(self.cfg.max_worst_loss_to_credit_ratio):
            reasons.append("WORST_LOSS_TO_CREDIT_TOO_HIGH")

        score = max(0.0, 100.0 - (18.0 * len(reasons)))
        ok = len(reasons) == 0
        return {
            "ok": bool(ok),
            "status": "PASS" if ok else "BLOCKED",
            "reason": "QUALITY_OK" if ok else reasons[0],
            "reasons": reasons if reasons else ["QUALITY_OK"],
            "score": round(score, 1),
            "metrics": metrics,
        }

    def _balance_hedges(self, legs: List[Leg], atm: float, hedge_q_ce: int, hedge_q_pe: int) -> Tuple[int, int, bool]:
        if not self.cfg.enable_balance:
            return hedge_q_ce, hedge_q_pe, True
        max_lots = max(self.cfg.max_hedge_lots, 2)
        while True:
            max_up, max_down = self._max_losses(legs, atm)
            if abs(max_up - max_down) <= self.cfg.balance_tolerance:
                return hedge_q_ce, hedge_q_pe, True
            if hedge_q_ce >= max_lots and hedge_q_pe >= max_lots:
                return hedge_q_ce, hedge_q_pe, False
            next_hedge_q_ce = hedge_q_ce
            next_hedge_q_pe = hedge_q_pe
            if max_up < max_down:
                next_hedge_q_ce = min(hedge_q_ce + 1, max_lots)
            else:
                next_hedge_q_pe = min(hedge_q_pe + 1, max_lots)
            if next_hedge_q_ce == hedge_q_ce and next_hedge_q_pe == hedge_q_pe:
                return hedge_q_ce, hedge_q_pe, False
            hedge_q_ce = next_hedge_q_ce
            hedge_q_pe = next_hedge_q_pe
            # rebuild legs with new hedge qty
            strikes = {
                "ce_buy": next(l.strike for l in legs if l.option_type == "CE" and l.side == "BUY" and l.qty == legs[0].qty),
                "ce_sell": next(l.strike for l in legs if l.option_type == "CE" and l.side == "SELL"),
                "ce_hedge": next(l.strike for l in legs if l.option_type == "CE" and l.side == "BUY" and l.qty != legs[0].qty),
                "pe_buy": next(l.strike for l in legs if l.option_type == "PE" and l.side == "BUY" and l.qty == legs[0].qty),
                "pe_sell": next(l.strike for l in legs if l.option_type == "PE" and l.side == "SELL"),
                "pe_hedge": next(l.strike for l in legs if l.option_type == "PE" and l.side == "BUY" and l.qty != legs[0].qty),
            }
            # update hedge quantities
            for leg in legs:
                if leg.option_type == "CE" and leg.side == "BUY" and leg.strike == strikes["ce_hedge"]:
                    leg.qty = hedge_q_ce * self.cfg.lot_size
                if leg.option_type == "PE" and leg.side == "BUY" and leg.strike == strikes["pe_hedge"]:
                    leg.qty = hedge_q_pe * self.cfg.lot_size

    # ── Public API ──────────────────────────────────────────────────────────
    def maybe_enter(self, spot: float, chain: List[Dict[str, Any]], expiry: date) -> Tuple[Optional[BatmanBKMBasket], str]:
        # No re-entry same expiry
        if expiry in self.entered_expiries:
            return None, "ALREADY_ENTERED"

        base_d = self.cfg.base_distance_points
        iterations = 0
        basket: Optional[BatmanBKMBasket] = None
        last_quality_reason = "QUALITY_UNKNOWN"
        while iterations <= self.cfg.max_widen_iterations:
            strikes = self._build_strikes(spot, base_d)
            legs = self._build_legs(
                strikes,
                chain,
                hedge_q_ce=2 * self.cfg.lot_multiplier,
                hedge_q_pe=2 * self.cfg.lot_multiplier,
                expiry=expiry,
            )
            if not legs:
                return None, "BAD_QUOTES"
            net_credit = self._net_credit(legs)
            margin = float(self.cfg.estimated_margin)
            credit_pct = (net_credit / margin) * 100 if margin > 0 else 0.0
            if credit_pct <= self.cfg.max_credit_pct:
                hedge_q_ce = 2 * self.cfg.lot_multiplier
                hedge_q_pe = 2 * self.cfg.lot_multiplier
                hedge_q_ce, hedge_q_pe, balanced = self._balance_hedges(legs, strikes["atm"], hedge_q_ce, hedge_q_pe)
                if not balanced:
                    return None, "UNBALANCED_PAYOFF"
                quality = self.assess_legs_quality(
                    legs=legs,
                    spot=float(spot or 0.0),
                    margin_required=margin,
                    net_credit=net_credit,
                )
                self.last_quality_report = quality
                if not bool(quality.get("ok", False)) and bool(self.cfg.quality_block_on_fail):
                    last_quality_reason = str(quality.get("reason") or "QUALITY_BLOCKED")
                    if last_quality_reason == "CREDIT_TOO_LOW":
                        return None, "CREDIT_TOO_LOW"
                    iterations += 1
                    base_d += self.cfg.credit_step_points
                    continue
                entry_ts = _ist_now()
                basket = BatmanBKMBasket(
                    expiry=expiry,
                    legs=legs,
                    net_credit=net_credit,
                    margin_required=margin,
                    credit_pct=credit_pct,
                    entry_ts=entry_ts,
                    hedge_qty_call=hedge_q_ce,
                    hedge_qty_put=hedge_q_pe,
                    widened_iterations=iterations,
                    quality_status=str(quality.get("status") or "UNKNOWN"),
                    quality_score=float(quality.get("score") or 0.0),
                    quality_reasons=list(quality.get("reasons") or []),
                    quality_metrics=dict(quality.get("metrics") or {}),
                )
                self.basket = basket
                self.entered_expiries.add(expiry)
                return basket, "ENTER"
            iterations += 1
            base_d += self.cfg.credit_step_points
        if last_quality_reason != "QUALITY_UNKNOWN":
            return None, last_quality_reason
        return None, "CREDIT_TOO_HIGH_AFTER_WIDEN"

    def update_mtm(self, chain: List[Dict[str, Any]]) -> Optional[float]:
        if not self.basket:
            return None
        for leg in self.basket.legs:
            ltp, _ = self._price_lookup(chain, leg.strike, leg.option_type)
            leg.ltp = ltp if ltp is not None else leg.ltp
        return self.basket.mtm()

    def maybe_exit(self, pnl: float, as_of: datetime) -> Optional[str]:
        if not self.basket:
            return None
        tp = self.cfg.tp_pct * self.basket.margin_required
        sl = self.cfg.sl_pct * self.basket.margin_required
        if pnl >= tp:
            self.basket.exit_reason = "TP"
        elif pnl <= -sl:
            self.basket.exit_reason = "SL"
        elif as_of.date() >= self.basket.expiry and as_of.time() >= self.cfg.exit_time:
            self.basket.exit_reason = "TIME_EXIT"
        else:
            return None
        self.basket.exit_ts = as_of
        self.basket.pnl_exit = pnl
        return self.basket.exit_reason

# ────────────────────────────────────────────────────────────────────────────
