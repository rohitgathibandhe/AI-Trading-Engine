from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BatmanSide(Enum):
    PUT = "PUT"
    CALL = "CALL"


class BatmanPhase(Enum):
    FULL = auto()
    NAKED_ROLLED = auto()
    CLOSED = auto()


@dataclass
class BatmanConfig:
    target_delta: float = 0.22
    delta_band: Tuple[float, float] = (0.20, 0.25)
    net_credit_required: bool = True
    max_entry_time: str = "10:30"  # HH:MM
    min_vix: float = 12.0
    max_gap_pct: float = 1.0
    monthly_center_band: Tuple[float, float] = (0.2, 0.8)
    lots: int = 1
    lot_size: int = 75
    capital: float = 500_000.0
    # PnL management on deployed capital
    tp_pct_deployed: float = 0.025  # +2.5% of deployed capital
    sl_pct_deployed: float = 0.02   # -2% of deployed capital
    time_exit_days: int = 2         # safety exit near expiry
    # Risk/structure guards (kept for selection only)
    max_both_side_delta: float = 0.45
    be_buffer: float = 50.0
    mtm_naked_loss_x: float = 1.8  # if naked leg LTP >= 1.8x entry
    roll_recovery_x: float = 0.5   # recover 50% of adjustment loss
    roll_hold_dte: int = 3
    one_trade_per_expiry: bool = True


@dataclass
class BatmanLeg:
    side: BatmanSide  # structure side
    opt_type: str  # "CE"/"PE"
    is_long: bool
    strike: float
    expiry: date
    quantity: int
    entry_price: float
    instrument_id: Optional[str] = None
    last_price: Optional[float] = None
    delta: Optional[float] = None

    @property
    def direction(self) -> str:
        return "BUY" if self.is_long else "SELL"

    @property
    def signed_qty(self) -> int:
        return self.quantity if self.is_long else -self.quantity


@dataclass
class BatmanState:
    created_at: datetime
    opened_at: datetime
    deployed_capital: float
    expiry: date
    long_put: BatmanLeg
    short_put_1: BatmanLeg
    short_put_2: BatmanLeg
    long_call: BatmanLeg
    short_call_1: BatmanLeg
    short_call_2: BatmanLeg
    phase: BatmanPhase = BatmanPhase.FULL
    pnl_unrealized: float = 0.0
    pnl_realized: float = 0.0
    adjustment_loss: float = 0.0
    rolled_leg: Optional[BatmanLeg] = None
    rolled_realized: float = 0.0
    just_opened: bool = True

    def all_legs(self) -> List[BatmanLeg]:
        return [
            self.long_put,
            self.short_put_1,
            self.short_put_2,
            self.long_call,
            self.short_call_1,
            self.short_call_2,
        ]

    def call_shorts(self) -> List[BatmanLeg]:
        return [self.short_call_1, self.short_call_2]

    def put_shorts(self) -> List[BatmanLeg]:
        return [self.short_put_1, self.short_put_2]

    def is_active(self) -> bool:
        return self.phase in (BatmanPhase.FULL, BatmanPhase.NAKED_ROLLED)


class BatmanStrategy:
    """
    Simplified double-ratio (paper only for now).
    Entry: 1 long + 2 shorts per side around target delta, net credit.
    Monitoring: basic MTM, delta stress, DTE exit.
    """

    def __init__(self, cfg: BatmanConfig):
        self.cfg = cfg
        self.state: Optional[BatmanState] = None
        self.notice_logged = False
        self.last_exit_at: Optional[datetime] = None
        self.last_dte_check_date: Optional[date] = None
        self.entered_expiries: set[date] = set()

    # API expected by start_agent
    def maybe_enter(
        self,
        as_of: datetime,
        spot: float,
        expiry: date,
        vix: Optional[float],
        gap_pct: float,
        monthly_range_pos: Optional[float],
        chain: List[Dict[str, Any]],
    ) -> Optional[BatmanState]:
        if self.state and self.state.is_active():
            return self.state
        if self.cfg.one_trade_per_expiry and expiry in self.entered_expiries:
            logger.info("Batman V2 entry blocked: already traded this expiry")
            return None

        hhmm = as_of.strftime("%H:%M")
        if hhmm > self.cfg.max_entry_time:
            return None
        vix_gate_enabled = self.cfg.min_vix is not None and self.cfg.min_vix > 0
        if vix_gate_enabled:
            if vix is None:
                logger.info("Batman V2 entry blocked: VIX_MISSING (min_vix=%.2f)", self.cfg.min_vix)
                return None
            if vix < self.cfg.min_vix:
                return None
        if abs(gap_pct) > self.cfg.max_gap_pct:
            return None
        lo, hi = self.cfg.monthly_center_band
        if monthly_range_pos is None:
            return None
        if not (lo <= monthly_range_pos <= hi):
            return None

        puts = [r for r in chain if (r.get("option_type") or "").upper() == "PE"]
        calls = [r for r in chain if (r.get("option_type") or "").upper() == "CE"]

        lp, sp1, sp2 = self._pick_side(puts, spot, BatmanSide.PUT)
        lc, sc1, sc2 = self._pick_side(calls, spot, BatmanSide.CALL)
        if not all([lp, sp1, sp2, lc, sc1, sc2]):
            return None

        if self.cfg.net_credit_required and not self._has_net_credit(lp, sp1, sp2, lc, sc1, sc2):
            return None

        state = BatmanState(
            created_at=as_of,
            opened_at=as_of,
            deployed_capital=self.cfg.capital,
            expiry=expiry,
            long_put=lp,
            short_put_1=sp1,
            short_put_2=sp2,
            long_call=lc,
            short_call_1=sc1,
            short_call_2=sc2,
        )
        self.state = state
        return state

    def on_tick(self, as_of: datetime, spot: float, vix: Optional[float], adx: float, chain: List[Dict[str, Any]]) -> Optional[str]:
        state = self.state
        if not state or not state.is_active():
            return None

        # Skip exit checks on the entry tick only
        if state.just_opened:
            state.just_opened = False
            return "HOLD"

        self._refresh(state, chain)
        pnl = self._compute_pnl(state)
        state.pnl_unrealized = pnl
        tp = state.deployed_capital * self.cfg.tp_pct_deployed
        sl = state.deployed_capital * self.cfg.sl_pct_deployed

        today = as_of.date()
        dte = (state.expiry - today).days

        if pnl >= tp:
            self._flatten("TP_DEPLOYED", as_of=as_of)
            return "TP_DEPLOYED"
        if pnl <= -sl:
            self._flatten("SL_DEPLOYED", as_of=as_of)
            return "SL_DEPLOYED"
        if dte <= self.cfg.time_exit_days:
            self._flatten("TIME_EXIT_SAFETY", as_of=as_of)
            return "TIME_EXIT_SAFETY"

        return "HOLD"

    def _refresh(self, state: BatmanState, chain: List[Dict[str, Any]]) -> None:
        lookup = {}
        for row in chain:
            key = (row.get("strike"), (row.get("option_type") or "").upper())
            lookup[key] = row
        for leg in state.all_legs():
            key = (leg.strike, leg.opt_type)
            row = lookup.get(key)
            if row:
                leg.last_price = row.get("ltp") or row.get("last_price") or row.get("close") or row.get("open")
                leg.delta = row.get("delta") if row.get("delta") is not None else self._synth_delta(row, leg)

    def _compute_pnl(self, state: BatmanState) -> float:
        total = 0.0
        for leg in state.all_legs():
            if leg.last_price is None:
                continue
            diff = leg.last_price - leg.entry_price
            if not leg.is_long:
                diff = -diff
            total += diff * leg.quantity * self.cfg.lot_size
        return total

    def _both_sides_delta_stressed(self, state: BatmanState) -> bool:
        call_d = [abs(l.delta) for l in state.call_shorts() if l.delta is not None]
        put_d = [abs(l.delta) for l in state.put_shorts() if l.delta is not None]
        if not call_d or not put_d:
            return False
        return max(call_d) > self.cfg.max_both_side_delta and max(put_d) > self.cfg.max_both_side_delta

    def _flatten(self, reason: str, as_of: Optional[datetime] = None) -> None:
        if self.state:
            self.state.phase = BatmanPhase.CLOSED
        exit_ts = as_of or datetime.now()
        self.last_exit_at = exit_ts
        # track expiries already traded
        try:
            if self.state:
                self.entered_expiries.add(self.state.expiry)
        except Exception:
            pass

    def _pick_side(
        self, rows: List[Dict[str, Any]], spot: float, side: BatmanSide
    ) -> Tuple[Optional[BatmanLeg], Optional[BatmanLeg], Optional[BatmanLeg]]:
        band_lo, band_hi = self.cfg.delta_band
        attempt_bands = [(band_lo, band_hi), (0.1, 0.35)]
        rows = [r for r in rows if r.get("expiry")]
        scored = []
        for band in attempt_bands:
            scored.clear()
            for r in rows:
                d = r.get("delta")
                if d is None:
                    d = self._synth_delta(r, None)
                if d is None:
                    continue
                ad = abs(d)
                if not (band[0] <= ad <= band[1]):
                    continue
                scored.append((ad, r))
            if scored:
                break
        if not scored:
            # Fallback: pick closest strikes to spot if deltas unavailable
            try:
                rows_sorted = sorted(rows, key=lambda r: abs(float(r.get("strike", 0)) - float(r.get("spot", 0))))
            except Exception:
                rows_sorted = rows
            if len(rows_sorted) < 2:
                return None, None, None
            short = rows_sorted[0]
            long = rows_sorted[1]
        scored.sort(key=lambda x: abs(x[0] - self.cfg.target_delta))
        short = scored[0][1]
        # Long further OTM: pick smallest abs delta below short
        long_rows = [r for _, r in scored if abs(r.get("delta") or self._synth_delta(r, None) or 0) < scored[0][0]]
        if not long_rows:
            return None, None, None
        long = sorted(long_rows, key=lambda r: abs((r.get("delta") or 0) - 0.1))[0]
        qty_short = self.cfg.lots * self.cfg.lot_size * 3  # 3 shorts per side
        qty_long = self.cfg.lots * self.cfg.lot_size
        leg_long = BatmanLeg(
            side=side,
            opt_type="PE" if side is BatmanSide.PUT else "CE",
            is_long=True,
            strike=float(long["strike"]),
            expiry=long["expiry"],
            quantity=qty_long,
            entry_price=float(long.get("ltp") or long.get("last_price") or long.get("close") or 0.0),
            instrument_id=long.get("security_id") or long.get("instrument_id"),
            delta=long.get("delta"),
        )
        leg_short = BatmanLeg(
            side=side,
            opt_type=leg_long.opt_type,
            is_long=False,
            strike=float(short["strike"]),
            expiry=short["expiry"],
            quantity=qty_short,
            entry_price=float(short.get("ltp") or short.get("last_price") or short.get("close") or 0.0),
            instrument_id=short.get("security_id") or short.get("instrument_id"),
            delta=short.get("delta"),
        )
        # duplicate short for 2x
        leg_short2 = BatmanLeg(**{**leg_short.__dict__})
        return leg_long, leg_short, leg_short2

    def _has_net_credit(
        self,
        lp: BatmanLeg,
        sp1: BatmanLeg,
        sp2: BatmanLeg,
        lc: BatmanLeg,
        sc1: BatmanLeg,
        sc2: BatmanLeg,
    ) -> bool:
        credit = (sp1.entry_price + sp2.entry_price + sc1.entry_price + sc2.entry_price)
        debit = (lp.entry_price + lc.entry_price)
        net = (credit - debit) * lp.quantity
        return net >= 0.0

    def _estimate_be(self, state: BatmanState) -> Tuple[float, float]:
        # crude estimate: +/- 300 from short strikes
        put_s = state.short_put_1.strike
        call_s = state.short_call_1.strike
        return put_s - 300.0, call_s + 300.0

    # --------- adjustment helpers ---------
    def _detect_challenged_side(self, state: BatmanState, spot: float, be_low: float, be_high: float) -> Optional[BatmanSide]:
        buf = self.cfg.be_buffer
        if spot < be_low - buf:
            return BatmanSide.PUT
        if spot > be_high + buf:
            return BatmanSide.CALL
        # fallback: check naked stress
        for leg in state.put_shorts():
            if leg.last_price and leg.last_price >= leg.entry_price * self.cfg.mtm_naked_loss_x:
                return BatmanSide.PUT
        for leg in state.call_shorts():
            if leg.last_price and leg.last_price >= leg.entry_price * self.cfg.mtm_naked_loss_x:
                return BatmanSide.CALL
        return None

    def _adjust_structure(self, state: BatmanState, challenged: BatmanSide, chain: List[Dict[str, Any]]) -> None:
        # exit unchallenged side
        if challenged is BatmanSide.PUT:
            legs_to_exit = state.call_shorts() + [state.long_call]
        else:
            legs_to_exit = state.put_shorts() + [state.long_put]
        for leg in legs_to_exit:
            self._exit_leg(leg, "UNCHALLENGED_EXIT")

        # book spread on challenged side
        if challenged is BatmanSide.PUT:
            long_leg = state.long_put
            short1 = state.short_put_1
            short2 = state.short_put_2
        else:
            long_leg = state.long_call
            short1 = state.short_call_1
            short2 = state.short_call_2
        pnl_spread = self._exit_pair_as_spread(long_leg, short1)
        state.adjustment_loss += pnl_spread

        # roll naked to next expiry
        rolled = self._roll_naked(short2, chain)
        state.rolled_leg = rolled
        state.phase = BatmanPhase.NAKED_ROLLED

    def _exit_pair_as_spread(self, long_leg: BatmanLeg, short_leg: BatmanLeg) -> float:
        pnl = 0.0
        if long_leg.last_price is not None:
            pnl += (long_leg.last_price - long_leg.entry_price) * long_leg.quantity * self.cfg.lot_size
        if short_leg.last_price is not None:
            pnl += (short_leg.entry_price - short_leg.last_price) * short_leg.quantity * self.cfg.lot_size
        self._exit_leg(long_leg, "SPREAD_EXIT")
        self._exit_leg(short_leg, "SPREAD_EXIT")
        return pnl

    def _roll_naked(self, naked: BatmanLeg, chain: List[Dict[str, Any]]) -> BatmanLeg:
        # close current naked
        self._exit_leg(naked, "NAKED_CLOSE")
        # pick next expiry from chain (latest expiry present)
        expiries = sorted({row.get("expiry") for row in chain if row.get("expiry")})
        next_exp = expiries[-1] if expiries else naked.expiry
        # choose strike closest delta to original (synth)
        rows = [r for r in chain if (r.get("option_type") or "").upper() == naked.opt_type and r.get("expiry") == next_exp]
        best = None
        target_delta = naked.delta if naked.delta is not None else 0.2
        best_diff = 999
        for r in rows:
            d = r.get("delta")
            if d is None:
                d = self._synth_delta(r, None)
            if d is None:
                continue
            diff = abs(abs(d) - abs(target_delta))
            if diff < best_diff:
                best_diff = diff
                best = r
        if not best:
            # fallback same strike
            best = {"strike": naked.strike, "ltp": naked.entry_price, "expiry": next_exp}
        rolled = BatmanLeg(
            side=naked.side,
            opt_type=naked.opt_type,
            is_long=False,
            strike=float(best.get("strike")),
            expiry=best.get("expiry") or next_exp,
            quantity=naked.quantity,
            entry_price=float(best.get("ltp") or best.get("last_price") or best.get("close") or naked.entry_price),
            instrument_id=best.get("security_id"),
            delta=best.get("delta"),
        )
        return rolled

    def _exit_leg(self, leg: BatmanLeg, reason: str) -> None:
        # paper-only: mark leg as closed by zeroing quantity (no broker call)
        leg.quantity = 0

    def _synth_delta(self, row: Dict[str, Any], leg: Optional[BatmanLeg]) -> Optional[float]:
        # very crude synthetic delta using moneyness; better to use real greeks
        try:
            spot = float(row.get("spot") or 0)
            strike = float(row.get("strike") or 0)
            if spot <= 0 or strike <= 0:
                return None
            m = (spot - strike) / spot
            if (row.get("option_type") or "").upper() == "CE":
                return max(0.05, min(0.95, 0.5 + m))
            else:
                return -max(0.05, min(0.95, 0.5 - m))
        except Exception:
            return None
