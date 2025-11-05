# market_ai/modules/strategies/managed_strangle.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

LOG = logging.getLogger(__name__)

@dataclass
class Legs:
    ceK: float
    peK: float
    ce_px: float
    pe_px: float
    lots: int
    lot_size: int
    expiry: str

    @property
    def entry_credit(self) -> float:
        # premium received (per lot * lots)
        return (self.ce_px + self.pe_px) * self.lot_size * self.lots

# ---------- helpers ----------

def _closest_key(keys, target: float) -> str:
    return min(keys, key=lambda k: abs(float(k) - float(target)))

def _safe_row(chain: Dict[str, Any], strike: float) -> Optional[Dict[str, Any]]:
    """Return the row for strike; if missing, return nearest strike row."""
    k = str(int(round(float(strike))))
    row = chain.get(k)
    if row is not None:
        return row
    if not chain:
        return None
    nearest = _closest_key(chain.keys(), float(strike))
    return chain.get(nearest)

def _safe_ce_ltp(chain: Dict[str, Any], strike: float, default: float) -> float:
    row = _safe_row(chain, strike)
    if not row:
        return float(default)
    return float(row.get("ce", {}).get("last_price") or default)

def _safe_pe_ltp(chain: Dict[str, Any], strike: float, default: float) -> float:
    row = _safe_row(chain, strike)
    if not row:
        return float(default)
    return float(row.get("pe", {}).get("last_price") or default)

def _safe_delta(chain: Dict[str, Any], strike: float, is_call: bool, default: float = 0.0) -> float:
    row = _safe_row(chain, strike)
    if not row:
        return float(default)
    greeks = row.get("ce" if is_call else "pe", {}).get("greeks") or {}
    d = greeks.get("delta")
    try:
        return float(d) if d is not None else float(default)
    except Exception:
        return float(default)

def _find_delta_target(chain: Dict[str, Any], delta_target: float, is_call: bool) -> Optional[Tuple[float, float]]:
    """
    Pick strike closest to |delta|≈delta_target on the requested side.
    is_call=True → use CE delta (typically +ve), else PE delta (typically -ve).
    """
    best = None
    best_err = 1e9
    for sk, row in chain.items():
        greeks = row["ce"]["greeks"] if is_call else row["pe"]["greeks"]
        d = greeks.get("delta")
        if d is None:
            continue
        try:
            err = abs(abs(float(d)) - float(delta_target))
        except Exception:
            continue
        if err < best_err:
            best_err = err
            px = (row["ce"]["last_price"] if is_call else row["pe"]["last_price"]) or 0.0
            best = (float(sk), float(px))
    return best

# ---------- public API ----------

def enter_strangle(chain: Dict[str, Any], delta_target: float, lots: int, lot_size: int, expiry: str) -> Optional[Legs]:
    ce = _find_delta_target(chain, delta_target, True)
    pe = _find_delta_target(chain, delta_target, False)
    if not ce or not pe:
        LOG.error("Could not find both CE/PE targets for entry.")
        return None
    ceK, ce_px = ce
    peK, pe_px = pe
    LOG.debug("ENTRY picker → CE %.0f @ %.2f | PE %.0f @ %.2f", ceK, ce_px, peK, pe_px)
    return Legs(ceK=ceK, peK=peK, ce_px=float(ce_px), pe_px=float(pe_px),
                lots=int(lots), lot_size=int(lot_size), expiry=expiry)

def compute_payback(chain: Dict[str, Any], legs: Legs) -> Optional[float]:
    """
    Amount (₹) required to buy back both short legs now.
    Robust to missing/misaligned strikes by using nearest row.
    """
    ce_ltp = _safe_ce_ltp(chain, legs.ceK, legs.ce_px)
    pe_ltp = _safe_pe_ltp(chain, legs.peK, legs.pe_px)
    try:
        return (float(ce_ltp) + float(pe_ltp)) * legs.lot_size * legs.lots
    except Exception:
        return None

def roll_up_call(chain: Dict[str, Any], legs: Legs, delta_target: float, broker,
                 slip_pct: float = 0.02, min_jump: int = 100):
    """
    Buy back old CE, sell new CE at ~target Δ. Guard: move at least min_jump points and not same strike.
    """
    new_ce = _find_delta_target(chain, delta_target, True)
    if not new_ce:
        return None, 0.0
    newK, new_px = new_ce
    if int(round(newK)) == int(round(legs.ceK)) or abs(newK - legs.ceK) < min_jump:
        LOG.info("ROLL_UP_CALL skipped (new strike %.0f too close to/equal old %.0f).", newK, legs.ceK)
        return legs, 0.0

    old_ltp = _safe_ce_ltp(chain, legs.ceK, legs.ce_px)
    buy_cost = float(old_ltp) * legs.lot_size * legs.lots
    sell_credit = float(new_px) * (1 - slip_pct) * legs.lot_size * legs.lots

    broker.place("BUY", "CE", legs.ceK, legs.lots, float(old_ltp), legs.expiry)
    broker.place("SELL", "CE", newK,   legs.lots, float(new_px) * (1 - slip_pct), legs.expiry)

    legs.ceK, legs.ce_px = float(newK), float(new_px)
    return legs, (sell_credit - buy_cost)

def roll_down_put(chain: Dict[str, Any], legs: Legs, delta_target: float, broker,
                  slip_pct: float = 0.02, min_jump: int = 100):
    """
    Buy back old PE, sell new PE at ~target Δ. Guard: move at least min_jump points and not same strike.
    """
    new_pe = _find_delta_target(chain, delta_target, False)
    if not new_pe:
        return None, 0.0
    newK, new_px = new_pe
    if int(round(newK)) == int(round(legs.peK)) or abs(newK - legs.peK) < min_jump:
        LOG.info("ROLL_DOWN_PUT skipped (new strike %.0f too close to/equal old %.0f).", newK, legs.peK)
        return legs, 0.0

    old_ltp = _safe_pe_ltp(chain, legs.peK, legs.pe_px)
    buy_cost = float(old_ltp) * legs.lot_size * legs.lots
    sell_credit = float(new_px) * (1 - slip_pct) * legs.lot_size * legs.lots

    broker.place("BUY", "PE", legs.peK, legs.lots, float(old_ltp), legs.expiry)
    broker.place("SELL", "PE", newK,   legs.lots, float(new_px) * (1 - slip_pct), legs.expiry)

    legs.peK, legs.pe_px = float(newK), float(new_px)
    return legs, (sell_credit - buy_cost)

def roll_both_in(chain: Dict[str, Any], legs: Legs, delta_target: float, broker,
                 slip_pct: float = 0.02, min_jump: int = 100):
    l1, pnl1 = roll_up_call(chain, legs, delta_target, broker, slip_pct, min_jump)
    legs = l1 or legs
    l2, pnl2 = roll_down_put(chain, legs, delta_target, broker, slip_pct, min_jump)
    legs = l2 or legs
    return legs, (pnl1 + pnl2)

def add_hedge(chain: Dict[str, Any], legs: Legs, call_side: bool, broker,
              wing_mult: int = 3, slip_pct: float = 0.02) -> float:
    """
    Add cheap wing ~ wing_mult * 100 points away (paper implementation).
    Returns debit (positive cost).
    """
    if call_side:
        newK = float(legs.ceK + 100 * wing_mult)
        row_key = _closest_key(chain.keys(), newK)
        ltp = float(chain[row_key]["ce"]["last_price"] or 1.0)
        price = ltp * (1 + slip_pct)
        broker.place("BUY", "CE", float(row_key), legs.lots, price, legs.expiry)
        return price * legs.lot_size * legs.lots
    else:
        newK = float(legs.peK - 100 * wing_mult)
        row_key = _closest_key(chain.keys(), newK)
        ltp = float(chain[row_key]["pe"]["last_price"] or 1.0)
        price = ltp * (1 + slip_pct)
        broker.place("BUY", "PE", float(row_key), legs.lots, price, legs.expiry)
        return price * legs.lot_size * legs.lots
