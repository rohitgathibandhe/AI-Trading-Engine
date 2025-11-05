# tools/option_helpers.py
from __future__ import annotations
import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

def _mid_price(entry: Dict[str, Any]) -> float:
    """Return best mid of top_bid/top_ask or fallback to last_price."""
    try:
        bid = float(entry.get("top_bid_price", 0) or 0)
        ask = float(entry.get("top_ask_price", 0) or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return float(entry.get("last_price", 0) or 0)
    except Exception:
        return 0.0

def flatten_chain(chain_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Convert Dhan optionchain JSON to dict keyed by strike (int) with 'ce' and 'pe' entries.
    Expects chain_json like your response['data']['<expiry>'] or top-level mapping.
    """
    # data object likely contains strikes as numeric-strings. We accept either top-level structure or nested.
    # caller should pass the right portion of JSON (e.g. response['data'])
    out = {}
    # chain_json may be nested under 'data' or be the strike mapping itself.
    src = chain_json
    if "data" in chain_json and isinstance(chain_json["data"], dict):
        src = chain_json["data"]
    for k, v in src.items():
        # skip non-strike keys
        try:
            strike = int(float(k))
        except Exception:
            continue
        out[strike] = {
            "ce": v.get("ce", {}) if isinstance(v, dict) else {},
            "pe": v.get("pe", {}) if isinstance(v, dict) else {},
        }
    return out

def find_strike_by_delta(flat_chain: Dict[int, Dict[str, Any]], option_side: str, target_delta: float) -> Optional[int]:
    """
    Find strike nearest to target_delta.
    option_side: "ce" or "pe"
    target_delta: positive e.g. 0.15 (we use absolute difference)
    Returns selected strike (int) or None.
    """
    cand: List[Tuple[float, int]] = []
    for strike, d in flat_chain.items():
        opt = d.get(option_side, {})
        greeks = opt.get("greeks") or {}
        # Dhan CE delta presumably positive, PE delta may be 0 or negative — take absolute
        delta = greeks.get("delta")
        if delta is None:
            # try top-level key names
            delta = opt.get("delta") or opt.get("greeks", {}).get("delta")
        try:
            delta_f = abs(float(delta))
        except Exception:
            continue
        cand.append((abs(delta_f - abs(target_delta)), strike))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0])
    return cand[0][1]

def find_weekly_hedge(flat_chain: Dict[int, Dict[str, Any]], option_side: str, price_min: float, price_max: float) -> Optional[int]:
    """
    Prefer an option whose mid price between price_min and price_max.
    Return strike or None.
    """
    candidates = []
    for strike, d in flat_chain.items():
        opt = d.get(option_side, {})
        price = _mid_price(opt)
        if price_min <= price <= price_max:
            # prefer smallest price (closer to the range center) or choose smallest abs(strike-atm)
            candidates.append((abs(price - (price_min + price_max)/2.0), strike))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def payoff_strangle(strikes_and_premiums: Dict[str, Tuple[float, float]], underlying_range: Tuple[float, float], units:int=75):
    """
    strikes_and_premiums: dict like {"CE": (strike_ce,prem_ce), "PE": (strike_pe,prem_pe), "HEDGE": (strike_h,prem_h)}.
    Units: number of lots (units per lot is handled outside).
    Returns x (underlying levels) and y (net payoff per unit).
    """
    low, high = underlying_range
    xs = np.linspace(low, high, 800)
    y = np.zeros_like(xs)
    # short CE (sold credit) -> payoff = +prem - max(0, S-K)
    if "CE" in strikes_and_premiums:
        k_ce, prem_ce = strikes_and_premiums["CE"]
        y += prem_ce - np.maximum(0, xs - k_ce)
    # short PE -> payoff = +prem - max(0, K - S)
    if "PE" in strikes_and_premiums:
        k_pe, prem_pe = strikes_and_premiums["PE"]
        y += prem_pe - np.maximum(0, k_pe - xs)
    # long hedge (buy) adds negative premium + long payoff
    if "HEDGE" in strikes_and_premiums:
        k_h, prem_h = strikes_and_premiums["HEDGE"]
        # buying CE or PE: add long payoff minus premium (assuming it's CE if strike above atm else PE)
        # We'll add both shapes and rely on strike comparison to decide sign.
        y -= prem_h
        # If hedge strike is above central region we assume it's a CE bought
        # To be robust the caller can specify 'HEDGE_TYPE' if needed.
        # Here we add both payoffs and it will only meaningfully affect region where it matters:
        y += np.maximum(0, xs - k_h)  # long CE payoff
        y += np.maximum(0, k_h - xs)  # long PE payoff
    # return per-unit payoff. Multiply by units or lot_size outside.
    return xs, y

def plot_payoff(xs, y, strikes_and_premiums: Dict[str, Tuple[float,float]], units:int=75, title:str="Strangle payoff"):
    net = y * units
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(xs, net)
    ax.axhline(0, color="k", linewidth=0.6)
    # annotate breakevens where payoff crosses zero
    idx = np.where(np.isclose(np.sign(net[:-1]), np.sign(net[1:]))==False)[0]
    # simpler: find roots approximately
    roots = []
    for i in range(len(net)-1):
        if net[i] == 0 or net[i]*net[i+1] < 0:
            # linear interpolate
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = net[i], net[i+1]
            if (y1 - y0) != 0:
                root = x0 - y0*(x1-x0)/(y1-y0)
                roots.append(root)
    for r in roots:
        ax.axvline(r, color="gray", linestyle="--", linewidth=0.8)
        ax.text(r, max(net)*0.02, f"BE {r:.1f}", rotation=90, va="bottom", ha="right")
    # annotate strikes and premiums
    yloc = max(net)*0.6 if max(net)>0 else min(net)*0.6
    for name,(k,p) in strikes_and_premiums.items():
        ax.axvline(k, linestyle=":", linewidth=0.8)
        ax.text(k, yloc, f"{name} K={k} P={p:.2f}", rotation=90, va="bottom")
    ax.set_title(title)
    ax.set_xlabel("Underlying price")
    ax.set_ylabel(f"Net payoff x units ({units})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
