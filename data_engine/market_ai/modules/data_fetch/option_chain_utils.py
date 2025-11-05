from __future__ import annotations
import os
import time
import math
import logging
from typing import Any, Dict, List, Optional, Tuple
import requests
import numpy as np

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

DHAN_BASE = os.environ.get("DHAN_API_BASE", "https://api.dhan.co")
DHAN_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")
DHAN_CLIENT = os.environ.get("DHAN_CLIENT_ID", "")

HEADERS = {
    "Content-Type": "application/json",
    "access-token": DHAN_TOKEN,
    "client-id": DHAN_CLIENT,
}


def _post(path: str, payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    url = DHAN_BASE.rstrip("/") + path
    r = requests.post(url, json=payload, headers=HEADERS, timeout=timeout)
    try:
        return {"status_code": r.status_code, "body": r.json()}
    except Exception:
        return {"status_code": r.status_code, "body": r.text}


def get_expiry_list(underlying_scrip: int, seg: str = "NSE_FNO") -> Optional[List[str]]:
    """Request expiry list for underlying (returns list of YYYY-MM-DD strings)."""
    payload = {"UnderlyingScrip": int(underlying_scrip), "UnderlyingSeg": seg}
    resp = _post("/v2/optionchain/expirylist", payload)
    if resp["status_code"] != 200:
        LOG.debug("expirylist failed: %s", resp)
        return None
    body = resp["body"]
    # many versions: {'data': [...]} or data keyed by underlying id
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, list):
        return sorted(data)
    if isinstance(data, dict):
        # flatten lists in values
        for v in data.values():
            if isinstance(v, list):
                return sorted(v)
    LOG.debug("expirylist returned unexpected structure: %s", body)
    return None


def pick_nearest_expiry(expiries: List[str], prefer_weekday: Optional[int] = None) -> Optional[str]:
    """
    Pick nearest expiry >= today. If prefer_weekday specified (0=Mon..6=Sun), prefer that weekday
    if it exists among the nearest few expiries.
    """
    import pandas as pd
    if not expiries:
        return None
    now = pd.Timestamp.now().normalize()
    parsed = []
    for e in expiries:
        try:
            parsed.append((pd.to_datetime(e).normalize(), e))
        except Exception:
            continue
    parsed.sort(key=lambda x: x[0])
    # choose first >= today
    for dt_obj, raw in parsed:
        if dt_obj >= now:
            # if prefer weekday and matches, pick it
            if prefer_weekday is None:
                return raw
            if int(dt_obj.dayofweek) == int(prefer_weekday):
                return raw
            # otherwise still return first >= today as fallback
            return raw
    # fallback to last available
    return parsed[-1][1] if parsed else None


def fetch_option_chain(underlying_scrip: int, seg: str = "NSE_FNO", expiry: Optional[str] = None, use_cache: bool = False) -> Optional[Dict[str, Any]]:
    """
    Fetch option chain body['data'] for a given underlying & expiry.
    If expiry is None the function will first request expiry list and pick nearest (optionally choose weekly Tue).
    Returns the parsed JSON dict or None.
    """
    if expiry is None:
        exps = get_expiry_list(underlying_scrip, seg)
        if not exps:
            LOG.debug("No expiries found for %s/%s", underlying_scrip, seg)
            return None
        # prefer Tuesday for NIFTY weekly if available (weekday 1 = Tue)
        preferred = pick_nearest_expiry(exps, prefer_weekday=1)
        expiry = preferred or exps[0]

    payload = {"UnderlyingScrip": int(underlying_scrip), "UnderlyingSeg": seg, "Expiry": expiry}
    resp = _post("/v2/optionchain", payload)
    if resp["status_code"] != 200:
        LOG.debug("optionchain call failed %s -> %s", resp["status_code"], resp["body"])
        return None
    body = resp["body"]
    # Many Dhan responses place the chain under data or data['oc'] etc.
    data = body.get("data") if isinstance(body, dict) else None
    if data is None:
        return body if isinstance(body, dict) else None
    # canonicalize to return the inner mapping (often data['oc'] or data itself)
    if isinstance(data, dict) and "oc" in data and isinstance(data["oc"], dict):
        return data["oc"]
    if isinstance(data, dict) and any(k in data for k in ("option_chain", "optionchain", "oc")):
        for k in ("oc", "option_chain", "optionchain"):
            if k in data and isinstance(data[k], dict):
                return data[k]
    return data


def synth_vix_from_chain(chain: Dict[str, Any], strikes_window: int = 10) -> Optional[float]:
    """
    Compute a synthetic VIX-like value from chain IVs:
      - take ATM strike (closest strike with non-zero volume / ltp)
      - take +/- strikes_window strikes around ATM and compute variance from implied vol (iv^2)
      - compute weighted average variance, then annualize sqrt(variance) -> percentage IV
    This is heuristic; adjust weighting as needed.
    Returns IV in percentage (e.g., 14.3) or None.
    """
    if not chain or not isinstance(chain, dict):
        return None
    # chain keys are strike strings — convert to ordered numeric list
    try:
        strikes = sorted([float(s) for s in chain.keys()])
    except Exception:
        return None
    # determine ATM using underlying_price in any leg or pick median strike (best-effort)
    atm = None
    # try to find an entry with 'ce' or 'pe' having 'underlying_price'
    for s in strikes:
        ent = chain.get(str(int(s))) or chain.get(str(s))
        if not ent:
            continue
        for side in ("ce", "pe"):
            leg = ent.get(side)
            if isinstance(leg, dict) and "underlying_price" in leg and leg.get("underlying_price"):
                atm = float(leg["underlying_price"])
                break
        if atm:
            break
    if atm is None:
        # fallback: pick strike with highest OI sum
        max_oi = -1
        best = strikes[len(strikes)//2]
        for s in strikes:
            ent = chain.get(str(int(s))) or chain.get(str(s))
            if not ent:
                continue
            oi_sum = 0
            for side in ("ce", "pe"):
                leg = ent.get(side)
                if isinstance(leg, dict):
                    oi_sum += int(leg.get("oi", 0) or 0)
            if oi_sum > max_oi:
                max_oi = oi_sum
                best = s
        atm = best

    # find index of closest strike to atm
    idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - atm))
    lo = max(0, idx - strikes_window)
    hi = min(len(strikes), idx + strikes_window + 1)
    selected = strikes[lo:hi]
    variances = []
    weights = []
    for s in selected:
        ent = chain.get(str(int(s))) or chain.get(str(s))
        if not ent:
            continue
        # prefer average of CE & PE implied vol if present
        ivs = []
        for side in ("ce", "pe"):
            leg = ent.get(side)
            if isinstance(leg, dict):
                iv = leg.get("implied_volatility") or leg.get("iv") or leg.get("impliedVolatility")
                try:
                    iv = float(iv)
                    if iv > 0:
                        ivs.append(iv)
                except Exception:
                    continue
        if not ivs:
            continue
        iv_mid = float(sum(ivs) / len(ivs))
        # weight by sum of OI & volume (prefer liquid strikes)
        oi_sum = 0
        vol_sum = 0
        for side in ("ce", "pe"):
            leg = ent.get(side)
            if isinstance(leg, dict):
                oi_sum += int(leg.get("oi", 0) or 0)
                vol_sum += int(leg.get("volume", 0) or 0)
        w = 1.0 + math.log(1 + oi_sum + vol_sum)
        variances.append((iv_mid / 100.0) ** 2)  # convert pct->decimal and square
        weights.append(w)
    if not variances:
        return None
    # weighted average variance
    w_arr = np.array(weights, dtype=float)
    v_arr = np.array(variances, dtype=float)
    wavg_var = float((w_arr * v_arr).sum() / w_arr.sum())
    # annualize: assume IV in chain is already annualized; convert to pct
    synth_iv_pct = math.sqrt(wavg_var) * 100.0
    return round(synth_iv_pct, 3)


def find_strike_by_delta(chain: Dict[str, Any], side: str, target_delta: float = 0.15) -> Optional[Tuple[float, Dict[str, Any]]]:
    """
    Locate the strike whose option (CE/PE) delta is closest to target_delta.
    side: "CE" or "PE" (case-insensitive)
    Returns (strike, leg_dict) or None.
    """
    side = side.strip().lower()
    if side not in ("ce", "pe"):
        return None
    strikes = []
    for k, v in chain.items():
        try:
            strike = float(k)
        except Exception:
            continue
        leg = None
        if isinstance(v, dict):
            leg = v.get(side) or v.get(side.upper()) or v.get(side.lower())
        if isinstance(leg, dict):
            # many providers provide delta under greeks.delta or just delta
            delta = None
            if "greeks" in leg and isinstance(leg["greeks"], dict):
                delta = leg["greeks"].get("delta")
            delta = delta if delta is not None else leg.get("delta") or leg.get("deltaValue")
            try:
                delta_f = float(delta)
            except Exception:
                continue
            strikes.append((abs(delta_f) , strike, leg, delta_f))
    if not strikes:
        return None
    # choose the strike with minimal delta distance to target
    best = min(strikes, key=lambda t: abs(t[3] - target_delta))
    # return strike, leg
    return (best[1], best[2])


def find_weekly_hedge_candidates(chain: Dict[str, Any], max_price_low: float = 2.5, max_price_high: float = 3.5) -> List[Tuple[float, str, Dict[str, Any]]]:
    """
    Return list of (strike, side, leg) where the mid price is between [low, high].
    mid price = (top_bid_price + top_ask_price)/2 or last_price fallback.
    """
    out = []
    for k, v in chain.items():
        try:
            strike = float(k)
        except Exception:
            continue
        for side in ("ce", "pe"):
            leg = v.get(side)
            if not isinstance(leg, dict):
                continue
            bid = leg.get("top_bid_price") or leg.get("topBidPrice") or leg.get("bid")
            ask = leg.get("top_ask_price") or leg.get("topAskPrice") or leg.get("ask")
            last = leg.get("last_price") or leg.get("lastPrice") or leg.get("ltp")
            mid = None
            try:
                if bid is not None and ask is not None:
                    mid = (float(bid) + float(ask)) / 2.0
                elif last is not None:
                    mid = float(last)
            except Exception:
                continue
            if mid is None:
                continue
            if max_price_low <= mid <= max_price_high:
                out.append((strike, side.upper(), leg))
    # sort by strike distance from ATM (if ATM present)
    return sorted(out, key=lambda x: x[0])


def payoff_short_strangle_with_hedge(underlying_spot: float, short_ce_strike: float, short_pe_strike: float,
                                     short_premium_ce: float, short_premium_pe: float,
                                     long_hedge_strike: float, long_hedge_side: str, long_hedge_premium: float,
                                     contracts: int = 1, lot_size: int = 75, px_min: Optional[float] = None, px_max: Optional[float] = None):
    """
    Compute payoff arrays for portfolio:
      - short 1 CE at short_ce_strike (receive short_premium_ce)
      - short 1 PE at short_pe_strike (receive short_premium_pe)
      - long 1 (CE/PE) hedge at long_hedge_strike paying long_hedge_premium
    Returns (xs, total_payoff, components) where components is dict of individual series.
    """
    if px_min is None:
        px_min = underlying_spot * 0.7
    if px_max is None:
        px_max = underlying_spot * 1.3
    xs = np.linspace(px_min, px_max, 401)
    # single contract payoff per unit
    def call_payoff(S, K, premium):
        return np.maximum(S - K, 0.0) - premium
    def put_payoff(S, K, premium):
        return np.maximum(K - S, 0.0) - premium
    short_ce = -call_payoff(xs, short_ce_strike, short_premium_ce)
    short_pe = -put_payoff(xs, short_pe_strike, short_premium_pe)
    if long_hedge_side.strip().upper() == "CE":
        long_hedge = call_payoff(xs, long_hedge_strike, long_hedge_premium)
    else:
        long_hedge = put_payoff(xs, long_hedge_strike, long_hedge_premium)
    total = (short_ce + short_pe + long_hedge) * contracts * lot_size
    components = {"short_ce": short_ce * contracts * lot_size, "short_pe": short_pe * contracts * lot_size, "long_hedge": long_hedge * contracts * lot_size}
    return xs, total, components


# Optional plotting helper (requires matplotlib)
def plot_payoff(xs, total, components, short_ce_strike, short_pe_strike, long_hedge_strike, title: str = "Payoff"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        LOG.error("matplotlib not available; cannot plot payoff")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(xs, total, label="Total payoff", linewidth=2)
    for k, v in components.items():
        plt.plot(xs, v, linestyle="--", label=k)
    plt.axvline(short_ce_strike, color="C1", linestyle=":", label="short CE strike")
    plt.axvline(short_pe_strike, color="C2", linestyle=":", label="short PE strike")
    plt.axvline(long_hedge_strike, color="C3", linestyle=":", label="long hedge strike")
    # breakevens
    # Identify crossings where total changes sign
    sign = np.sign(total)
    idx = np.where(np.diff(sign) != 0)[0]
    for i in idx:
        be = xs[i]
        plt.scatter([be], [0], color="k")
        plt.text(be, 0, f" BE {be:.0f}", rotation=45)
    plt.title(title)
    plt.xlabel("Underlying price")
    plt.ylabel("Profit / Loss (₹)")
    plt.legend()
    plt.grid(True)
    plt.show()
