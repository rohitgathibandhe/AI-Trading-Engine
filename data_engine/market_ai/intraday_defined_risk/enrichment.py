from __future__ import annotations

from datetime import date, datetime, time
import math


def float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_f):
        return None
    return value_f


def derive_delta_from_iv(
    *,
    spot: float,
    strike: float,
    option_type: str,
    iv: float | None,
    as_of: datetime,
    expiry: date,
    risk_free_rate: float,
) -> float | None:
    if not iv or iv <= 0 or spot <= 0 or strike <= 0:
        return None
    expiry_dt = datetime.combine(expiry, time(15, 30))
    t = max((expiry_dt - as_of).total_seconds(), 60.0) / (365.0 * 24.0 * 3600.0)
    sigma = iv / 100.0
    if sigma <= 0:
        return None
    variance_term = sigma * math.sqrt(t)
    if variance_term <= 0:
        return None
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * sigma * sigma) * t) / variance_term
    cdf = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    if option_type == "CALL":
        return max(min(cdf, 1.0), 0.0)
    if option_type == "PUT":
        return max(min(cdf - 1.0, 0.0), -1.0)
    return None


def derive_bid_ask_from_ltp(*, ltp: float, oi: float | None, iv: float | None) -> tuple[float, float]:
    if ltp <= 0:
        return 0.0, 0.0
    spread_ratio = 0.04
    if ltp < 5:
        spread_ratio += 0.08
    elif ltp < 20:
        spread_ratio += 0.04
    if oi is None or oi < 25000:
        spread_ratio += 0.06
    elif oi < 100000:
        spread_ratio += 0.03
    if iv is not None and iv > 30:
        spread_ratio += 0.02
    spread_ratio = min(max(spread_ratio, 0.03), 0.15)
    bid = max(0.05, ltp * (1.0 - (spread_ratio / 2.0)))
    ask = max(bid + 0.05, ltp * (1.0 + (spread_ratio / 2.0)))
    return round(bid, 2), round(ask, 2)
