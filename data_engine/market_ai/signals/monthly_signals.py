from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence
import datetime as dt

import pandas as pd

try:
    import pandas_ta as ta  # type: ignore
except ImportError:  # pragma: no cover
    ta = None


# ---- Configs ---------------------------------------------------------------


@dataclass
class MonthlyFiltersConfig:
    use_adx: bool
    use_gap: bool
    use_range_band: bool
    use_vix: bool
    adx_length: int = 14
    adx_max: float = 25.0
    max_gap_pct: float = 0.8
    range_band_min: float = 0.3
    range_band_max: float = 0.7
    min_vix: float = 12.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonthlyFiltersConfig":
        # Defaults match the provided spec
        base = {
            "use_adx": True,
            "use_gap": True,
            "use_range_band": True,
            "use_vix": False,
            "adx_length": 14,
            "adx_max": 25.0,
            "max_gap_pct": 0.8,
            "range_band_min": 0.3,
            "range_band_max": 0.7,
            "min_vix": 12.0,
        }
        merged = {**base, **(data or {})}
        return cls(**merged)


@dataclass
class UnderlyingConfig:
    security_id: int
    exchange_segment: str
    instrument: str
    intraday_timeframe: str = "15m"

    @classmethod
    def from_dict(cls, data: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> "UnderlyingConfig":
        base = defaults or {}
        merged = {**base, **(data or {})}
        return cls(
            security_id=int(merged.get("security_id", 13)),
            exchange_segment=str(merged.get("exchange_segment", "IDX_I")),
            instrument=str(merged.get("instrument", "INDEX")),
            intraday_timeframe=str(merged.get("intraday_timeframe", "15m")),
        )


@dataclass
class MonthlySignalSnapshot:
    adx: Optional[float]
    gap_pct: Optional[float]
    range_position: Optional[float]
    vix: Optional[float]
    max_body_pct: Optional[float] = None
    spot: Optional[float] = None
    candles_intraday_count: int = 0
    candles_daily_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---- Public API ------------------------------------------------------------


def build_monthly_signals(
    *,
    dhan_client: Any,
    underlying_cfg: UnderlyingConfig,
    filters_cfg: MonthlyFiltersConfig,
    spot: float,
    option_chain_df: Optional[pd.DataFrame] = None,
    now: Optional[dt.datetime] = None,
) -> MonthlySignalSnapshot:
    """Fetch candles from Dhan and compute ADX, gap, range, and proxy VIX.

    Returns None for missing fields; never fakes 0.0 when data is absent.
    """
    now = now or dt.datetime.now()

    # Daily candles: current month
    from_date = dt.date(now.year, now.month, 1)
    to_date = now.date() + dt.timedelta(days=1)
    daily_df = _fetch_daily_candles_from_dhan(
        dhan_client=dhan_client,
        cfg=underlying_cfg,
        from_date=from_date,
        to_date=to_date,
    )

    # Intraday candles for ADX
    intraday_df: Optional[pd.DataFrame] = None
    lookback_days = 5
    intraday_from = now.date() - dt.timedelta(days=lookback_days)
    intraday_to = now.date() + dt.timedelta(days=1)
    intraday_df = _fetch_intraday_candles_from_dhan(
        dhan_client=dhan_client,
        cfg=underlying_cfg,
        from_date=intraday_from,
        to_date=intraday_to,
    )

    adx: Optional[float] = None
    if intraday_df is not None and not intraday_df.empty:
        adx = _compute_adx(intraday_df, length=filters_cfg.adx_length)

    gap_pct: Optional[float] = None
    if daily_df is not None and not daily_df.empty:
        gap_pct = _compute_gap_pct(daily_df)

    range_position: Optional[float] = None
    if daily_df is not None and not daily_df.empty:
        range_position = _compute_monthly_range_position(daily_df, spot=spot)

    vix: Optional[float] = None
    if option_chain_df is not None:
        vix = _compute_proxy_vix_from_option_chain(option_chain_df)

    max_body = _compute_max_body_pct(daily_df) if daily_df is not None else None

    return MonthlySignalSnapshot(
        adx=adx,
        gap_pct=gap_pct,
        range_position=range_position,
        vix=vix,
        spot=spot,
        candles_intraday_count=0 if intraday_df is None else len(intraday_df),
        candles_daily_count=0 if daily_df is None else len(daily_df),
        max_body_pct=max_body,
    )


def check_monthly_filters(signals: MonthlySignalSnapshot, cfg: MonthlyFiltersConfig) -> List[str]:
    """Return reasons blocking entry; empty list means all enabled filters pass."""
    reasons: List[str] = []

    if cfg.use_adx:
        if signals.adx is None:
            reasons.append("ADX_MISSING")
        elif signals.adx > cfg.adx_max:
            reasons.append(f"ADX_TOO_HIGH {signals.adx:.2f}>{cfg.adx_max:.2f}")

    if cfg.use_gap:
        if signals.gap_pct is None:
            reasons.append("GAP_MISSING")
        elif abs(signals.gap_pct) > cfg.max_gap_pct:
            reasons.append(f"GAP_TOO_LARGE {signals.gap_pct:.2f}>{cfg.max_gap_pct:.2f}")

    if cfg.use_range_band:
        if signals.range_position is None:
            reasons.append("RANGE_MISSING")
        elif not (cfg.range_band_min <= signals.range_position <= cfg.range_band_max):
            reasons.append(
                f"RANGE_OUTSIDE_BAND {signals.range_position:.2f} not in "
                f"[{cfg.range_band_min:.2f},{cfg.range_band_max:.2f}]"
            )

    if cfg.use_vix:
        if signals.vix is None:
            reasons.append("VIX_MISSING")
        elif signals.vix < cfg.min_vix:
            reasons.append(f"VIX_TOO_LOW {signals.vix:.2f}<{cfg.min_vix:.2f}")

    return reasons


# ---- Dhan helpers ----------------------------------------------------------


def _fetch_daily_candles_from_dhan(
    *,
    dhan_client: Any,
    cfg: UnderlyingConfig,
    from_date: dt.date,
    to_date: dt.date,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLC from Dhan; return None on error/missing."""
    try:
        if hasattr(dhan_client, "get_daily_candles"):
            data = dhan_client.get_daily_candles(
                cfg.security_id,
                cfg.exchange_segment,
                cfg.instrument,
                from_date=from_date.isoformat(),
                to_date=to_date.isoformat(),
            )
            return _candles_list_to_df(data)
        # fallback: POST interface
        payload = {
            "securityId": cfg.security_id,
            "exchangeSegment": cfg.exchange_segment,
            "instrument": cfg.instrument,
            "expiryCode": 0,
            "oi": False,
            "fromDate": from_date.isoformat(),
            "toDate": to_date.isoformat(),
        }
        resp = dhan_client.post("/charts/historical", json=payload)
        return _candles_response_to_df(resp)
    except Exception:
        return None


def _fetch_intraday_candles_from_dhan(
    *,
    dhan_client: Any,
    cfg: UnderlyingConfig,
    from_date: dt.date,
    to_date: dt.date,
) -> Optional[pd.DataFrame]:
    """Fetch intraday OHLC from Dhan; return None on error/missing."""
    try:
        if hasattr(dhan_client, "get_intraday_candles"):
            interval = _tf_to_minutes(cfg.intraday_timeframe)
            data = dhan_client.get_intraday_candles(
                cfg.security_id,
                cfg.exchange_segment,
                cfg.instrument,
                interval=interval,
                from_date=from_date.isoformat(),
                to_date=to_date.isoformat(),
            )
            return _candles_list_to_df(data)
        payload = {
            "securityId": cfg.security_id,
            "exchangeSegment": cfg.exchange_segment,
            "instrument": cfg.instrument,
            "expiryCode": 0,
            "oi": False,
            "fromDate": from_date.isoformat(),
            "toDate": to_date.isoformat(),
        }
        resp = dhan_client.post("/charts/intraday", json=payload)
        return _candles_response_to_df(resp)
    except Exception:
        return None


def _candles_list_to_df(candles: Any) -> Optional[pd.DataFrame]:
    """Convert list-of-dict candles to DataFrame."""
    if not isinstance(candles, list) or not candles:
        return None
    df = pd.DataFrame(candles)
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        return None
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        except Exception:
            pass
    return df


def _candles_response_to_df(resp: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Convert Dhan array response to DataFrame (open/high/low/close/volume)."""
    if not isinstance(resp, dict):
        return None
    required = ("open", "high", "low", "close")
    if not all(k in resp for k in required):
        return None
    length = len(resp["open"])
    for key in required + ("volume", "timestamp"):
        if key in resp and len(resp[key]) != length:
            return None
    data = {k: resp[k] for k in required}
    if "volume" in resp:
        data["volume"] = resp["volume"]
    df = pd.DataFrame(data)
    if "timestamp" in resp:
        try:
            df.index = pd.to_datetime(resp["timestamp"], unit="s")
        except Exception:
            pass
    return df


# ---- Signal computations ----------------------------------------------------


def _tf_to_minutes(tf: str) -> int:
    try:
        tf = tf.strip().lower()
        if tf.endswith("m"):
            return int(tf[:-1])
        if tf.endswith("min"):
            return int(tf[:-3])
        if tf.endswith("h"):
            return int(float(tf[:-1]) * 60)
        return int(tf)
    except Exception:
        return 15


def _compute_adx(candles: pd.DataFrame, length: int = 14) -> Optional[float]:
    if ta is None:
        return None
    if not {"high", "low", "close"}.issubset(candles.columns):
        return None
    if len(candles) < length + 1:
        return None
    adx_df = ta.adx(high=candles["high"], low=candles["low"], close=candles["close"], length=length)
    col = f"ADX_{length}"
    if col not in adx_df.columns:
        return None
    series = adx_df[col].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _compute_gap_pct(daily: pd.DataFrame) -> Optional[float]:
    if not {"open", "close"}.issubset(daily.columns):
        return None
    if len(daily) < 2:
        return None
    prev_close = float(daily["close"].iloc[-2])
    today_open = float(daily["open"].iloc[-1])
    if prev_close == 0:
        return None
    return (today_open - prev_close) / prev_close * 100.0


def _compute_monthly_range_position(daily: pd.DataFrame, spot: float) -> Optional[float]:
    if not {"low", "high"}.issubset(daily.columns):
        return None
    if daily.empty:
        return None
    idx = daily.index
    last_date = idx[-1]
    year = last_date.year
    month = last_date.month
    month_start = dt.date(year, month, 1)
    mask = idx.date >= month_start
    month_data = daily.loc[mask]
    if month_data.empty:
        return None
    low = float(month_data["low"].min())
    high = float(month_data["high"].max())
    if high <= low:
        return None
    return (spot - low) / (high - low)


def _compute_max_body_pct(daily: pd.DataFrame) -> Optional[float]:
    if daily is None or daily.empty:
        return None
    if not {"open", "close"}.issubset(daily.columns):
        return None
    bodies: List[float] = []
    for _, row in daily.tail(3).iterrows():
        o = row.get("open")
        c = row.get("close")
        if o is None or c is None:
            continue
        ref = o if o else 1.0
        bodies.append(abs(c - o) / ref * 100.0)
    return max(bodies) if bodies else None


def _compute_proxy_vix_from_option_chain(option_chain: pd.DataFrame) -> Optional[float]:
    if option_chain is None or option_chain.empty:
        return None
    required_cols = {"strike", "ltp", "iv", "side"}
    if not required_cols.issubset(option_chain.columns):
        return None
    oc = option_chain[option_chain["ltp"] > 0].copy()
    if oc.empty:
        return None
    grouped = oc.groupby("strike")
    strike_mid_iv: List[tuple[float, float]] = []
    for strike, grp in grouped:
        iv_ce = grp.loc[grp["side"] == "CE", "iv"].mean()
        iv_pe = grp.loc[grp["side"] == "PE", "iv"].mean()
        if pd.isna(iv_ce) or pd.isna(iv_pe):
            continue
        strike_mid_iv.append((strike, float((iv_ce + iv_pe) / 2.0)))
    if not strike_mid_iv:
        return None
    strike_mid_iv.sort(key=lambda t: t[1], reverse=True)
    _, proxy_vix = strike_mid_iv[0]
    return float(proxy_vix)
