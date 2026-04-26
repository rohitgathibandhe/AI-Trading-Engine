from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .collector import IST, _flatten_option_chain_payload
from .data_models import (
    AccountRiskLimits,
    AccountState,
    DecisionOutput,
    MarketSnapshot,
    OhlcvBar,
    OhlcvSeries,
    OptionType,
    OptionsChainSnapshot,
    OptionsContractQuote,
    OpenPosition,
)


STATE_ROOT = Path(__file__).resolve().parents[1] / "state"
DEFAULT_UNDERLYING_ID = 13
DEFAULT_UNDERLYING_SEG = "IDX_I"


def _now_ist_naive() -> datetime:
    return datetime.now(IST).replace(tzinfo=None)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _normalize_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            epoch = float(value)
            if epoch > 1_000_000_000_000:
                epoch /= 1000.0
            if epoch > 10_000_000:
                return datetime.fromtimestamp(epoch, tz=IST).replace(tzinfo=None, second=0, microsecond=0)
        except Exception:
            pass
        try:
            parsed = datetime.fromisoformat(str(value))
        except Exception:
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(IST).replace(tzinfo=None)
    return parsed.replace(second=0, microsecond=0)


def _candle_to_bar(row: dict[str, Any]) -> OhlcvBar | None:
    ts = _normalize_timestamp(row.get("timestamp") or row.get("time") or row.get("startTime"))
    if ts is None:
        return None
    close = _as_float(row.get("close") or row.get("ltp"))
    open_px = _as_float(row.get("open"), close)
    high = _as_float(row.get("high"), max(open_px, close))
    low = _as_float(row.get("low"), min(open_px, close))
    high = max(high, open_px, close)
    low = min(low, open_px, close)
    if close <= 0.0 or open_px <= 0.0 or high <= 0.0 or low <= 0.0:
        return None
    return OhlcvBar(
        timestamp=ts,
        open=open_px,
        high=high,
        low=low,
        close=close,
        volume=_as_float(row.get("volume")),
    )


def _bars_from_candles(candles: list[dict[str, Any]]) -> list[OhlcvBar]:
    bars = [_candle_to_bar(dict(row)) for row in candles if isinstance(row, dict)]
    return [bar for bar in bars if bar is not None]


def _bar_summary(bars: list[OhlcvBar], *, minimum_required: int) -> dict[str, Any]:
    return {
        "candle_count": len(bars),
        "first_timestamp": bars[0].timestamp.isoformat() if bars else None,
        "last_timestamp": bars[-1].timestamp.isoformat() if bars else None,
        "minimum_required": minimum_required,
        "timestamps_valid": all(bar.timestamp is not None for bar in bars),
    }


def _vwap_from_bars(bars: list[OhlcvBar]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for bar in bars:
        vol = max(float(bar.volume or 0.0), 0.0)
        if vol <= 0.0:
            continue
        typical = (bar.high + bar.low + bar.close) / 3.0
        numerator += typical * vol
        denominator += vol
    if denominator > 0.0:
        return numerator / denominator
    return bars[-1].close if bars else None


class LiveDataReadinessError(RuntimeError):
    """Raised when live data is present-but-insufficient for strategy evaluation."""

    def __init__(self, reason_code: str, message: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.diagnostics = dict(diagnostics or {})


def _front_expiry(dw: Any, *, underlying_id: int, underlying_seg: str, today: date) -> str:
    expiries = []
    if hasattr(dw, "get_optionchain_expirylist"):
        expiries = list(dw.get_optionchain_expirylist(underlying_seg, underlying_id) or [])
    elif hasattr(dw, "get_expiry_list") and hasattr(dw, "_coerce_expiry_list"):
        expiries = list(dw._coerce_expiry_list(dw.get_expiry_list(underlying_id, underlying_seg)) or [])
    valid: list[date] = []
    for item in expiries:
        try:
            expiry = date.fromisoformat(str(item).split("T", 1)[0])
        except Exception:
            continue
        if expiry >= today:
            valid.append(expiry)
    if not valid:
        raise RuntimeError("No future NIFTY option expiry available from Dhan.")
    return sorted(valid)[0].isoformat()


def _quote_from_row(row: dict[str, Any]) -> OptionsContractQuote | None:
    try:
        option_type = OptionType(str(row.get("option_type") or "").upper())
    except Exception:
        return None
    strike = _as_float(row.get("strike"))
    if strike <= 0.0:
        return None
    ltp = _as_float(row.get("ltp"))
    bid = _as_float(row.get("bid"), ltp)
    ask = _as_float(row.get("ask"), ltp)
    if bid <= 0.0 and ltp > 0.0:
        bid = max(ltp * 0.98, 0.05)
    if ask <= 0.0 and ltp > 0.0:
        ask = max(ltp * 1.02, bid)
    if ask < bid:
        ask = bid
    if ltp <= 0.0:
        ltp = (bid + ask) / 2.0 if bid > 0.0 and ask > 0.0 else 0.0
    if ltp <= 0.0:
        return None
    delta_raw = row.get("delta")
    iv_raw = row.get("iv")
    return OptionsContractQuote(
        strike=strike,
        option_type=option_type,
        bid=bid,
        ask=ask,
        ltp=ltp,
        delta=float(delta_raw) if delta_raw is not None and delta_raw != "" else None,
        iv=float(iv_raw) if iv_raw is not None and iv_raw != "" else None,
        oi=_as_int(row.get("oi")) if row.get("oi") is not None and row.get("oi") != "" else None,
        symbol=str(row.get("symbol")) if row.get("symbol") else None,
    )


def _previous_close(dw: Any, *, underlying_id: int, underlying_seg: str, today: date) -> float | None:
    if not hasattr(dw, "get_daily_candles"):
        return None
    start = (today - timedelta(days=10)).isoformat()
    end = (today - timedelta(days=1)).isoformat()
    candles = dw.get_daily_candles(
        underlying_id,
        underlying_seg,
        "INDEX",
        from_date=start,
        to_date=end,
    )
    bars = _bars_from_candles(candles or [])
    return bars[-1].close if bars else None


class DhanLiveMarketDataProvider:
    def __init__(self, config: dict[str, object]) -> None:
        from market_ai.dhan_wrapper import DhanWrapper

        self.config = dict(config or {})
        self.underlying_id = int(self.config.get("underlying_id") or DEFAULT_UNDERLYING_ID)
        self.underlying_seg = str(self.config.get("underlying_seg") or DEFAULT_UNDERLYING_SEG)
        self.lot_size = int(self.config.get("lot_size") or 65)
        self.slippage_points = float(self.config.get("slippage_points") or 0.5)
        self.min_5m_candles = int(self.config.get("min_5m_candles") or 5)
        self.min_15m_candles = int(self.config.get("min_15m_candles") or 1)
        self._dw = DhanWrapper(logger=logging.getLogger("intraday_defined_risk.v83"))
        http_timeout = int(self.config.get("dhan_http_timeout_sec") or 8)
        http_retries = int(self.config.get("dhan_http_retries") or 1)
        option_chain_attempts = int(self.config.get("dhan_option_chain_attempts") or 1)
        if hasattr(self._dw, "http"):
            setattr(self._dw.http, "timeout", max(3, http_timeout))
            setattr(self._dw.http, "max_retries", max(1, http_retries))
        setattr(self._dw, "option_chain_attempts", max(1, option_chain_attempts))
        self._previous_chain: OptionsChainSnapshot | None = None
        self._front_expiry_cache: tuple[date, str] | None = None
        self._previous_close_cache: tuple[date, float | None] | None = None

    def _readiness_diagnostics(
        self,
        *,
        now: datetime,
        bars_5m: list[OhlcvBar] | None = None,
        bars_15m: list[OhlcvBar] | None = None,
        spot: float | None = None,
        quotes: list[OptionsContractQuote] | None = None,
        expiry: str | None = None,
        reason: str,
    ) -> dict[str, Any]:
        bars_5m = bars_5m or []
        bars_15m = bars_15m or []
        return {
            "snapshot_status": "INSUFFICIENT_DATA",
            "reason": reason,
            "timestamp": now.isoformat(),
            "candle_count": len(bars_5m),
            "first_timestamp": bars_5m[0].timestamp.isoformat() if bars_5m else None,
            "last_timestamp": bars_5m[-1].timestamp.isoformat() if bars_5m else None,
            "timestamps_valid": all(bar.timestamp is not None for bar in bars_5m),
            "minimum_required_5m": self.min_5m_candles,
            "five_minute": _bar_summary(bars_5m, minimum_required=self.min_5m_candles),
            "fifteen_minute": _bar_summary(bars_15m, minimum_required=self.min_15m_candles),
            "spot_price": spot,
            "spot_available": bool(spot and spot > 0.0),
            "option_quote_count": len(quotes or []),
            "option_chain_available": bool(quotes),
            "expiry": expiry,
        }

    def _raise_insufficient(
        self,
        reason: str,
        *,
        now: datetime,
        bars_5m: list[OhlcvBar] | None = None,
        bars_15m: list[OhlcvBar] | None = None,
        spot: float | None = None,
        quotes: list[OptionsContractQuote] | None = None,
        expiry: str | None = None,
    ) -> None:
        diagnostics = self._readiness_diagnostics(
            now=now,
            bars_5m=bars_5m,
            bars_15m=bars_15m,
            spot=spot,
            quotes=quotes,
            expiry=expiry,
            reason=reason,
        )
        raise LiveDataReadinessError("INSUFFICIENT_DATA", reason, diagnostics)

    def _candles(self, *, interval: int, today: date) -> list[OhlcvBar]:
        candles = self._dw.get_intraday_candles(
            self.underlying_id,
            self.underlying_seg,
            "INDEX",
            interval=interval,
            from_date=today.isoformat(),
            to_date=today.isoformat(),
        )
        return _bars_from_candles(candles or [])

    def _cached_front_expiry(self, today: date) -> str:
        if self._front_expiry_cache and self._front_expiry_cache[0] == today:
            return self._front_expiry_cache[1]
        expiry = _front_expiry(
            self._dw,
            underlying_id=self.underlying_id,
            underlying_seg=self.underlying_seg,
            today=today,
        )
        self._front_expiry_cache = (today, expiry)
        return expiry

    def _cached_previous_close(self, today: date) -> float | None:
        if self._previous_close_cache and self._previous_close_cache[0] == today:
            return self._previous_close_cache[1]
        close = _previous_close(
            self._dw,
            underlying_id=self.underlying_id,
            underlying_seg=self.underlying_seg,
            today=today,
        )
        self._previous_close_cache = (today, close)
        return close

    def current_snapshot(self) -> MarketSnapshot:
        now = _now_ist_naive()
        today = now.date()
        bars_5m = self._candles(interval=5, today=today)
        bars_15m = self._candles(interval=15, today=today)
        if len(bars_5m) < self.min_5m_candles:
            self._raise_insufficient(
                "MIN_5M_CANDLES_NOT_READY",
                now=now,
                bars_5m=bars_5m,
                bars_15m=bars_15m,
            )
        if len(bars_15m) < self.min_15m_candles:
            self._raise_insufficient(
                "MIN_15M_CANDLES_NOT_READY",
                now=now,
                bars_5m=bars_5m,
                bars_15m=bars_15m,
            )

        spot = float(self._dw.get_ltp_once(self.underlying_seg, self.underlying_id) or bars_5m[-1].close)
        if spot <= 0.0:
            self._raise_insufficient(
                "SPOT_PRICE_UNAVAILABLE",
                now=now,
                bars_5m=bars_5m,
                bars_15m=bars_15m,
                spot=spot,
            )
        expiry = str(self.config.get("expiry") or "") or self._cached_front_expiry(today)
        chain_raw = self._dw.get_option_chain(self.underlying_id, self.underlying_seg, expiry)
        chain_rows = _flatten_option_chain_payload(
            chain_raw,
            timestamp=now,
            decision_time=now.strftime("%H:%M"),
            expiry=expiry,
        )
        quotes = [_quote_from_row(dict(row)) for row in chain_rows]
        quotes = [quote for quote in quotes if quote is not None]
        if not quotes:
            self._raise_insufficient(
                "OPTION_CHAIN_QUOTES_UNAVAILABLE",
                now=now,
                bars_5m=bars_5m,
                bars_15m=bars_15m,
                spot=spot,
                quotes=quotes,
                expiry=expiry,
            )

        chain = OptionsChainSnapshot(
            timestamp=now,
            expiry=date.fromisoformat(expiry),
            spot=spot,
            quotes=quotes,
            margin_estimate_per_lot=float(self.config.get("margin_estimate_per_lot") or 10_000.0),
        )
        snapshot = MarketSnapshot(
            nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
            nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=bars_15m),
            option_chain=chain,
            risk_limits=AccountRiskLimits(
                max_risk_rupees_per_trade=float(self.config.get("max_risk_rupees_per_trade") or 10_000.0),
                max_margin_rupees=float(self.config.get("max_margin_rupees") or 200_000.0),
                max_daily_loss_rupees=float(self.config.get("max_daily_loss_rupees") or 10_000.0),
            ),
            account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0),
            live_vwap=_vwap_from_bars(bars_5m),
            lot_size=self.lot_size,
            slippage_points=self.slippage_points,
            previous_option_chain=self._previous_chain,
            previous_session_close=self._cached_previous_close(today),
        )
        snapshot.validate()
        self._previous_chain = chain
        return snapshot

    def current_structure_quotes(self, position: OpenPosition) -> MarketSnapshot:
        return self.current_snapshot()


class PaperOnlyExecutor:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = dict(config or {})

    def get_positions_raw(self) -> list[dict[str, object]]:
        return []

    def enter_trade(self, decision: DecisionOutput) -> dict[str, object]:
        raise RuntimeError("PaperOnlyExecutor never places broker orders.")

    def exit_trade(self, position: OpenPosition, reason: str) -> dict[str, object]:
        raise RuntimeError("PaperOnlyExecutor never places broker orders.")
