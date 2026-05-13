from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import pandas as pd

from ..utils.nifty_expiry_calendar import infer_nifty_weekly_expiry

STATE_ROOT = Path(__file__).resolve().parents[1] / "state"
DEFAULT_DATASET_ROOT = STATE_ROOT / "intraday_research_dataset_2025_01_01_2026_03_30_5m"
DEFAULT_OUTPUT_ROOT = STATE_ROOT / "weekly_defined_risk"
DEFAULT_INTRADAY_V83_PATH = STATE_ROOT / "intraday_post_tuesday_expiry_benchmark_2026-04-12_v83.json"


@dataclass(frozen=True)
class ExitProfile:
    profit_target_pct: float
    credit_stop_multiple: float
    time_exit: str

    @property
    def profile_id(self) -> str:
        return f"pt{int(self.profit_target_pct * 100)}_sl{self.credit_stop_multiple:g}_{self.time_exit.lower()}"


@dataclass
class WeeklyResearchConfig:
    dataset_root: Path = DEFAULT_DATASET_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    event_calendar_path: Path = STATE_ROOT / "events.json"
    intraday_v83_benchmark_path: Path = DEFAULT_INTRADAY_V83_PATH
    preferred_expiry_weekday: str = "Tuesday"
    lot_size: int = 75
    stable_entry_time: str = "09:45"
    large_gap_entry_time: str = "10:00"
    very_large_gap_entry_time: str = "10:15"
    gap_large_pct: float = 0.006
    gap_very_large_pct: float = 0.010
    min_history_days: int = 50
    daily_atr_period: int = 14
    weekly_atr_period: int = 8
    short_delta_range: tuple[float, float] = (0.18, 0.30)
    condor_delta_range: tuple[float, float] = (0.12, 0.22)
    widths: tuple[int, ...] = (100, 150, 200)
    min_credit_width_ratio: float = 0.12
    min_condor_credit_width_ratio: float = 0.08
    min_liquidity_score: float = 0.35
    min_reward_risk_ratio: float = 0.90
    min_broken_wing_reward_risk: float = 1.05
    min_expected_move_buffer: float = 0.70
    max_snapshot_age_minutes: int = 15
    delta_breach_threshold: float = 0.40
    max_loss_stop_fraction_for_debit: float = 0.65
    walk_forward_training_months: int = 3
    in_sample_ratio: float = 0.70
    min_strategy_sample: int = 5
    max_open_positions: int = 1
    require_fresh_option_chain: bool = True
    high_vol_iv_threshold: float = 18.0
    low_credit_iv_floor: float = 10.5
    max_gap_for_immediate_entry: float = 0.004
    large_gap_wait_minutes: int = 15
    use_observed_historical_expiry: bool = True
    allow_adjustment_research: bool = False
    max_condor_wall_distance_points: float = 300.0
    large_gap_skip_first_minutes: int = 30
    entry_start_time: str = "09:30"
    monitor_end_time: str = "15:00"
    repair_weekly_expiry_history: bool = True
    allowed_strategies: tuple[str, ...] | None = None
    allowed_regimes: tuple[str, ...] | None = None
    research_variant: str = "baseline"
    output_tag: str = field(default_factory=lambda: datetime.now().date().isoformat())

    @property
    def price_5m_path(self) -> Path:
        return self.dataset_root / "nifty_5m.csv"

    @property
    def option_chain_path(self) -> Path:
        return self.dataset_root / "options_chain.csv"


@dataclass
class StrategyCandidate:
    strategy: str
    direction: str
    regime: str
    requested_width: int
    actual_width: int
    expiry: date
    entry_timestamp: datetime
    entry_spot: float
    expected_move_points: float
    entry_cashflow_points: float
    max_profit_rupees: float
    max_loss_rupees: float
    reward_risk_ratio: float
    credit_width_ratio: float
    liquidity_score: float
    short_leg_delta: float | None
    long_leg_delta: float | None
    score: float
    rejection_reason: str | None
    notes: list[str] = field(default_factory=list)
    legs: list[dict[str, Any]] = field(default_factory=list)
    strategy_family: str = "DEFINED_RISK"
    support_level: float | None = None
    resistance_level: float | None = None
    call_wall: float | None = None
    put_wall: float | None = None
    implied_volatility: float | None = None
    selected_by_regime: bool = False

    @property
    def is_valid(self) -> bool:
        return self.rejection_reason is None and self.max_loss_rupees > 0.0


@dataclass
class WeekContext:
    week_id: str
    deployment_date: date
    deployment_weekday: str
    entry_timestamp: datetime | None
    expected_entry_time: str
    next_weekly_expiry: date | None
    observed_historical_expiry_weekday: str | None
    preferred_expiry_weekday: str
    expiry_alignment_ok: bool
    regime: str
    daily_trend_state: str
    trend_60m_state: str
    monday_gap_pct: float | None
    weekly_atr: float | None
    daily_atr: float | None
    india_vix: float | None
    atm_iv: float | None
    call_wall: float | None
    put_wall: float | None
    pcr: float | None
    oi_change_pressure: float | None
    support_level: float | None
    resistance_level: float | None
    previous_week_high: float | None
    previous_week_low: float | None
    previous_week_close: float | None
    spot: float | None
    weekly_range_position: float | None
    event_flag: bool
    event_labels: list[str]
    option_chain_snapshot: dict[str, Any] | None
    missing_data_flags: dict[str, bool]
    trade_ready: bool
    no_trade_reason: str | None
    actual_expiry: date | None = None
    preferred_expiry: date | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataset_row(self) -> dict[str, Any]:
        payload = {
            "week_id": self.week_id,
            "deployment_date": self.deployment_date.isoformat(),
            "deployment_weekday": self.deployment_weekday,
            "entry_timestamp": self.entry_timestamp.isoformat() if self.entry_timestamp else None,
            "expected_entry_time": self.expected_entry_time,
            "next_weekly_expiry": self.next_weekly_expiry.isoformat() if self.next_weekly_expiry else None,
            "observed_historical_expiry_weekday": self.observed_historical_expiry_weekday,
            "preferred_expiry_weekday": self.preferred_expiry_weekday,
            "expiry_alignment_ok": self.expiry_alignment_ok,
            "spot": self.spot,
            "monday_gap_pct": self.monday_gap_pct,
            "weekly_atr": self.weekly_atr,
            "daily_atr": self.daily_atr,
            "india_vix": self.india_vix,
            "previous_week_high": self.previous_week_high,
            "previous_week_low": self.previous_week_low,
            "previous_week_close": self.previous_week_close,
            "daily_trend_state": self.daily_trend_state,
            "trend_60m_state": self.trend_60m_state,
            "regime": self.regime,
            "atm_iv": self.atm_iv,
            "call_wall": self.call_wall,
            "put_wall": self.put_wall,
            "pcr": self.pcr,
            "oi_change_pressure": self.oi_change_pressure,
            "support_level": self.support_level,
            "resistance_level": self.resistance_level,
            "weekly_range_position": self.weekly_range_position,
            "event_flag": self.event_flag,
            "event_labels": json.dumps(self.event_labels),
            "option_chain_snapshot": json.dumps(self.option_chain_snapshot or {}),
            "missing_data_flags": json.dumps(self.missing_data_flags),
            "trade_ready": self.trade_ready,
            "no_trade_reason": self.no_trade_reason,
        }
        payload.update({k: v for k, v in self.metadata.items() if not isinstance(v, (dict, list))})
        return payload


def _parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        return []
    events: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict) or "date" not in item:
            continue
        events.append(item)
    return events


def _load_price_frames(config: WeeklyResearchConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price = pd.read_csv(config.price_5m_path, parse_dates=["timestamp"])
    price = price[price["timestamp"].dt.weekday < 5].copy()
    price = price.sort_values("timestamp")
    price["date"] = price["timestamp"].dt.date
    price["session_idx"] = price.groupby("date").cumcount()
    daily = (
        price.groupby("date", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            first_timestamp=("timestamp", "first"),
            last_timestamp=("timestamp", "last"),
        )
        .sort_values("date")
    )
    daily["close_ema20"] = _ema(daily["close"], 20)
    daily["close_ema50"] = _ema(daily["close"], 50)
    daily["daily_atr_14"] = _atr(daily, config.daily_atr_period)
    daily["slope20"] = daily["close_ema20"].diff(3)
    daily["prev_close"] = daily["close"].shift(1)
    daily["week_start"] = pd.to_datetime(daily["date"]).dt.to_period("W-MON").apply(lambda p: p.start_time.date())

    week_bars = (
        daily.groupby("week_start", as_index=False)
        .agg(
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            open=("open", "first"),
            first_date=("date", "first"),
            last_date=("date", "last"),
        )
        .sort_values("week_start")
    )
    week_bars["weekly_atr"] = _atr(week_bars.rename(columns={"week_start": "date"}), config.weekly_atr_period)

    hourly = (
        price.assign(hour_bucket=price["session_idx"] // 12)
        .groupby(["date", "hour_bucket"], as_index=False)
        .agg(
            timestamp=("timestamp", "max"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .sort_values("timestamp")
    )
    hourly["close_ema20"] = _ema(hourly["close"], 20)
    hourly["close_ema50"] = _ema(hourly["close"], 50)
    hourly["slope20"] = hourly["close_ema20"].diff(3)
    return price, daily, hourly


def _load_option_chain(config: WeeklyResearchConfig) -> pd.DataFrame:
    header = pd.read_csv(config.option_chain_path, nrows=0)
    parse_dates = [column for column in ("timestamp", "previous_timestamp", "expiry") if column in header.columns]
    option_chain = pd.read_csv(config.option_chain_path, parse_dates=parse_dates)
    if "oi_change" not in option_chain.columns:
        option_chain["oi_change"] = 0.0
    option_chain = option_chain[option_chain["timestamp"].dt.weekday < 5].copy()
    option_chain = option_chain.sort_values(["timestamp", "strike", "option_type"])
    option_chain["date"] = option_chain["timestamp"].dt.date
    option_chain["time"] = option_chain["timestamp"].dt.time
    return option_chain


def _repair_weekly_option_chain_expiry_history(
    option_chain: pd.DataFrame,
    *,
    trading_days: set[date],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if option_chain.empty:
        return option_chain, {
            "repair_applied": False,
            "rows_repaired": 0,
            "days_repaired": 0,
            "reason_counts": {},
            "sample_repairs": [],
            "weekday_counts_before": {},
            "weekday_counts_after": {},
        }

    work = option_chain.copy()
    work["date"] = work["timestamp"].dt.date
    weekday_before = Counter(
        pd.Timestamp(value).day_name()
        for value in work["expiry"].dropna()
    )
    rows_repaired = 0
    repairs: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for session_date, group in work.groupby("date", sort=True):
        expected_expiry = infer_nifty_weekly_expiry(session_date, trading_days=trading_days)
        if expected_expiry is None:
            continue
        observed = sorted({pd.Timestamp(value).date() for value in group["expiry"].dropna()})
        observed_expiry = observed[0] if observed else None
        reason: str | None = None
        if observed_expiry is None:
            reason = "MISSING_EXPIRY"
        else:
            days_to_expiry = (observed_expiry - session_date).days
            if days_to_expiry < 0 or days_to_expiry > 8:
                reason = "STALE_EXPIRY"
            elif observed_expiry != expected_expiry:
                reason = "REGIME_MISMATCH"
        if reason is None:
            continue
        mask = work["date"].eq(session_date)
        work.loc[mask, "expiry"] = pd.Timestamp(expected_expiry)
        rows_repaired += int(mask.sum())
        reason_counts[reason] += 1
        if len(repairs) < 25:
            repairs.append(
                {
                    "session_date": session_date.isoformat(),
                    "observed_expiry": observed_expiry.isoformat() if observed_expiry else None,
                    "repaired_expiry": expected_expiry.isoformat(),
                    "reason": reason,
                }
            )

    weekday_after = Counter(
        pd.Timestamp(value).day_name()
        for value in work["expiry"].dropna()
    )
    report = {
        "repair_applied": rows_repaired > 0,
        "rows_repaired": rows_repaired,
        "days_repaired": sum(reason_counts.values()),
        "reason_counts": dict(reason_counts),
        "sample_repairs": repairs,
        "weekday_counts_before": dict(weekday_before),
        "weekday_counts_after": dict(weekday_after),
    }
    return work, report


def _classify_trend(close: float | None, ema20: float | None, ema50: float | None, slope20: float | None) -> str:
    if close is None or ema20 is None or ema50 is None or slope20 is None:
        return "MIXED"
    if pd.isna(close) or pd.isna(ema20) or pd.isna(ema50) or pd.isna(slope20):
        return "MIXED"
    if close > ema20 > ema50 and slope20 > 0:
        return "BULLISH"
    if close < ema20 < ema50 and slope20 < 0:
        return "BEARISH"
    return "MIXED"


def _compute_proxy_vix_from_chain(chain: pd.DataFrame, spot: float) -> float | None:
    if chain.empty or pd.isna(spot):
        return None
    atm = chain.assign(dist=(chain["strike"] - spot).abs()).sort_values("dist").head(4)
    iv = pd.to_numeric(atm["iv"], errors="coerce").dropna()
    if iv.empty:
        return None
    return float(iv.mean())


def _first_snapshot_at_or_after(day_chain: pd.DataFrame, target: time, max_age_minutes: int) -> datetime | None:
    timestamps = sorted(day_chain["timestamp"].unique())
    for ts in timestamps:
        ts_dt = pd.Timestamp(ts).to_pydatetime()
        if ts_dt.time() >= target:
            return ts_dt
    if not timestamps:
        return None
    last_before = [pd.Timestamp(ts).to_pydatetime() for ts in timestamps if pd.Timestamp(ts).to_pydatetime().time() <= target]
    if not last_before:
        return None
    ts_dt = last_before[-1]
    target_dt = datetime.combine(ts_dt.date(), target)
    if (target_dt - ts_dt) <= timedelta(minutes=max_age_minutes):
        return ts_dt
    return None


def _next_preferred_expiry(deployment_date: date, preferred_weekday: str) -> date:
    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
    }
    target_weekday = weekday_map.get(preferred_weekday.lower(), 1)
    cursor = deployment_date + timedelta(days=1)
    for _ in range(10):
        if cursor.weekday() == target_weekday:
            return cursor
        cursor += timedelta(days=1)
    return cursor


def _select_entry_time(gap_pct: float | None, config: WeeklyResearchConfig) -> str:
    if gap_pct is None:
        return config.stable_entry_time
    if abs(gap_pct) >= config.gap_very_large_pct:
        return config.very_large_gap_entry_time
    if abs(gap_pct) >= config.gap_large_pct:
        return config.large_gap_entry_time
    return config.stable_entry_time


def _previous_week_row(week_bars: pd.DataFrame, deployment_date: date) -> pd.Series | None:
    candidates = week_bars[week_bars["last_date"] < deployment_date]
    if candidates.empty:
        return None
    return candidates.iloc[-1]


def _latest_daily_row(daily: pd.DataFrame, deployment_date: date) -> pd.Series | None:
    candidates = daily[daily["date"] < deployment_date]
    if candidates.empty:
        return None
    return candidates.iloc[-1]


def _latest_hourly_row(hourly: pd.DataFrame, entry_timestamp: datetime | None) -> pd.Series | None:
    if entry_timestamp is None:
        return None
    candidates = hourly[hourly["timestamp"] <= entry_timestamp]
    if candidates.empty:
        return None
    return candidates.iloc[-1]


def _monday_gap_pct(day_prices: pd.DataFrame, previous_close: float | None) -> float | None:
    if day_prices.empty or previous_close in (None, 0) or pd.isna(previous_close):
        return None
    open_price = float(day_prices.iloc[0]["open"])
    return (open_price - float(previous_close)) / float(previous_close)


def _weekly_range_position(spot: float | None, prev_week_low: float | None, prev_week_high: float | None) -> float | None:
    if any(v is None or pd.isna(v) for v in (spot, prev_week_low, prev_week_high)):
        return None
    width = float(prev_week_high) - float(prev_week_low)
    if width <= 0:
        return None
    return max(0.0, min(1.0, (float(spot) - float(prev_week_low)) / width))


def _option_chain_snapshot_features(entry_chain: pd.DataFrame, spot: float) -> dict[str, Any]:
    if entry_chain.empty:
        return {}
    chain = entry_chain.copy()
    chain["dist"] = (chain["strike"] - spot).abs()
    atm = chain.sort_values("dist").head(4)
    atm_iv = pd.to_numeric(atm["iv"], errors="coerce").dropna().mean() if not atm.empty else None
    calls = chain[chain["option_type"] == "CALL"]
    puts = chain[chain["option_type"] == "PUT"]
    call_wall = float(calls.loc[calls["oi"].idxmax()]["strike"]) if not calls.empty and calls["oi"].notna().any() else None
    put_wall = float(puts.loc[puts["oi"].idxmax()]["strike"]) if not puts.empty and puts["oi"].notna().any() else None
    total_call_oi = float(pd.to_numeric(calls["oi"], errors="coerce").fillna(0.0).sum())
    total_put_oi = float(pd.to_numeric(puts["oi"], errors="coerce").fillna(0.0).sum())
    total_call_oi_change = float(pd.to_numeric(calls["oi_change"], errors="coerce").fillna(0.0).sum())
    total_put_oi_change = float(pd.to_numeric(puts["oi_change"], errors="coerce").fillna(0.0).sum())
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else None
    imbalance_base = abs(total_call_oi_change) + abs(total_put_oi_change)
    oi_pressure = None
    if imbalance_base > 0:
        oi_pressure = (total_put_oi_change - total_call_oi_change) / imbalance_base
    spread_mid = (pd.to_numeric(chain["bid"], errors="coerce") + pd.to_numeric(chain["ask"], errors="coerce")) / 2.0
    spread_width = (pd.to_numeric(chain["ask"], errors="coerce") - pd.to_numeric(chain["bid"], errors="coerce")).abs()
    valid_mid = spread_mid[(spread_mid > 0) & spread_width.notna()]
    liquidity_score = None
    if not valid_mid.empty:
        spread_pct = (spread_width.loc[valid_mid.index] / valid_mid).clip(lower=0.0)
        liquidity_score = float((1.0 - (spread_pct / 0.25)).clip(lower=0.0, upper=1.0).mean())
    return {
        "timestamp": pd.Timestamp(entry_chain["timestamp"].iloc[0]).isoformat(),
        "expiry": pd.Timestamp(entry_chain["expiry"].iloc[0]).date().isoformat(),
        "row_count": int(len(entry_chain)),
        "spot": float(spot),
        "atm_iv": float(atm_iv) if atm_iv is not None and not pd.isna(atm_iv) else None,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "pcr": pcr,
        "oi_change_pressure": oi_pressure,
        "liquidity_score": liquidity_score,
        "observed_expiry_weekday": pd.Timestamp(entry_chain["expiry"].iloc[0]).day_name(),
    }


def _classify_weekly_regime(
    *,
    monday_gap_pct: float | None,
    daily_trend: str,
    trend_60m: str,
    weekly_range_position: float | None,
    atm_iv: float | None,
    call_wall: float | None,
    put_wall: float | None,
    pcr: float | None,
    oi_pressure: float | None,
    support_level: float | None,
    resistance_level: float | None,
    spot: float | None,
    event_flag: bool,
) -> str:
    if event_flag or (atm_iv is not None and atm_iv >= 18.0):
        return "HIGH_VOL_EVENT"
    if spot is None or support_level is None or resistance_level is None:
        return "LOW_EDGE_NO_TRADE"
    wall_balance = None
    if call_wall is not None and put_wall is not None and spot is not None:
        wall_balance = abs((call_wall - spot) - (spot - put_wall))
    bullish_alignment = daily_trend == "BULLISH" and trend_60m in {"BULLISH", "MIXED"}
    bearish_alignment = daily_trend == "BEARISH" and trend_60m in {"BEARISH", "MIXED"}
    if monday_gap_pct is not None and monday_gap_pct >= 0.004:
        if bearish_alignment or spot < resistance_level:
            return "GAP_UP_FAILURE"
    if monday_gap_pct is not None and monday_gap_pct <= -0.004:
        if bullish_alignment or spot > support_level:
            return "GAP_DOWN_RECOVERY"
    if bullish_alignment and (weekly_range_position is None or weekly_range_position >= 0.55):
        if call_wall is None or call_wall - spot >= 100:
            return "BULLISH_CONTINUATION"
    if bearish_alignment and (weekly_range_position is None or weekly_range_position <= 0.45):
        if call_wall is not None and call_wall >= spot:
            return "BEARISH_CONTINUATION"
    if (
        weekly_range_position is not None
        and 0.35 <= weekly_range_position <= 0.65
        and (pcr is None or 0.85 <= pcr <= 1.15)
        and (oi_pressure is None or abs(oi_pressure) <= 0.25)
        and (wall_balance is None or wall_balance <= 250)
    ):
        return "SIDEWAYS_RANGE"
    return "LOW_EDGE_NO_TRADE"


def build_weekly_dataset(config: WeeklyResearchConfig) -> tuple[list[WeekContext], dict[str, pd.DataFrame]]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    price_5m, daily, hourly = _load_price_frames(config)
    option_chain = _load_option_chain(config)
    expiry_repair_report: dict[str, Any] = {
        "repair_applied": False,
        "rows_repaired": 0,
        "days_repaired": 0,
        "reason_counts": {},
        "sample_repairs": [],
        "weekday_counts_before": {},
        "weekday_counts_after": {},
    }
    if config.repair_weekly_expiry_history:
        trading_days = {pd.Timestamp(day).date() for day in price_5m["date"].unique()}
        option_chain, expiry_repair_report = _repair_weekly_option_chain_expiry_history(
            option_chain,
            trading_days=trading_days,
        )
    events = _load_events(config.event_calendar_path)

    weekly_rows: list[WeekContext] = []
    trading_dates = sorted(d for d in price_5m["date"].unique() if pd.Timestamp(d).weekday() == 0)
    week_bars = (
        daily.groupby("week_start", as_index=False)
        .agg(
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            open=("open", "first"),
            first_date=("date", "first"),
            last_date=("date", "last"),
            weekly_atr=("daily_atr_14", "last"),
        )
        .sort_values("week_start")
    )

    for deployment_date in trading_dates:
        day_prices = price_5m[price_5m["date"] == deployment_date]
        day_chain = option_chain[option_chain["date"] == deployment_date]
        latest_daily = _latest_daily_row(daily, deployment_date)
        previous_week = _previous_week_row(week_bars, deployment_date)
        previous_close = float(latest_daily["close"]) if latest_daily is not None else None
        monday_gap_pct = _monday_gap_pct(day_prices, previous_close)
        expected_entry_time = _select_entry_time(monday_gap_pct, config)
        entry_timestamp = _first_snapshot_at_or_after(day_chain, _parse_time(expected_entry_time), config.max_snapshot_age_minutes)
        preferred_expiry = _next_preferred_expiry(deployment_date, config.preferred_expiry_weekday)

        missing_flags = {
            "entry_snapshot_missing": entry_timestamp is None,
            "india_vix_missing": True,
            "daily_history_insufficient": latest_daily is None or len(daily[daily["date"] < deployment_date]) < config.min_history_days,
            "previous_week_missing": previous_week is None,
            "option_chain_missing": day_chain.empty,
            "expiry_path_incomplete": False,
            "historical_expiry_mismatch": False,
            "event_calendar_missing": not config.event_calendar_path.exists(),
        }

        actual_expiry: date | None = None
        entry_chain = pd.DataFrame()
        option_snapshot: dict[str, Any] | None = None
        spot = None
        if entry_timestamp is not None:
            snapshot_chain = day_chain[day_chain["timestamp"] == pd.Timestamp(entry_timestamp)]
            future_expiries = sorted(d for d in snapshot_chain["expiry"].dt.date.unique() if d >= deployment_date)
            if future_expiries:
                actual_expiry = future_expiries[0]
                entry_chain = snapshot_chain[snapshot_chain["expiry"].dt.date == actual_expiry].copy()
                if not entry_chain.empty:
                    spot = float(entry_chain["spot"].iloc[0])
                    option_snapshot = _option_chain_snapshot_features(entry_chain, spot)
            else:
                missing_flags["entry_snapshot_missing"] = True

        expiry_alignment_ok = actual_expiry is not None and actual_expiry.weekday() == _next_preferred_expiry(deployment_date, config.preferred_expiry_weekday).weekday()
        missing_flags["historical_expiry_mismatch"] = not expiry_alignment_ok if actual_expiry is not None else False

        observed_expiry_weekday = actual_expiry.strftime("%A") if actual_expiry else None
        latest_hourly = _latest_hourly_row(hourly, entry_timestamp)
        daily_trend = _classify_trend(
            float(latest_daily["close"]) if latest_daily is not None else None,
            float(latest_daily["close_ema20"]) if latest_daily is not None else None,
            float(latest_daily["close_ema50"]) if latest_daily is not None else None,
            float(latest_daily["slope20"]) if latest_daily is not None else None,
        )
        trend_60m = _classify_trend(
            float(latest_hourly["close"]) if latest_hourly is not None else None,
            float(latest_hourly["close_ema20"]) if latest_hourly is not None else None,
            float(latest_hourly["close_ema50"]) if latest_hourly is not None else None,
            float(latest_hourly["slope20"]) if latest_hourly is not None else None,
        )

        previous_week_high = float(previous_week["high"]) if previous_week is not None else None
        previous_week_low = float(previous_week["low"]) if previous_week is not None else None
        previous_week_close = float(previous_week["close"]) if previous_week is not None else None
        weekly_range_position = _weekly_range_position(spot, previous_week_low, previous_week_high)
        atm_iv = option_snapshot.get("atm_iv") if option_snapshot else None
        call_wall = option_snapshot.get("call_wall") if option_snapshot else None
        put_wall = option_snapshot.get("put_wall") if option_snapshot else None
        pcr = option_snapshot.get("pcr") if option_snapshot else None
        oi_pressure = option_snapshot.get("oi_change_pressure") if option_snapshot else None
        india_vix = None
        if entry_chain is not None and not entry_chain.empty and spot is not None:
            india_vix = _compute_proxy_vix_from_chain(entry_chain, spot)

        support_level = previous_week_low
        resistance_level = previous_week_high
        event_labels = []
        event_flag = False
        if actual_expiry is not None:
            event_window = {deployment_date + timedelta(days=i) for i in range((actual_expiry - deployment_date).days + 1)}
            for item in events:
                event_date = datetime.strptime(str(item["date"]), "%Y-%m-%d").date()
                if event_date in event_window:
                    event_labels.append(f"{item.get('severity', 'unknown')}:{item.get('label', 'event')}")
                    if str(item.get("severity", "")).lower() == "high":
                        event_flag = True
        regime = _classify_weekly_regime(
            monday_gap_pct=monday_gap_pct,
            daily_trend=daily_trend,
            trend_60m=trend_60m,
            weekly_range_position=weekly_range_position,
            atm_iv=india_vix or atm_iv,
            call_wall=call_wall,
            put_wall=put_wall,
            pcr=pcr,
            oi_pressure=oi_pressure,
            support_level=support_level,
            resistance_level=resistance_level,
            spot=spot,
            event_flag=event_flag,
        )

        trade_ready = True
        no_trade_reason = None
        if entry_timestamp is None:
            trade_ready = False
            no_trade_reason = "ENTRY_SNAPSHOT_MISSING"
        elif actual_expiry is None:
            trade_ready = False
            no_trade_reason = "EXPIRY_UNAVAILABLE"
        elif latest_daily is None or previous_week is None:
            trade_ready = False
            no_trade_reason = "HISTORY_INSUFFICIENT"
        elif regime == "HIGH_VOL_EVENT":
            trade_ready = False
            no_trade_reason = "EVENT_RISK"
        elif regime == "LOW_EDGE_NO_TRADE":
            trade_ready = False
            no_trade_reason = "LOW_EDGE_NO_TRADE"

        if actual_expiry is not None:
            cursor = deployment_date
            price_dates = set(price_5m["date"].unique())
            while cursor <= actual_expiry:
                if cursor.weekday() < 5 and cursor not in price_dates:
                    missing_flags["expiry_path_incomplete"] = True
                    trade_ready = False
                    no_trade_reason = "INCOMPLETE_EXPIRY_PATH"
                    break
                cursor += timedelta(days=1)

        context = WeekContext(
            week_id=f"{deployment_date.isoformat()}_{(actual_expiry or preferred_expiry).isoformat()}",
            deployment_date=deployment_date,
            deployment_weekday=deployment_date.strftime("%A"),
            entry_timestamp=entry_timestamp,
            expected_entry_time=expected_entry_time,
            next_weekly_expiry=actual_expiry or preferred_expiry,
            observed_historical_expiry_weekday=observed_expiry_weekday,
            preferred_expiry_weekday=config.preferred_expiry_weekday,
            expiry_alignment_ok=expiry_alignment_ok,
            regime=regime,
            daily_trend_state=daily_trend,
            trend_60m_state=trend_60m,
            monday_gap_pct=monday_gap_pct,
            weekly_atr=float(previous_week["weekly_atr"]) if previous_week is not None and not pd.isna(previous_week["weekly_atr"]) else None,
            daily_atr=float(latest_daily["daily_atr_14"]) if latest_daily is not None and not pd.isna(latest_daily["daily_atr_14"]) else None,
            india_vix=india_vix,
            atm_iv=atm_iv,
            call_wall=call_wall,
            put_wall=put_wall,
            pcr=pcr,
            oi_change_pressure=oi_pressure,
            support_level=support_level,
            resistance_level=resistance_level,
            previous_week_high=previous_week_high,
            previous_week_low=previous_week_low,
            previous_week_close=previous_week_close,
            spot=spot,
            weekly_range_position=weekly_range_position,
            event_flag=event_flag,
            event_labels=event_labels,
            option_chain_snapshot=option_snapshot,
            missing_data_flags=missing_flags,
            trade_ready=trade_ready,
            no_trade_reason=no_trade_reason,
            actual_expiry=actual_expiry,
            preferred_expiry=preferred_expiry,
            metadata={
                "historical_expiry_used": actual_expiry.isoformat() if actual_expiry else None,
                "historical_expiry_weekday": observed_expiry_weekday,
            },
        )
        weekly_rows.append(context)

    dataset_frame = pd.DataFrame([row.to_dataset_row() for row in weekly_rows])
    return weekly_rows, {
        "price_5m": price_5m,
        "daily": daily,
        "hourly": hourly,
        "option_chain": option_chain,
        "dataset_frame": dataset_frame,
        "expiry_repair_report": expiry_repair_report,
    }


def _option_rows_for_expiry(option_chain: pd.DataFrame, expiry: date, start_ts: datetime, end_ts: datetime | None = None) -> pd.DataFrame:
    mask = option_chain["expiry"].dt.date.eq(expiry) & option_chain["timestamp"].ge(pd.Timestamp(start_ts))
    if end_ts is not None:
        mask &= option_chain["timestamp"].le(pd.Timestamp(end_ts))
    return option_chain.loc[mask].copy()


def _candidate_expected_move_points(spot: float, atm_iv: float | None, deployment_date: date, expiry: date) -> float:
    days_to_expiry = max((expiry - deployment_date).days, 1)
    vol = atm_iv if atm_iv is not None and not pd.isna(atm_iv) else 14.0
    return float(spot * (vol / 100.0) * math.sqrt(days_to_expiry / 365.0))


def _normalize_delta(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return abs(float(value))


def _quote_sell_price(row: pd.Series) -> float:
    bid = pd.to_numeric(row.get("bid"), errors="coerce")
    ltp = pd.to_numeric(row.get("ltp"), errors="coerce")
    if pd.notna(bid) and bid > 0:
        return float(bid)
    return float(ltp) if pd.notna(ltp) else 0.0


def _quote_buy_price(row: pd.Series) -> float:
    ask = pd.to_numeric(row.get("ask"), errors="coerce")
    ltp = pd.to_numeric(row.get("ltp"), errors="coerce")
    if pd.notna(ask) and ask > 0:
        return float(ask)
    return float(ltp) if pd.notna(ltp) else 0.0


def _quote_close_short_price(row: pd.Series) -> float:
    return _quote_buy_price(row)


def _quote_close_long_price(row: pd.Series) -> float:
    return _quote_sell_price(row)


def _liquidity_score_for_rows(rows: Iterable[pd.Series]) -> float:
    scores: list[float] = []
    for row in rows:
        bid = pd.to_numeric(row.get("bid"), errors="coerce")
        ask = pd.to_numeric(row.get("ask"), errors="coerce")
        if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0 or ask < bid:
            scores.append(0.0)
            continue
        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / max(mid, 1e-6)
        scores.append(float(max(0.0, min(1.0, 1.0 - spread_pct / 0.25))))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _find_row_by_strike(chain: pd.DataFrame, strike: float, option_type: str) -> pd.Series | None:
    subset = chain[(chain["option_type"] == option_type) & (chain["strike"] == strike)]
    if subset.empty:
        return None
    return subset.iloc[0]


def _pick_nearest_otm_strike(chain: pd.DataFrame, *, target_strike: float, option_type: str, direction: str) -> float | None:
    subset = chain[chain["option_type"] == option_type]
    if subset.empty:
        return None
    strikes = sorted(float(v) for v in subset["strike"].unique())
    if direction == "LOWER":
        candidates = [strike for strike in strikes if strike <= target_strike]
        return candidates[-1] if candidates else None
    candidates = [strike for strike in strikes if strike >= target_strike]
    return candidates[0] if candidates else None


def _terminal_pnl_points(legs: list[dict[str, Any]], terminal_spot: float, entry_cashflow_points: float) -> float:
    payoff = entry_cashflow_points
    for leg in legs:
        strike = float(leg["strike"])
        qty = int(leg["qty"])
        if leg["option_type"] == "CALL":
            intrinsic = max(terminal_spot - strike, 0.0)
        else:
            intrinsic = max(strike - terminal_spot, 0.0)
        payoff += qty * intrinsic
    return payoff


def _max_profit_and_loss_rupees(legs: list[dict[str, Any]], entry_cashflow_points: float, lot_size: int) -> tuple[float, float]:
    strikes = sorted({float(leg["strike"]) for leg in legs})
    if not strikes:
        return 0.0, 0.0
    span = max(strikes) - min(strikes)
    buffer_points = max(500.0, span * 2.0)
    checkpoints = [max(0.0, min(strikes) - buffer_points)] + strikes + [max(strikes) + buffer_points]
    pnl_points = [_terminal_pnl_points(legs, spot, entry_cashflow_points) for spot in checkpoints]
    max_profit = max(pnl_points) * lot_size
    max_loss = abs(min(pnl_points) * lot_size)
    return float(max_profit), float(max_loss)


def _strategy_preference_for_regime(regime: str) -> tuple[str, ...]:
    mapping = {
        "BULLISH_CONTINUATION": ("BULL_PUT_CREDIT_SPREAD", "BROKEN_WING_PUT_FLY"),
        "GAP_DOWN_RECOVERY": ("BULL_PUT_CREDIT_SPREAD", "BROKEN_WING_PUT_FLY"),
        "BEARISH_CONTINUATION": ("BEAR_CALL_CREDIT_SPREAD", "BROKEN_WING_CALL_FLY"),
        "GAP_UP_FAILURE": ("BEAR_CALL_CREDIT_SPREAD", "BROKEN_WING_CALL_FLY"),
        "SIDEWAYS_RANGE": ("IRON_CONDOR",),
    }
    return mapping.get(regime, tuple())


def _credit_spread_candidate(
    *,
    context: WeekContext,
    entry_chain: pd.DataFrame,
    option_type: str,
    width_request: int,
    config: WeeklyResearchConfig,
) -> StrategyCandidate | None:
    direction = "BULLISH" if option_type == "PUT" else "BEARISH"
    strategy = "BULL_PUT_CREDIT_SPREAD" if option_type == "PUT" else "BEAR_CALL_CREDIT_SPREAD"
    delta_min, delta_max = config.short_delta_range
    expected_move = _candidate_expected_move_points(context.spot or 0.0, context.atm_iv or context.india_vix, context.deployment_date, context.actual_expiry or context.next_weekly_expiry or context.deployment_date)

    subset = entry_chain[entry_chain["option_type"] == ("PUT" if option_type == "PUT" else "CALL")].copy()
    if subset.empty or context.spot is None:
        return None
    subset["abs_delta"] = subset["delta"].map(_normalize_delta)
    subset = subset[(subset["abs_delta"] >= delta_min) & (subset["abs_delta"] <= delta_max)]
    if direction == "BULLISH":
        subset = subset[subset["strike"] < context.spot]
    else:
        subset = subset[subset["strike"] > context.spot]
    subset = subset.sort_values(["abs_delta", "strike"], ascending=[True, True])
    if subset.empty:
        return None

    best_candidate: StrategyCandidate | None = None
    best_key: tuple[int, float, float] | None = None
    for _, short_row in subset.iterrows():
        target_long = float(short_row["strike"]) - width_request if direction == "BULLISH" else float(short_row["strike"]) + width_request
        actual_long_strike = _pick_nearest_otm_strike(
            entry_chain,
            target_strike=target_long,
            option_type=str(short_row["option_type"]),
            direction="LOWER" if direction == "BULLISH" else "HIGHER",
        )
        if actual_long_strike is None or actual_long_strike == float(short_row["strike"]):
            continue
        long_row = _find_row_by_strike(entry_chain, actual_long_strike, str(short_row["option_type"]))
        if long_row is None:
            continue
        actual_width = abs(float(short_row["strike"]) - float(actual_long_strike))
        entry_credit = _quote_sell_price(short_row) - _quote_buy_price(long_row)
        if entry_credit <= 0:
            rejection = "NEGATIVE_CREDIT"
        else:
            rejection = None
        credit_width_ratio = entry_credit / actual_width if actual_width > 0 else 0.0
        support_buffer = None
        resistance_buffer = None
        if direction == "BULLISH" and context.support_level is not None:
            support_buffer = float(short_row["strike"]) - context.support_level
            if support_buffer < max(actual_width / 2.0, expected_move * 0.25):
                rejection = rejection or "SHORT_STRIKE_TOO_CLOSE_TO_SUPPORT"
        if direction == "BEARISH" and context.resistance_level is not None:
            resistance_buffer = float(short_row["strike"]) - context.resistance_level
            if resistance_buffer < max(actual_width / 2.0, expected_move * 0.25):
                rejection = rejection or "SHORT_STRIKE_TOO_CLOSE_TO_RESISTANCE"
        distance_from_spot = abs(float(short_row["strike"]) - float(context.spot))
        if distance_from_spot < expected_move * config.min_expected_move_buffer:
            rejection = rejection or "SHORT_STRIKE_TOO_CLOSE_TO_EXPECTED_MOVE"
        if credit_width_ratio < config.min_credit_width_ratio:
            rejection = rejection or "CREDIT_WIDTH_TOO_LOW"
        rows_for_liquidity = [short_row, long_row]
        liquidity_score = _liquidity_score_for_rows(rows_for_liquidity)
        if liquidity_score < config.min_liquidity_score:
            rejection = rejection or "LIQUIDITY_BAD"
        legs = [
            {"option_type": str(short_row["option_type"]), "strike": float(short_row["strike"]), "qty": -1, "entry_price": _quote_sell_price(short_row), "entry_delta": float(short_row.get("delta") or 0.0)},
            {"option_type": str(long_row["option_type"]), "strike": float(long_row["strike"]), "qty": 1, "entry_price": _quote_buy_price(long_row), "entry_delta": float(long_row.get("delta") or 0.0)},
        ]
        max_profit, max_loss = _max_profit_and_loss_rupees(legs, entry_credit, config.lot_size)
        reward_risk = max_profit / max(max_loss, 1e-6)
        score = (
            (2.4 * credit_width_ratio)
            + (1.3 * liquidity_score)
            + (0.8 * min(distance_from_spot / max(expected_move, 1.0), 2.0))
            + (0.7 * min((support_buffer or resistance_buffer or distance_from_spot) / max(actual_width, 1.0), 2.0))
            + (0.5 * (delta_max - float(short_row["abs_delta"])))
        )
        candidate = StrategyCandidate(
            strategy=strategy,
            direction=direction,
            regime=context.regime,
            requested_width=width_request,
            actual_width=int(actual_width),
            expiry=context.actual_expiry or context.next_weekly_expiry or context.deployment_date,
            entry_timestamp=context.entry_timestamp or datetime.combine(context.deployment_date, _parse_time(config.stable_entry_time)),
            entry_spot=float(context.spot),
            expected_move_points=expected_move,
            entry_cashflow_points=float(entry_credit),
            max_profit_rupees=max_profit,
            max_loss_rupees=max_loss,
            reward_risk_ratio=float(reward_risk),
            credit_width_ratio=float(credit_width_ratio),
            liquidity_score=float(liquidity_score),
            short_leg_delta=_normalize_delta(short_row.get("delta")),
            long_leg_delta=_normalize_delta(long_row.get("delta")),
            score=float(score),
            rejection_reason=rejection,
            legs=legs,
            strategy_family="CREDIT_SPREAD",
            support_level=context.support_level,
            resistance_level=context.resistance_level,
            call_wall=context.call_wall,
            put_wall=context.put_wall,
            implied_volatility=context.atm_iv or context.india_vix,
        )
        candidate_key = (1 if candidate.is_valid else 0, candidate.score, candidate.credit_width_ratio)
        if best_candidate is None or candidate_key > (best_key or (0, float("-inf"), float("-inf"))):
            best_candidate = candidate
            best_key = candidate_key
    return best_candidate


def _iron_condor_candidates(context: WeekContext, entry_chain: pd.DataFrame, config: WeeklyResearchConfig) -> list[StrategyCandidate]:
    if context.spot is None:
        return []
    delta_min, delta_max = config.condor_delta_range
    puts = entry_chain[(entry_chain["option_type"] == "PUT")].copy()
    calls = entry_chain[(entry_chain["option_type"] == "CALL")].copy()
    puts["abs_delta"] = puts["delta"].map(_normalize_delta)
    calls["abs_delta"] = calls["delta"].map(_normalize_delta)
    puts = puts[(puts["abs_delta"] >= delta_min) & (puts["abs_delta"] <= delta_max) & (puts["strike"] < context.spot)]
    calls = calls[(calls["abs_delta"] >= delta_min) & (calls["abs_delta"] <= delta_max) & (calls["strike"] > context.spot)]
    puts = puts.sort_values(["abs_delta", "strike"], ascending=[True, False])
    calls = calls.sort_values(["abs_delta", "strike"], ascending=[True, True])
    if puts.empty or calls.empty:
        return []
    expected_move = _candidate_expected_move_points(context.spot, context.atm_iv or context.india_vix, context.deployment_date, context.actual_expiry or context.next_weekly_expiry or context.deployment_date)
    candidates: list[StrategyCandidate] = []
    stable_walls = (
        context.call_wall is not None
        and context.put_wall is not None
        and abs((context.call_wall - context.spot) - (context.spot - context.put_wall)) <= config.max_condor_wall_distance_points
    )
    for width_request in config.widths:
        for _, short_put in puts.head(3).iterrows():
            for _, short_call in calls.head(3).iterrows():
                target_long_put = float(short_put["strike"]) - width_request
                target_long_call = float(short_call["strike"]) + width_request
                actual_long_put = _pick_nearest_otm_strike(entry_chain, target_strike=target_long_put, option_type="PUT", direction="LOWER")
                actual_long_call = _pick_nearest_otm_strike(entry_chain, target_strike=target_long_call, option_type="CALL", direction="HIGHER")
                if actual_long_put is None or actual_long_call is None:
                    continue
                long_put = _find_row_by_strike(entry_chain, actual_long_put, "PUT")
                long_call = _find_row_by_strike(entry_chain, actual_long_call, "CALL")
                if long_put is None or long_call is None:
                    continue
                actual_width = int(max(abs(float(short_put["strike"]) - actual_long_put), abs(actual_long_call - float(short_call["strike"]))))
                entry_credit = (
                    _quote_sell_price(short_put)
                    + _quote_sell_price(short_call)
                    - _quote_buy_price(long_put)
                    - _quote_buy_price(long_call)
                )
                credit_width_ratio = entry_credit / max(actual_width, 1)
                liquidity = _liquidity_score_for_rows([short_put, short_call, long_put, long_call])
                rejection = None
                if entry_credit <= 0:
                    rejection = "NEGATIVE_CREDIT"
                elif credit_width_ratio < config.min_condor_credit_width_ratio:
                    rejection = "CREDIT_WIDTH_TOO_LOW"
                elif liquidity < config.min_liquidity_score:
                    rejection = "LIQUIDITY_BAD"
                elif not stable_walls:
                    rejection = "UNSTABLE_WALLS"
                elif (float(short_call["strike"]) - float(context.spot)) < expected_move * 0.55:
                    rejection = "SHORT_CALL_TOO_CLOSE"
                elif (float(context.spot) - float(short_put["strike"])) < expected_move * 0.55:
                    rejection = "SHORT_PUT_TOO_CLOSE"
                legs = [
                    {"option_type": "PUT", "strike": float(short_put["strike"]), "qty": -1, "entry_price": _quote_sell_price(short_put), "entry_delta": float(short_put.get("delta") or 0.0)},
                    {"option_type": "PUT", "strike": float(long_put["strike"]), "qty": 1, "entry_price": _quote_buy_price(long_put), "entry_delta": float(long_put.get("delta") or 0.0)},
                    {"option_type": "CALL", "strike": float(short_call["strike"]), "qty": -1, "entry_price": _quote_sell_price(short_call), "entry_delta": float(short_call.get("delta") or 0.0)},
                    {"option_type": "CALL", "strike": float(long_call["strike"]), "qty": 1, "entry_price": _quote_buy_price(long_call), "entry_delta": float(long_call.get("delta") or 0.0)},
                ]
                max_profit, max_loss = _max_profit_and_loss_rupees(legs, entry_credit, config.lot_size)
                reward_risk = max_profit / max(max_loss, 1e-6)
                score = (2.0 * credit_width_ratio) + (1.2 * liquidity) + (0.8 * min(expected_move / max(actual_width, 1), 2.0))
                candidates.append(
                    StrategyCandidate(
                        strategy="IRON_CONDOR",
                        direction="NEUTRAL",
                        regime=context.regime,
                        requested_width=width_request,
                        actual_width=actual_width,
                        expiry=context.actual_expiry or context.next_weekly_expiry or context.deployment_date,
                        entry_timestamp=context.entry_timestamp or datetime.combine(context.deployment_date, _parse_time(config.stable_entry_time)),
                        entry_spot=float(context.spot),
                        expected_move_points=expected_move,
                        entry_cashflow_points=float(entry_credit),
                        max_profit_rupees=max_profit,
                        max_loss_rupees=max_loss,
                        reward_risk_ratio=float(reward_risk),
                        credit_width_ratio=float(credit_width_ratio),
                        liquidity_score=float(liquidity),
                        short_leg_delta=max(_normalize_delta(short_put.get("delta")) or 0.0, _normalize_delta(short_call.get("delta")) or 0.0),
                        long_leg_delta=None,
                        score=float(score),
                        rejection_reason=rejection,
                        strategy_family="CONDOR",
                        legs=legs,
                        support_level=context.support_level,
                        resistance_level=context.resistance_level,
                        call_wall=context.call_wall,
                        put_wall=context.put_wall,
                        implied_volatility=context.atm_iv or context.india_vix,
                    )
                )
    return candidates


def _broken_wing_fly_candidates(context: WeekContext, entry_chain: pd.DataFrame, config: WeeklyResearchConfig, side: str) -> list[StrategyCandidate]:
    if context.spot is None:
        return []
    strategy = "BROKEN_WING_CALL_FLY" if side == "CALL" else "BROKEN_WING_PUT_FLY"
    direction = "BEARISH" if side == "CALL" else "BULLISH"
    delta_min, delta_max = config.short_delta_range
    subset = entry_chain[entry_chain["option_type"] == side].copy()
    subset["abs_delta"] = subset["delta"].map(_normalize_delta)
    subset = subset[(subset["abs_delta"] >= delta_min) & (subset["abs_delta"] <= delta_max)]
    if side == "CALL":
        subset = subset[subset["strike"] > context.spot].sort_values(["abs_delta", "strike"], ascending=[True, True])
    else:
        subset = subset[subset["strike"] < context.spot].sort_values(["abs_delta", "strike"], ascending=[True, False])
    if subset.empty:
        return []
    expected_move = _candidate_expected_move_points(context.spot, context.atm_iv or context.india_vix, context.deployment_date, context.actual_expiry or context.next_weekly_expiry or context.deployment_date)
    results: list[StrategyCandidate] = []
    for width_request in config.widths:
        for _, short_row in subset.head(4).iterrows():
            inner_target = float(short_row["strike"]) - width_request if side == "CALL" else float(short_row["strike"]) + width_request
            outer_target = float(short_row["strike"]) + width_request + 100 if side == "CALL" else float(short_row["strike"]) - width_request - 100
            inner_strike = _pick_nearest_otm_strike(entry_chain, target_strike=inner_target, option_type=side, direction="LOWER" if side == "CALL" else "HIGHER")
            outer_strike = _pick_nearest_otm_strike(entry_chain, target_strike=outer_target, option_type=side, direction="HIGHER" if side == "CALL" else "LOWER")
            if inner_strike is None or outer_strike is None:
                continue
            if side == "CALL" and not (inner_strike < float(short_row["strike"]) < outer_strike):
                continue
            if side == "PUT" and not (outer_strike < float(short_row["strike"]) < inner_strike):
                continue
            inner_row = _find_row_by_strike(entry_chain, inner_strike, side)
            outer_row = _find_row_by_strike(entry_chain, outer_strike, side)
            if inner_row is None or outer_row is None:
                continue
            entry_cashflow = (2 * _quote_sell_price(short_row)) - _quote_buy_price(inner_row) - _quote_buy_price(outer_row)
            legs = [
                {"option_type": side, "strike": float(inner_row["strike"]), "qty": 1, "entry_price": _quote_buy_price(inner_row), "entry_delta": float(inner_row.get("delta") or 0.0)},
                {"option_type": side, "strike": float(short_row["strike"]), "qty": -2, "entry_price": _quote_sell_price(short_row), "entry_delta": float(short_row.get("delta") or 0.0)},
                {"option_type": side, "strike": float(outer_row["strike"]), "qty": 1, "entry_price": _quote_buy_price(outer_row), "entry_delta": float(outer_row.get("delta") or 0.0)},
            ]
            max_profit, max_loss = _max_profit_and_loss_rupees(legs, entry_cashflow, config.lot_size)
            reward_risk = max_profit / max(max_loss, 1e-6)
            liquidity = _liquidity_score_for_rows([inner_row, short_row, outer_row])
            actual_width = int(max(abs(float(short_row["strike"]) - float(inner_row["strike"])), abs(float(outer_row["strike"]) - float(short_row["strike"]))))
            rejection = None
            if max_loss <= 0 or max_profit <= 0:
                rejection = "NO_DEFINED_EDGE"
            elif reward_risk < config.min_broken_wing_reward_risk:
                rejection = "REWARD_RISK_TOO_LOW"
            elif liquidity < config.min_liquidity_score:
                rejection = "LIQUIDITY_BAD"
            elif abs(float(short_row["strike"]) - float(context.spot)) < expected_move * 0.55:
                rejection = "SHORT_STRIKE_TOO_CLOSE"
            vertical_alt = _credit_spread_candidate(context=context, entry_chain=entry_chain, option_type="CALL" if side == "CALL" else "PUT", width_request=width_request, config=config)
            if vertical_alt and max_loss > vertical_alt.max_loss_rupees * 1.20:
                rejection = rejection or "TAIL_RISK_EXCESSIVE"
            score = (1.8 * reward_risk) + (0.9 * liquidity) + (0.6 * min(abs(float(short_row["strike"]) - float(context.spot)) / max(expected_move, 1.0), 2.0))
            results.append(
                StrategyCandidate(
                    strategy=strategy,
                    direction=direction,
                    regime=context.regime,
                    requested_width=width_request,
                    actual_width=actual_width,
                    expiry=context.actual_expiry or context.next_weekly_expiry or context.deployment_date,
                    entry_timestamp=context.entry_timestamp or datetime.combine(context.deployment_date, _parse_time(config.stable_entry_time)),
                    entry_spot=float(context.spot),
                    expected_move_points=expected_move,
                    entry_cashflow_points=float(entry_cashflow),
                    max_profit_rupees=max_profit,
                    max_loss_rupees=max_loss,
                    reward_risk_ratio=float(reward_risk),
                    credit_width_ratio=float(entry_cashflow / max(actual_width, 1)),
                    liquidity_score=float(liquidity),
                    short_leg_delta=_normalize_delta(short_row.get("delta")),
                    long_leg_delta=max(_normalize_delta(inner_row.get("delta")) or 0.0, _normalize_delta(outer_row.get("delta")) or 0.0),
                    score=float(score),
                    rejection_reason=rejection,
                    strategy_family="BROKEN_WING_FLY",
                    legs=legs,
                    support_level=context.support_level,
                    resistance_level=context.resistance_level,
                    call_wall=context.call_wall,
                    put_wall=context.put_wall,
                    implied_volatility=context.atm_iv or context.india_vix,
                )
            )
    return results


def generate_weekly_candidates(context: WeekContext, frames: dict[str, pd.DataFrame], config: WeeklyResearchConfig) -> list[StrategyCandidate]:
    if not context.trade_ready or context.entry_timestamp is None or context.actual_expiry is None:
        return []
    if config.allowed_regimes and context.regime not in set(config.allowed_regimes):
        return []
    option_chain = frames["option_chain"]
    entry_chain = option_chain[
        (option_chain["timestamp"] == pd.Timestamp(context.entry_timestamp))
        & (option_chain["expiry"].dt.date == context.actual_expiry)
    ].copy()
    if entry_chain.empty:
        return []
    candidates: list[StrategyCandidate] = []
    if context.atm_iv is not None and context.atm_iv < config.low_credit_iv_floor and context.regime in {"BULLISH_CONTINUATION", "BEARISH_CONTINUATION", "SIDEWAYS_RANGE", "GAP_UP_FAILURE", "GAP_DOWN_RECOVERY"}:
        return []
    if context.regime in {"BULLISH_CONTINUATION", "GAP_DOWN_RECOVERY"}:
        for width in config.widths:
            candidate = _credit_spread_candidate(context=context, entry_chain=entry_chain, option_type="PUT", width_request=width, config=config)
            if candidate is not None:
                candidate.selected_by_regime = True
                candidates.append(candidate)
        candidates.extend(_broken_wing_fly_candidates(context, entry_chain, config, side="PUT"))
    elif context.regime in {"BEARISH_CONTINUATION", "GAP_UP_FAILURE"}:
        for width in config.widths:
            candidate = _credit_spread_candidate(context=context, entry_chain=entry_chain, option_type="CALL", width_request=width, config=config)
            if candidate is not None:
                candidate.selected_by_regime = True
                candidates.append(candidate)
        candidates.extend(_broken_wing_fly_candidates(context, entry_chain, config, side="CALL"))
    elif context.regime == "SIDEWAYS_RANGE":
        candidates.extend(_iron_condor_candidates(context, entry_chain, config))
    if config.allowed_strategies:
        allowed = set(config.allowed_strategies)
        candidates = [candidate for candidate in candidates if candidate.strategy in allowed]
    return candidates


def _choose_best_candidate(candidates: list[StrategyCandidate]) -> StrategyCandidate | None:
    valid = [candidate for candidate in candidates if candidate.is_valid]
    if not valid:
        return None
    valid.sort(key=lambda item: (item.score, item.reward_risk_ratio, item.credit_width_ratio), reverse=True)
    return valid[0]


def _time_exit_deadline(profile: ExitProfile, context: WeekContext, option_history: pd.DataFrame) -> datetime:
    if context.entry_timestamp is None:
        raise ValueError("context.entry_timestamp is required")
    entry_date = context.entry_timestamp.date()
    dates = sorted(option_history["date"].unique())
    entry_day_rows = option_history[option_history["date"] == entry_date]
    if profile.time_exit == "MONDAY_CLOSE":
        return pd.Timestamp(entry_day_rows["timestamp"].max()).to_pydatetime()
    if len(dates) > 1:
        next_day = dates[1]
    else:
        next_day = entry_date
    next_day_rows = option_history[option_history["date"] == next_day]
    if profile.time_exit == "TUESDAY_OPEN":
        subset = next_day_rows[next_day_rows["time"] >= _parse_time("09:45")]
        return pd.Timestamp((subset["timestamp"].min() if not subset.empty else next_day_rows["timestamp"].max())).to_pydatetime()
    if profile.time_exit == "TUESDAY_13:00":
        subset = next_day_rows[next_day_rows["time"] >= _parse_time("13:00")]
        return pd.Timestamp((subset["timestamp"].min() if not subset.empty else next_day_rows["timestamp"].max())).to_pydatetime()
    if profile.time_exit == "TUESDAY_15:00":
        subset = next_day_rows[next_day_rows["time"] >= _parse_time("15:00")]
        return pd.Timestamp((subset["timestamp"].min() if not subset.empty else next_day_rows["timestamp"].max())).to_pydatetime()
    expiry_rows = option_history[option_history["date"] == context.actual_expiry]
    return pd.Timestamp(expiry_rows["timestamp"].max()).to_pydatetime()


def _mark_to_market(candidate: StrategyCandidate, snapshot: pd.DataFrame) -> tuple[float, dict[str, float], float]:
    pnl_points = 0.0
    short_deltas: dict[str, float] = {}
    current_spot = float(snapshot["spot"].iloc[0]) if not snapshot.empty else candidate.entry_spot
    for leg in candidate.legs:
        quote = _find_row_by_strike(snapshot, leg["strike"], leg["option_type"])
        if quote is None:
            continue
        qty = int(leg["qty"])
        if qty < 0:
            close_price = _quote_close_short_price(quote)
            pnl_points += abs(qty) * (float(leg["entry_price"]) - close_price)
            short_deltas[f"{leg['option_type']}:{leg['strike']}"] = _normalize_delta(quote.get("delta")) or 0.0
        else:
            close_price = _quote_close_long_price(quote)
            pnl_points += qty * (close_price - float(leg["entry_price"]))
    return pnl_points, short_deltas, current_spot


def simulate_trade(
    candidate: StrategyCandidate,
    context: WeekContext,
    frames: dict[str, pd.DataFrame],
    profile: ExitProfile,
    config: WeeklyResearchConfig,
) -> dict[str, Any]:
    option_history = _option_rows_for_expiry(frames["option_chain"], candidate.expiry, candidate.entry_timestamp)
    if option_history.empty:
        return {
            "week_id": context.week_id,
            "deployment_date": context.deployment_date.isoformat(),
            "strategy": candidate.strategy,
            "regime": context.regime,
            "result": "NO_TRADE",
            "reason": "NO_OPTION_HISTORY",
            "exit_profile": profile.profile_id,
        }
    deadline = _time_exit_deadline(profile, context, option_history)
    option_history = option_history[option_history["timestamp"] <= pd.Timestamp(deadline)].copy()
    timestamps = sorted(option_history["timestamp"].unique())
    if not timestamps:
        return {
            "week_id": context.week_id,
            "deployment_date": context.deployment_date.isoformat(),
            "strategy": candidate.strategy,
            "regime": context.regime,
            "result": "NO_TRADE",
            "reason": "NO_MARKS_BEFORE_DEADLINE",
            "exit_profile": profile.profile_id,
        }
    profit_target_rupees = candidate.max_profit_rupees * profile.profit_target_pct
    loss_limit_rupees = max(0.0, candidate.entry_cashflow_points * config.lot_size * profile.credit_stop_multiple)
    debit_loss_cap_rupees = candidate.max_loss_rupees * config.max_loss_stop_fraction_for_debit
    mfe = float("-inf")
    mae = float("inf")
    final_snapshot = option_history[option_history["timestamp"] == timestamps[-1]]
    exit_reason = f"TIME_EXIT_{profile.time_exit}"
    exit_timestamp = pd.Timestamp(timestamps[-1]).to_pydatetime()
    realized_pnl = 0.0
    for ts in timestamps:
        snapshot = option_history[option_history["timestamp"] == ts]
        pnl_points, short_deltas, current_spot = _mark_to_market(candidate, snapshot)
        pnl_rupees = pnl_points * config.lot_size
        mfe = max(mfe, pnl_rupees)
        mae = min(mae, pnl_rupees)
        exit_reason_local = None
        if pnl_rupees >= profit_target_rupees:
            exit_reason_local = f"PROFIT_TARGET_{int(profile.profit_target_pct * 100)}"
        elif candidate.strategy_family in {"CREDIT_SPREAD", "CONDOR"} and pnl_rupees <= -loss_limit_rupees:
            exit_reason_local = f"STOP_{profile.credit_stop_multiple:g}X_CREDIT"
        elif candidate.strategy_family == "BROKEN_WING_FLY" and pnl_rupees <= -debit_loss_cap_rupees:
            exit_reason_local = "MAX_LOSS_FRACTION_STOP"
        else:
            short_strikes = [float(leg["strike"]) for leg in candidate.legs if int(leg["qty"]) < 0]
            if candidate.direction == "BULLISH" and short_strikes and current_spot <= max(short_strikes):
                exit_reason_local = "SHORT_STRIKE_BREACH"
            elif candidate.direction == "BEARISH" and short_strikes and current_spot >= min(short_strikes):
                exit_reason_local = "SHORT_STRIKE_BREACH"
            elif candidate.direction == "NEUTRAL" and short_strikes:
                if current_spot <= min(short_strikes) or current_spot >= max(short_strikes):
                    exit_reason_local = "SHORT_STRIKE_BREACH"
        if exit_reason_local is None and short_deltas:
            if max(short_deltas.values()) >= config.delta_breach_threshold:
                exit_reason_local = "DELTA_BREACH"
        if exit_reason_local is None and candidate.direction == "BULLISH" and context.support_level is not None and current_spot < context.support_level - max(context.daily_atr or 0.0, 50.0) * 0.25:
            exit_reason_local = "STRUCTURE_INVALIDATION"
        if exit_reason_local is None and candidate.direction == "BEARISH" and context.resistance_level is not None and current_spot > context.resistance_level + max(context.daily_atr or 0.0, 50.0) * 0.25:
            exit_reason_local = "STRUCTURE_INVALIDATION"
        if exit_reason_local is not None:
            exit_reason = exit_reason_local
            exit_timestamp = pd.Timestamp(ts).to_pydatetime()
            realized_pnl = pnl_rupees
            final_snapshot = snapshot
            break
        realized_pnl = pnl_rupees
        final_snapshot = snapshot
    holding_minutes = (exit_timestamp - candidate.entry_timestamp).total_seconds() / 60.0
    capital_at_risk = candidate.max_loss_rupees
    return {
        "week_id": context.week_id,
        "deployment_date": context.deployment_date.isoformat(),
        "entry_timestamp": candidate.entry_timestamp.isoformat(),
        "exit_timestamp": exit_timestamp.isoformat(),
        "days_held": round(holding_minutes / (60.0 * 24.0), 4),
        "holding_minutes": holding_minutes,
        "strategy": candidate.strategy,
        "strategy_family": candidate.strategy_family,
        "direction": candidate.direction,
        "regime": context.regime,
        "exit_profile": profile.profile_id,
        "exit_reason": exit_reason,
        "pnl_rupees": round(realized_pnl, 2),
        "max_profit_rupees": round(candidate.max_profit_rupees, 2),
        "max_loss_rupees": round(candidate.max_loss_rupees, 2),
        "capital_at_risk_rupees": round(capital_at_risk, 2),
        "reward_risk_ratio": round(candidate.reward_risk_ratio, 4),
        "credit_width_ratio": round(candidate.credit_width_ratio, 4),
        "liquidity_score": round(candidate.liquidity_score, 4),
        "requested_width": candidate.requested_width,
        "actual_width": candidate.actual_width,
        "short_leg_delta": round(candidate.short_leg_delta or 0.0, 4) if candidate.short_leg_delta is not None else None,
        "long_leg_delta": round(candidate.long_leg_delta or 0.0, 4) if candidate.long_leg_delta is not None else None,
        "entry_cashflow_points": round(candidate.entry_cashflow_points, 4),
        "mfe_rupees": round(mfe if mfe != float("-inf") else 0.0, 2),
        "mae_rupees": round(mae if mae != float("inf") else 0.0, 2),
        "win": realized_pnl > 0,
        "loss": realized_pnl < 0,
        "legs": candidate.legs,
        "selected_candidate_score": round(candidate.score, 4),
        "selected_candidate_notes": candidate.notes,
        "final_spot": round(float(final_snapshot["spot"].iloc[0]), 2) if not final_snapshot.empty else None,
    }


def _summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    realized = [float(trade["pnl_rupees"]) for trade in trades if trade.get("pnl_rupees") is not None]
    wins = [value for value in realized if value > 0]
    losses = [value for value in realized if value < 0]
    capital = [float(trade.get("capital_at_risk_rupees") or 0.0) for trade in trades]
    if realized:
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for value in realized:
            equity += value
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, equity - peak)
    else:
        max_drawdown = 0.0
    exit_reasons = Counter(str(trade.get("exit_reason") or "UNKNOWN") for trade in trades)
    strategy_breakdown: dict[str, dict[str, Any]] = {}
    regime_breakdown: dict[str, dict[str, Any]] = {}
    for key_name, bucket in (("strategy", strategy_breakdown), ("regime", regime_breakdown)):
        grouped: defaultdict[str, list[float]] = defaultdict(list)
        for trade in trades:
            grouped[str(trade.get(key_name) or "UNKNOWN")].append(float(trade.get("pnl_rupees") or 0.0))
        for group_key, values in grouped.items():
            bucket[group_key] = {
                "trades": len(values),
                "wins": sum(1 for value in values if value > 0),
                "losses": sum(1 for value in values if value < 0),
                "net_pnl_rupees": round(sum(values), 2),
                "avg_pnl_rupees": round(mean(values), 2) if values else 0.0,
            }
    return {
        "trades": len(realized),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round((len(wins) / len(realized)) * 100.0, 2) if realized else 0.0,
        "net_pnl_rupees": round(sum(realized), 2),
        "average_win_rupees": round(mean(wins), 2) if wins else 0.0,
        "average_loss_rupees": round(mean(losses), 2) if losses else 0.0,
        "max_drawdown_rupees": round(abs(max_drawdown), 2),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 4) if losses else round(sum(wins), 4),
        "expectancy_rupees": round(mean(realized), 2) if realized else 0.0,
        "estimated_return_on_margin_pct": round((sum(realized) / max(mean(capital), 1.0)) * 100.0, 2) if capital else 0.0,
        "best_week": max(trades, key=lambda trade: float(trade.get("pnl_rupees") or float("-inf")), default=None),
        "worst_week": min(trades, key=lambda trade: float(trade.get("pnl_rupees") or float("inf")), default=None),
        "exit_reason_distribution": dict(exit_reasons),
        "strategy_wise_performance": strategy_breakdown,
        "regime_wise_performance": regime_breakdown,
        "time_in_market_days_median": round(median(float(trade.get("days_held") or 0.0) for trade in trades), 4) if trades else 0.0,
    }


def _build_exit_profiles(config: WeeklyResearchConfig) -> list[ExitProfile]:
    profiles: list[ExitProfile] = []
    for profit_target in (0.40, 0.50, 0.60, 0.70):
        for stop_multiple in (1.5, 2.0):
            for time_exit in ("MONDAY_CLOSE", "TUESDAY_OPEN", "TUESDAY_13:00", "TUESDAY_15:00", "EXPIRY_EXIT"):
                profiles.append(ExitProfile(profit_target_pct=profit_target, credit_stop_multiple=stop_multiple, time_exit=time_exit))
    return profiles


def _evaluate_profiles(
    contexts: list[WeekContext],
    frames: dict[str, pd.DataFrame],
    config: WeeklyResearchConfig,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    exit_profiles = _build_exit_profiles(config)
    candidates_by_week: dict[str, StrategyCandidate | None] = {}
    selection_log: list[dict[str, Any]] = []
    no_trade_counter: Counter[str] = Counter()
    for context in contexts:
        candidates = generate_weekly_candidates(context, frames, config)
        best = _choose_best_candidate(candidates)
        candidates_by_week[context.week_id] = best
        if best is None:
            reason = context.no_trade_reason or "NO_VALID_STRATEGY"
            no_trade_counter[reason] += 1
        selection_log.append(
            {
                "week_id": context.week_id,
                "deployment_date": context.deployment_date.isoformat(),
                "regime": context.regime,
                "trade_ready": context.trade_ready,
                "selected_strategy": best.strategy if best else "NO_TRADE",
                "selected_score": round(best.score, 4) if best else None,
                "selected_width": best.actual_width if best else None,
                "selected_delta": round(best.short_leg_delta, 4) if best and best.short_leg_delta is not None else None,
                "candidate_count": len(candidates),
                "valid_candidate_count": sum(1 for candidate in candidates if candidate.is_valid),
                "no_trade_reason": context.no_trade_reason if best is None else None,
                "candidate_rejections": Counter(candidate.rejection_reason or "VALID" for candidate in candidates),
            }
        )
    trades_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in exit_profiles:
        for context in contexts:
            candidate = candidates_by_week.get(context.week_id)
            if candidate is None:
                continue
            trades_by_profile[profile.profile_id].append(simulate_trade(candidate, context, frames, profile, config))
    return trades_by_profile, selection_log, {"no_trade_reasons": dict(no_trade_counter)}


def _objective(summary: dict[str, Any]) -> float:
    sample_penalty = 0.0 if summary["trades"] >= 5 else (5 - summary["trades"]) * 500.0
    return float(summary["net_pnl_rupees"] - summary["max_drawdown_rupees"] * 0.6 + summary["expectancy_rupees"] * 4.0 - sample_penalty)


def _pick_best_profile(trades_by_profile: dict[str, list[dict[str, Any]]], min_sample: int) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    summaries: dict[str, dict[str, Any]] = {}
    best_profile_id = ""
    best_score = float("-inf")
    best_summary: dict[str, Any] = {}
    for profile_id, trades in trades_by_profile.items():
        summary = _summarize_trades(trades)
        summary["objective_score"] = round(_objective(summary), 2)
        summary["profile_id"] = profile_id
        summaries[profile_id] = summary
        if summary["trades"] < min_sample:
            continue
        if summary["objective_score"] > best_score:
            best_score = summary["objective_score"]
            best_profile_id = profile_id
            best_summary = summary
    if not best_profile_id and summaries:
        best_profile_id, best_summary = max(summaries.items(), key=lambda item: item[1]["objective_score"])
    return best_profile_id, best_summary, summaries


def _split_in_sample(contexts: list[WeekContext], ratio: float) -> tuple[set[str], set[str]]:
    ordered = sorted([context for context in contexts if context.trade_ready], key=lambda item: item.deployment_date)
    cut = max(1, int(len(ordered) * ratio))
    in_sample = {context.week_id for context in ordered[:cut]}
    out_of_sample = {context.week_id for context in ordered[cut:]}
    return in_sample, out_of_sample


def _summarize_subset(trades: list[dict[str, Any]], allowed_week_ids: set[str]) -> dict[str, Any]:
    return _summarize_trades([trade for trade in trades if trade.get("week_id") in allowed_week_ids])


def _walk_forward_validation(
    contexts: list[WeekContext],
    trades_by_profile: dict[str, list[dict[str, Any]]],
    config: WeeklyResearchConfig,
) -> dict[str, Any]:
    month_to_weeks: defaultdict[str, list[str]] = defaultdict(list)
    for context in sorted(contexts, key=lambda item: item.deployment_date):
        if context.trade_ready:
            month_to_weeks[context.deployment_date.strftime("%Y-%m")].append(context.week_id)
    months = sorted(month_to_weeks)
    monthly_runs: list[dict[str, Any]] = []
    aggregated_trades: list[dict[str, Any]] = []
    for idx in range(config.walk_forward_training_months, len(months)):
        train_months = months[:idx]
        test_month = months[idx]
        train_ids = {week_id for month in train_months for week_id in month_to_weeks[month]}
        best_profile_id = None
        best_score = float("-inf")
        for profile_id, trades in trades_by_profile.items():
            summary = _summarize_subset(trades, train_ids)
            score = _objective(summary)
            if summary["trades"] >= max(3, config.min_strategy_sample // 2) and score > best_score:
                best_profile_id = profile_id
                best_score = score
        if best_profile_id is None:
            continue
        test_ids = set(month_to_weeks[test_month])
        selected_trades = [trade for trade in trades_by_profile[best_profile_id] if trade.get("week_id") in test_ids]
        aggregated_trades.extend(selected_trades)
        monthly_runs.append(
            {
                "test_month": test_month,
                "selected_profile": best_profile_id,
                "train_months": train_months,
                "summary": _summarize_trades(selected_trades),
            }
        )
    return {
        "months_evaluated": len(monthly_runs),
        "monthly_runs": monthly_runs,
        "aggregate_summary": _summarize_trades(aggregated_trades),
    }


def _best_strike_selection_rules(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {}
    wins = [trade for trade in trades if float(trade.get("pnl_rupees") or 0.0) > 0]
    source = wins or trades
    width_counter = Counter(int(trade.get("actual_width") or 0) for trade in source)
    delta_values = [float(trade.get("short_leg_delta")) for trade in source if trade.get("short_leg_delta") is not None]
    strategy_counter = Counter(str(trade.get("strategy") or "UNKNOWN") for trade in source)
    return {
        "dominant_actual_width": width_counter.most_common(1)[0][0] if width_counter else None,
        "dominant_strategy": strategy_counter.most_common(1)[0][0] if strategy_counter else None,
        "median_short_delta": round(median(delta_values), 4) if delta_values else None,
        "recommended_short_delta_band": [round(min(delta_values), 4), round(max(delta_values), 4)] if delta_values else None,
    }


def _compare_with_intraday_v83(weekly_summary: dict[str, Any], config: WeeklyResearchConfig) -> dict[str, Any]:
    intraday = json.loads(config.intraday_v83_benchmark_path.read_text()) if config.intraday_v83_benchmark_path.exists() else {}
    intraday_trade_count = int(intraday.get("wins", 0) + intraday.get("losses", 0))
    intraday_avg = float(intraday.get("realized_pnl_rupees", 0.0)) / intraday_trade_count if intraday_trade_count else 0.0
    comparison = {
        "weekly": {
            "trade_count": weekly_summary.get("trades", 0),
            "average_pnl_per_trade_rupees": weekly_summary.get("expectancy_rupees", 0.0),
            "max_drawdown_rupees": weekly_summary.get("max_drawdown_rupees", 0.0),
            "estimated_return_on_margin_pct": weekly_summary.get("estimated_return_on_margin_pct", 0.0),
            "time_in_market_days_median": weekly_summary.get("time_in_market_days_median", 0.0),
            "overnight_risk": "HIGH",
            "operational_complexity": "MEDIUM",
            "consistency": weekly_summary.get("win_rate", 0.0),
        },
        "intraday_v83": {
            "trade_count": intraday_trade_count,
            "average_pnl_per_trade_rupees": round(intraday_avg, 2),
            "max_drawdown_rupees": float(intraday.get("max_drawdown_rupees", 0.0)),
            "capital_usage": float(intraday.get("max_margin_utilisation", 0.0) or 0.0),
            "time_in_market_days_median": 0.0,
            "overnight_risk": "LOW",
            "operational_complexity": "LOW",
            "consistency": round(float(intraday.get("win_rate", 0.0)) * 100.0 if float(intraday.get("win_rate", 0.0)) <= 1.0 else float(intraday.get("win_rate", 0.0)), 2),
        },
    }
    comparison["delta"] = {
        "trade_count": comparison["weekly"]["trade_count"] - comparison["intraday_v83"]["trade_count"],
        "average_pnl_per_trade_rupees": round(comparison["weekly"]["average_pnl_per_trade_rupees"] - comparison["intraday_v83"]["average_pnl_per_trade_rupees"], 2),
        "max_drawdown_rupees": round(comparison["weekly"]["max_drawdown_rupees"] - comparison["intraday_v83"]["max_drawdown_rupees"], 2),
        "time_in_market_days_median": round(comparison["weekly"]["time_in_market_days_median"] - comparison["intraday_v83"]["time_in_market_days_median"], 4),
    }
    return comparison


def run_weekly_defined_risk_research(config: WeeklyResearchConfig | None = None) -> dict[str, Any]:
    config = config or WeeklyResearchConfig()
    contexts, frames = build_weekly_dataset(config)
    dataset_frame = frames["dataset_frame"]
    expiry_repair_report = frames.get("expiry_repair_report", {})
    dataset_csv_path = config.output_root / f"weekly_nifty_defined_risk_dataset_{config.output_tag}.csv"
    dataset_json_path = config.output_root / f"weekly_nifty_defined_risk_dataset_{config.output_tag}.json"
    expiry_repair_path = config.output_root / f"weekly_nifty_expiry_repair_{config.output_tag}.json"
    dataset_frame.to_csv(dataset_csv_path, index=False)
    dataset_json_path.write_text(json.dumps([context.to_dataset_row() for context in contexts], indent=2))
    expiry_repair_path.write_text(json.dumps(expiry_repair_report, indent=2))

    trades_by_profile, selection_log, selection_meta = _evaluate_profiles(contexts, frames, config)
    best_profile_id, best_profile_summary, profile_summaries = _pick_best_profile(trades_by_profile, config.min_strategy_sample)
    best_trades = trades_by_profile.get(best_profile_id, [])
    in_sample_ids, out_of_sample_ids = _split_in_sample(contexts, config.in_sample_ratio)
    in_sample_summary = _summarize_subset(best_trades, in_sample_ids)
    out_of_sample_summary = _summarize_subset(best_trades, out_of_sample_ids)
    walk_forward = _walk_forward_validation(contexts, trades_by_profile, config)
    comparison = _compare_with_intraday_v83(best_profile_summary, config)

    no_trade_weeks = [context for context in contexts if context.week_id not in {trade.get("week_id") for trade in best_trades}]
    no_trade_reason_counter = Counter(context.no_trade_reason or "NO_VALID_STRATEGY" for context in no_trade_weeks)
    strategy_sample: defaultdict[str, list[float]] = defaultdict(list)
    for trade in best_trades:
        strategy_sample[str(trade.get("strategy") or "UNKNOWN")].append(float(trade.get("pnl_rupees") or 0.0))
    robust_strategies = {
        strategy: {
            "trades": len(values),
            "net_pnl_rupees": round(sum(values), 2),
            "avg_pnl_rupees": round(mean(values), 2) if values else 0.0,
        }
        for strategy, values in strategy_sample.items()
        if len(values) >= config.min_strategy_sample
    }
    best_weekly_strategy_candidate = None
    if robust_strategies:
        best_weekly_strategy_candidate = max(
            robust_strategies.items(),
            key=lambda item: (item[1]["net_pnl_rupees"], item[1]["avg_pnl_rupees"]),
        )[0]
    elif best_profile_summary.get("strategy_wise_performance"):
        best_weekly_strategy_candidate = max(
            best_profile_summary["strategy_wise_performance"].items(),
            key=lambda item: (item[1]["net_pnl_rupees"], item[1]["avg_pnl_rupees"]),
        )[0]

    strike_rules = _best_strike_selection_rules(best_trades)
    recommendation = "keep research only"
    if (
        best_profile_summary.get("trades", 0) >= 12
        and best_profile_summary.get("expectancy_rupees", 0.0) > 0
        and out_of_sample_summary.get("net_pnl_rupees", 0.0) > 0
        and walk_forward["aggregate_summary"].get("net_pnl_rupees", 0.0) > 0
    ):
        recommendation = "promote to shadow weekly engine"
    if (
        recommendation == "promote to shadow weekly engine"
        and best_profile_summary.get("trades", 0) >= 20
        and best_profile_summary.get("max_drawdown_rupees", 0.0) <= max(comparison["intraday_v83"].get("max_drawdown_rupees", 0.0), 1.0) * 4.0
        and out_of_sample_summary.get("win_rate", 0.0) >= 55.0
    ):
        recommendation = "paper trade weekly"
    if best_profile_summary.get("net_pnl_rupees", 0.0) <= 0 and out_of_sample_summary.get("net_pnl_rupees", 0.0) <= 0:
        recommendation = "reject weekly"

    summary_payload = {
        "generated_at": datetime.now().isoformat(),
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(config).items()},
        "dataset_artifacts": {
            "csv": str(dataset_csv_path),
            "json": str(dataset_json_path),
            "expiry_repair_report": str(expiry_repair_path),
        },
        "historical_expiry_note": {
            "preferred_expiry_weekday": config.preferred_expiry_weekday,
            "observed_weekday_counts": dict(Counter(context.observed_historical_expiry_weekday or "UNKNOWN" for context in contexts if context.observed_historical_expiry_weekday)),
            "uses_observed_historical_expiry": config.use_observed_historical_expiry,
        },
        "expiry_repair_report": expiry_repair_report,
        "candidate_weeks": len(contexts),
        "trade_ready_weeks": sum(1 for context in contexts if context.trade_ready),
        "complete_backtest_weeks": sum(1 for context in contexts if context.trade_ready and not context.missing_data_flags.get("expiry_path_incomplete")),
        "best_exit_profile": best_profile_id,
        "best_exit_rules": profile_summaries.get(best_profile_id),
        "best_profile_summary": best_profile_summary,
        "in_sample_summary": in_sample_summary,
        "out_of_sample_summary": out_of_sample_summary,
        "walk_forward_validation": walk_forward,
        "best_weekly_strategy_candidate": best_weekly_strategy_candidate,
        "best_strike_selection_rules": strike_rules,
        "strategy_sample": robust_strategies,
        "selection_log": selection_log,
        "no_trade_week_count": len(no_trade_weeks),
        "no_trade_reasons": dict(no_trade_reason_counter),
        "selection_meta": selection_meta,
        "comparison_vs_intraday_v83": comparison,
        "recommendation": recommendation,
        "allow_adjustment_research": config.allow_adjustment_research,
        "adjustment_research_result": "not_enabled" if not config.allow_adjustment_research else "not_implemented",
    }

    summary_path = config.output_root / f"weekly_nifty_defined_risk_backtest_summary_{config.output_tag}.json"
    comparison_path = config.output_root / f"weekly_nifty_vs_intraday_v83_{config.output_tag}.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    comparison_path.write_text(json.dumps(comparison, indent=2))

    per_week_path = config.output_root / f"weekly_nifty_defined_risk_per_week_{config.output_tag}.json"
    per_week_payload = []
    best_trades_by_week = {trade["week_id"]: trade for trade in best_trades}
    for context in contexts:
        per_week_payload.append(
            {
                **context.to_dataset_row(),
                "selected_trade": best_trades_by_week.get(context.week_id),
            }
        )
    per_week_path.write_text(json.dumps(per_week_payload, indent=2))

    return {
        "summary_path": str(summary_path),
        "dataset_csv_path": str(dataset_csv_path),
        "dataset_json_path": str(dataset_json_path),
        "comparison_path": str(comparison_path),
        "per_week_path": str(per_week_path),
        "summary": summary_payload,
    }
