from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import json
from pathlib import Path

import pandas as pd

from .dataset import (
    _build_5m_bars,
    _filter_session_spot_outliers,
    _normalize_option_type,
    _prepare_spot_series,
    _resolve_expiry_text,
)
from .enrichment import derive_bid_ask_from_ltp, derive_delta_from_iv, float_or_none
from ..utils.nifty_expiry_calendar import infer_nifty_weekly_expiry


REQUIRED_DATASET_FILES = {
    "nifty_5m.csv": ["timestamp", "open", "high", "low", "close", "volume"],
    "nifty_15m.csv": ["timestamp", "open", "high", "low", "close", "volume"],
    "options_chain.csv": ["timestamp", "expiry", "spot", "strike", "option_type", "bid", "ask", "ltp", "delta", "iv", "oi"],
}
DEFAULT_MONITOR_START = "09:30"
DEFAULT_MONITOR_END = "15:00"
DEFAULT_INTERVAL_MINUTES = 5
DEFAULT_MAX_STALENESS_MINUTES = 5
DATASET_METADATA_FILENAME = "dataset_metadata.json"
DATASET_MISSING_LOG_FILENAME = "missing_timestamps.csv"


def _summary_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0}
    series = pd.Series(values, dtype=float)
    return {
        "min": int(series.min()),
        "p25": round(float(series.quantile(0.25)), 2),
        "median": round(float(series.quantile(0.50)), 2),
        "p75": round(float(series.quantile(0.75)), 2),
        "max": int(series.max()),
    }


def _time_to_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def _build_time_grid(*, start: str = DEFAULT_MONITOR_START, end: str = DEFAULT_MONITOR_END, interval_minutes: int = DEFAULT_INTERVAL_MINUTES) -> tuple[str, ...]:
    start_ts = pd.Timestamp(f"2000-01-01 {start}:00")
    end_ts = pd.Timestamp(f"2000-01-01 {end}:00")
    current = start_ts
    values: list[str] = []
    while current <= end_ts:
        values.append(current.strftime("%H:%M"))
        current += pd.Timedelta(minutes=interval_minutes)
    return tuple(values)


def _build_15m_bars(ohlcv_5m: pd.DataFrame) -> pd.DataFrame:
    if ohlcv_5m.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    work = ohlcv_5m.copy().sort_values("timestamp")
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"])
    work["session_date"] = work["timestamp"].dt.date
    frames: list[pd.DataFrame] = []
    for _, group in work.groupby("session_date", sort=True):
        group = group.reset_index(drop=True)
        group["bucket"] = group.index // 3
        frames.append(
            group.groupby("bucket", as_index=False)
            .agg(
                timestamp=("timestamp", "last"),
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .drop(columns=["bucket"], errors="ignore")
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


@dataclass(frozen=True)
class DatasetInspectionReport:
    data_path: str
    trading_days: int
    total_unique_decision_timestamps: int
    min_timestamp: str | None
    max_timestamp: str | None
    expected_timestamps_per_day: int
    total_expected_timestamps: int
    density_ratio: float
    per_day_timestamp_count_summary: dict[str, float | int]
    days_with_fewer_than_40_timestamps: list[str]
    days_ending_before_1445: list[str]
    missing_required_columns: dict[str, list[str]]
    option_chain_rows_per_timestamp_summary: dict[str, float | int]
    metadata_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class DenseDatasetBuildReport:
    rolling_root: str
    output_path: str
    start_date: str
    end_date: str
    interval: str
    trading_days: int
    expected_timestamps_per_day: int
    total_expected_timestamps: int
    populated_timestamps: int
    density_ratio: float
    per_day_timestamp_count_summary: dict[str, float | int]
    days_with_fewer_than_40_timestamps: list[str]
    days_ending_before_1445: list[str]
    missing_reason_counts: dict[str, int]
    metadata_path: str
    missing_log_path: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class DatasetDensityGuardrail:
    warning_required: bool
    reasons: list[str]
    report: DatasetInspectionReport


def inspect_backtest_dataset(
    data_path: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    monitor_start: str = DEFAULT_MONITOR_START,
    monitor_end: str = DEFAULT_MONITOR_END,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> DatasetInspectionReport:
    root = Path(data_path)
    missing_required_columns: dict[str, list[str]] = {}
    chain_path = root / "options_chain.csv"
    bars_path = root / "nifty_5m.csv"
    bar_days: set[date] = set()
    if bars_path.exists():
        bars_df = pd.read_csv(bars_path, usecols=lambda column: column == "timestamp")
        bars_ts = pd.to_datetime(bars_df.get("timestamp"), errors="coerce").dropna()
        if start:
            bars_ts = bars_ts.loc[bars_ts >= pd.Timestamp(start)]
        if end:
            bars_ts = bars_ts.loc[bars_ts <= pd.Timestamp(end)]
        bar_days = set(bars_ts.dt.date.unique().tolist())
    chain_cols: list[str] = []
    if chain_path.exists():
        chain_head = pd.read_csv(chain_path, nrows=0)
        chain_cols = list(chain_head.columns)
    for filename, required in REQUIRED_DATASET_FILES.items():
        path = root / filename
        if not path.exists():
            missing_required_columns[filename] = required
            continue
        header = pd.read_csv(path, nrows=0)
        missing = [column for column in required if column not in header.columns]
        if missing:
            missing_required_columns[filename] = missing

    if not chain_path.exists():
        report = DatasetInspectionReport(
            data_path=str(root),
            trading_days=len(bar_days),
            total_unique_decision_timestamps=0,
            min_timestamp=None,
            max_timestamp=None,
            expected_timestamps_per_day=len(_build_time_grid(start=monitor_start, end=monitor_end, interval_minutes=interval_minutes)),
            total_expected_timestamps=0,
            density_ratio=0.0,
            per_day_timestamp_count_summary=_summary_stats([]),
            days_with_fewer_than_40_timestamps=[],
            days_ending_before_1445=[],
            missing_required_columns=missing_required_columns,
            option_chain_rows_per_timestamp_summary=_summary_stats([]),
            metadata_path=str(root / DATASET_METADATA_FILENAME) if (root / DATASET_METADATA_FILENAME).exists() else None,
            warnings=["Missing options_chain.csv; dataset cannot be used for backtest decisions."],
        )
        return report

    chain = pd.read_csv(chain_path)
    if "timestamp" not in chain.columns:
        missing_required_columns.setdefault("options_chain.csv", []).append("timestamp")
        chain = pd.DataFrame(columns=["timestamp"])
    chain["timestamp"] = pd.to_datetime(chain.get("timestamp"), errors="coerce")
    chain = chain.dropna(subset=["timestamp"]).copy()
    if start:
        chain = chain.loc[chain["timestamp"] >= pd.Timestamp(start)]
    if end:
        chain = chain.loc[chain["timestamp"] <= pd.Timestamp(end)]
    chain["session_date"] = chain["timestamp"].dt.strftime("%Y-%m-%d")
    row_counts = chain.groupby("timestamp").size().sort_index() if not chain.empty else pd.Series(dtype=int)
    unique_ts = row_counts.index.to_list()
    counts_per_day = row_counts.groupby(pd.Index(ts.date().isoformat() for ts in row_counts.index)).size()
    last_ts_by_day = row_counts.groupby(pd.Index(ts.date().isoformat() for ts in row_counts.index)).apply(lambda series: max(series.index))

    trading_days = max(len(bar_days), int(counts_per_day.size))
    expected_timestamps_per_day = len(_build_time_grid(start=monitor_start, end=monitor_end, interval_minutes=interval_minutes))
    total_expected = trading_days * expected_timestamps_per_day
    density_ratio = round((len(unique_ts) / total_expected), 4) if total_expected > 0 else 0.0
    days_with_fewer_than_40 = sorted(day for day, count in counts_per_day.items() if int(count) < 40)
    cutoff_minutes = _time_to_minutes("14:45")
    days_ending_before_1445 = sorted(
        day
        for day, ts in last_ts_by_day.items()
        if ((ts.hour * 60) + ts.minute) < cutoff_minutes
    )

    warnings: list[str] = []
    if density_ratio < 0.70:
        warnings.append("Total unique decision timestamps are below 70% of the expected dense grid.")
    if counts_per_day.size and float(counts_per_day.median()) < 40:
        warnings.append("Median timestamps per day are below 40.")
    if counts_per_day.size and (len(days_ending_before_1445) / len(counts_per_day)) > 0.25:
        warnings.append("More than 25% of days end before 14:45.")

    report = DatasetInspectionReport(
        data_path=str(root),
        trading_days=trading_days,
        total_unique_decision_timestamps=int(len(unique_ts)),
        min_timestamp=min(unique_ts).isoformat() if unique_ts else None,
        max_timestamp=max(unique_ts).isoformat() if unique_ts else None,
        expected_timestamps_per_day=expected_timestamps_per_day,
        total_expected_timestamps=total_expected,
        density_ratio=density_ratio,
        per_day_timestamp_count_summary=_summary_stats(counts_per_day.astype(int).tolist()),
        days_with_fewer_than_40_timestamps=days_with_fewer_than_40,
        days_ending_before_1445=days_ending_before_1445,
        missing_required_columns=missing_required_columns,
        option_chain_rows_per_timestamp_summary=_summary_stats(row_counts.astype(int).tolist()),
        metadata_path=str(root / DATASET_METADATA_FILENAME) if (root / DATASET_METADATA_FILENAME).exists() else None,
        warnings=warnings,
    )
    return report


def assess_dataset_density(
    data_path: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    monitor_start: str = DEFAULT_MONITOR_START,
    monitor_end: str = DEFAULT_MONITOR_END,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> DatasetDensityGuardrail:
    report = inspect_backtest_dataset(
        data_path,
        start=start,
        end=end,
        monitor_start=monitor_start,
        monitor_end=monitor_end,
        interval_minutes=interval_minutes,
    )
    reasons: list[str] = []
    if report.total_expected_timestamps > 0 and report.total_unique_decision_timestamps < (0.70 * report.total_expected_timestamps):
        reasons.append("total_unique_decision_timestamps_below_70pct_expected")
    if float(report.per_day_timestamp_count_summary.get("median", 0.0)) < 40:
        reasons.append("median_timestamps_per_day_below_40")
    denominator = report.trading_days if report.trading_days > 0 else 0
    if denominator and (len(report.days_ending_before_1445) / denominator) > 0.25:
        reasons.append("more_than_25pct_days_end_before_1445")
    return DatasetDensityGuardrail(
        warning_required=bool(reasons),
        reasons=reasons,
        report=report,
    )


def _build_chain_rows_for_snapshot(
    front: pd.DataFrame,
    *,
    decision_timestamp: pd.Timestamp,
    previous_decision_timestamp: pd.Timestamp | None,
    risk_free_rate: float,
) -> pd.DataFrame:
    chain = pd.DataFrame(
        {
            "timestamp": decision_timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
            "previous_timestamp": previous_decision_timestamp.strftime("%Y-%m-%dT%H:%M:%S") if previous_decision_timestamp is not None else None,
            "expiry": front["resolved_expiry"].astype(str).str[:10],
            "spot": pd.to_numeric(front["spot"], errors="coerce"),
            "strike": pd.to_numeric(front["strikePrice"], errors="coerce").round().astype("Int64"),
            "option_type": front["normalized_option_type"],
            "bid": None,
            "ask": None,
            "prev_bid": None,
            "prev_ask": None,
            "bid_change": None,
            "ask_change": None,
            "ltp": pd.to_numeric(front["ltp"], errors="coerce"),
            "prev_ltp": None,
            "ltp_change": None,
            "delta": pd.to_numeric(front.get("delta"), errors="coerce"),
            "iv": pd.to_numeric(front.get("iv"), errors="coerce"),
            "prev_iv": None,
            "iv_change": None,
            "oi": pd.to_numeric(front.get("oi"), errors="coerce"),
            "prev_oi": None,
            "oi_change": None,
            "margin_estimate_per_lot": "",
            "delta_source": "observed",
            "bid_ask_source": "observed",
        }
    )
    missing_delta = chain["delta"].isna()
    if missing_delta.any():
        derived_delta = chain.loc[missing_delta].apply(
            lambda row: derive_delta_from_iv(
                spot=float(row["spot"]),
                strike=float(row["strike"]),
                option_type=str(row["option_type"]),
                iv=float_or_none(row.get("iv")),
                as_of=decision_timestamp.to_pydatetime(),
                expiry=date.fromisoformat(str(row["expiry"])),
                risk_free_rate=risk_free_rate,
            ),
            axis=1,
        )
        chain.loc[missing_delta, "delta"] = pd.to_numeric(derived_delta, errors="coerce")
        chain.loc[missing_delta & chain["delta"].notna(), "delta_source"] = "derived_bs"
    missing_bid_ask = chain["bid"].isna() | chain["ask"].isna() | (pd.to_numeric(chain["bid"], errors="coerce") <= 0) | (pd.to_numeric(chain["ask"], errors="coerce") <= 0)
    if missing_bid_ask.any():
        synthetic = chain.loc[missing_bid_ask].apply(
            lambda row: derive_bid_ask_from_ltp(
                ltp=float(row["ltp"]),
                oi=float_or_none(row.get("oi")),
                iv=float_or_none(row.get("iv")),
            ),
            axis=1,
            result_type="expand",
        )
        synthetic.columns = ["bid", "ask"]
        chain.loc[missing_bid_ask, ["bid", "ask"]] = synthetic.values
        chain.loc[missing_bid_ask, "bid_ask_source"] = "derived_liquidity_model"
    return chain


def build_dense_backtest_dataset(
    rolling_root: str | Path,
    output_path: str | Path,
    *,
    start_date: str,
    end_date: str,
    interval: str = "5m",
    monitor_start: str = DEFAULT_MONITOR_START,
    monitor_end: str = DEFAULT_MONITOR_END,
    risk_free_rate: float = 0.06,
    max_staleness_minutes: int = DEFAULT_MAX_STALENESS_MINUTES,
) -> DenseDatasetBuildReport:
    if interval != "5m":
        raise ValueError("Only 5m interval is currently supported.")
    interval_minutes = 5
    monitor_grid = _build_time_grid(start=monitor_start, end=monitor_end, interval_minutes=interval_minutes)
    rolling_root = Path(rolling_root)
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    session_dirs = sorted(path for path in rolling_root.iterdir() if path.is_dir() and start_date <= path.name <= end_date)
    trading_days = {
        datetime.strptime(path.name, "%Y-%m-%d").date()
        for path in session_dirs
        if len(path.name) == 10
    }

    bars_5m_frames: list[pd.DataFrame] = []
    chain_frames: list[pd.DataFrame] = []
    missing_rows: list[dict[str, object]] = []
    for session_dir in session_dirs:
        parquet_path = session_dir / "rolling_option.parquet"
        session_date = session_dir.name
        previous_decision_timestamp: pd.Timestamp | None = None
        if not parquet_path.exists():
            for monitor_time in monitor_grid:
                missing_rows.append(
                    {
                        "session_date": session_date,
                        "decision_timestamp": f"{session_date}T{monitor_time}:00",
                        "reason": "ROLLING_PARQUET_MISSING",
                        "raw_snapshot_timestamp": None,
                        "snapshot_age_seconds": None,
                    }
                )
            continue
        session_df = pd.read_parquet(parquet_path)
        if session_df.empty:
            for monitor_time in monitor_grid:
                missing_rows.append(
                    {
                        "session_date": session_date,
                        "decision_timestamp": f"{session_date}T{monitor_time}:00",
                        "reason": "ROLLING_SESSION_EMPTY",
                        "raw_snapshot_timestamp": None,
                        "snapshot_age_seconds": None,
                    }
                )
            continue
        spot_df = _prepare_spot_series(session_df)
        bars_5m = _build_5m_bars(spot_df)
        if not bars_5m.empty:
            bars_5m_frames.append(bars_5m)
        work = session_df.copy()
        work["timestamp"] = pd.to_datetime(work["tradeDate"].astype(str) + " " + work["tradeTime"].astype(str), errors="coerce")
        work["strikePrice"] = pd.to_numeric(work.get("strikePrice"), errors="coerce")
        work["ltp"] = pd.to_numeric(work.get("ltp"), errors="coerce")
        work["delta"] = pd.to_numeric(work.get("delta"), errors="coerce")
        work["iv"] = pd.to_numeric(work.get("iv"), errors="coerce")
        work["oi"] = pd.to_numeric(work.get("oi"), errors="coerce")
        work["spot"] = pd.to_numeric(work.get("spot"), errors="coerce")
        work = work.dropna(subset=["timestamp", "strikePrice", "spot", "ltp"])
        work = _filter_session_spot_outliers(work)
        if work.empty:
            for monitor_time in monitor_grid:
                missing_rows.append(
                    {
                        "session_date": session_date,
                        "decision_timestamp": f"{session_date}T{monitor_time}:00",
                        "reason": "NO_VALID_CHAIN_ROWS",
                        "raw_snapshot_timestamp": None,
                        "snapshot_age_seconds": None,
                    }
                )
            continue
        available = sorted(pd.Timestamp(value) for value in work["timestamp"].dropna().unique())
        raw_idx = -1
        for monitor_time in monitor_grid:
            decision_ts = pd.Timestamp(f"{session_date} {monitor_time}:00")
            while raw_idx + 1 < len(available) and available[raw_idx + 1] <= decision_ts:
                raw_idx += 1
            if raw_idx < 0:
                missing_rows.append(
                    {
                        "session_date": session_date,
                        "decision_timestamp": decision_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "reason": "NO_SNAPSHOT_AT_OR_BEFORE_TIMESTAMP",
                        "raw_snapshot_timestamp": None,
                        "snapshot_age_seconds": None,
                    }
                )
                continue
            raw_ts = available[raw_idx]
            age_seconds = float((decision_ts - raw_ts).total_seconds())
            if age_seconds > max_staleness_minutes * 60:
                missing_rows.append(
                    {
                        "session_date": session_date,
                        "decision_timestamp": decision_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "reason": "SNAPSHOT_STALE",
                        "raw_snapshot_timestamp": raw_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "snapshot_age_seconds": age_seconds,
                    }
                )
                continue
            snap = work.loc[work["timestamp"] == raw_ts].copy()
            if snap.empty:
                missing_rows.append(
                    {
                        "session_date": session_date,
                        "decision_timestamp": decision_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "reason": "RAW_SNAPSHOT_EMPTY",
                        "raw_snapshot_timestamp": raw_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "snapshot_age_seconds": age_seconds,
                    }
                )
                continue
            snap["resolved_expiry"] = snap["expiryDate"].map(
                lambda value: _resolve_expiry_text(value, session_date=session_date)[0]
            )
            valid_expiries = sorted(expiry for expiry in snap["resolved_expiry"].dropna().unique() if str(expiry) >= session_date)
            if not valid_expiries:
                inferred_expiry = infer_nifty_weekly_expiry(session_date, trading_days=trading_days)
                if inferred_expiry is None:
                    missing_rows.append(
                        {
                            "session_date": session_date,
                            "decision_timestamp": decision_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                            "reason": "EXPIRY_UNRESOLVED",
                            "raw_snapshot_timestamp": raw_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                            "snapshot_age_seconds": age_seconds,
                        }
                    )
                    continue
                front_expiry = inferred_expiry.isoformat()
                snap["resolved_expiry"] = front_expiry
            else:
                front_expiry = valid_expiries[0]
            front = snap.loc[snap["resolved_expiry"] == front_expiry].copy()
            if front.empty:
                missing_rows.append(
                    {
                        "session_date": session_date,
                        "decision_timestamp": decision_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "reason": "FRONT_EXPIRY_EMPTY",
                        "raw_snapshot_timestamp": raw_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "snapshot_age_seconds": age_seconds,
                    }
                )
                continue
            front["normalized_option_type"] = front["optionType"].map(_normalize_option_type)
            front = front.loc[front["normalized_option_type"] != ""].copy()
            if front.empty:
                missing_rows.append(
                    {
                        "session_date": session_date,
                        "decision_timestamp": decision_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "reason": "OPTION_TYPE_UNRESOLVED",
                        "raw_snapshot_timestamp": raw_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "snapshot_age_seconds": age_seconds,
                    }
                )
                continue
            chain_frames.append(
                _build_chain_rows_for_snapshot(
                    front,
                    decision_timestamp=decision_ts,
                    previous_decision_timestamp=previous_decision_timestamp,
                    risk_free_rate=risk_free_rate,
                )
            )
            previous_decision_timestamp = decision_ts

    bars_5m_df = pd.concat(bars_5m_frames, ignore_index=True) if bars_5m_frames else pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    bars_15m_df = _build_15m_bars(bars_5m_df)
    chain_df = pd.concat(chain_frames, ignore_index=True) if chain_frames else pd.DataFrame(columns=[
        "timestamp",
        "previous_timestamp",
        "expiry",
        "spot",
        "strike",
        "option_type",
        "bid",
        "ask",
        "prev_bid",
        "prev_ask",
        "bid_change",
        "ask_change",
        "ltp",
        "prev_ltp",
        "ltp_change",
        "delta",
        "iv",
        "prev_iv",
        "iv_change",
        "oi",
        "prev_oi",
        "oi_change",
        "margin_estimate_per_lot",
        "delta_source",
        "bid_ask_source",
    ])
    if not chain_df.empty:
        chain_df = chain_df.sort_values(["expiry", "option_type", "strike", "timestamp"]).reset_index(drop=True)
        grouping = ["expiry", "option_type", "strike"]
        chain_df["prev_bid"] = chain_df.groupby(grouping)["bid"].shift(1)
        chain_df["prev_ask"] = chain_df.groupby(grouping)["ask"].shift(1)
        chain_df["bid_change"] = pd.to_numeric(chain_df["bid"], errors="coerce") - pd.to_numeric(chain_df["prev_bid"], errors="coerce")
        chain_df["ask_change"] = pd.to_numeric(chain_df["ask"], errors="coerce") - pd.to_numeric(chain_df["prev_ask"], errors="coerce")
        chain_df["prev_ltp"] = chain_df.groupby(grouping)["ltp"].shift(1)
        chain_df["ltp_change"] = pd.to_numeric(chain_df["ltp"], errors="coerce") - pd.to_numeric(chain_df["prev_ltp"], errors="coerce")
        chain_df["prev_iv"] = chain_df.groupby(grouping)["iv"].shift(1)
        chain_df["iv_change"] = pd.to_numeric(chain_df["iv"], errors="coerce") - pd.to_numeric(chain_df["prev_iv"], errors="coerce")
        chain_df["prev_oi"] = chain_df.groupby(grouping)["oi"].shift(1)
        chain_df["oi_change"] = pd.to_numeric(chain_df["oi"], errors="coerce") - pd.to_numeric(chain_df["prev_oi"], errors="coerce")
        chain_df = chain_df.sort_values(["timestamp", "option_type", "strike"]).reset_index(drop=True)

    bars_5m_df.to_csv(output_root / "nifty_5m.csv", index=False)
    bars_15m_df.to_csv(output_root / "nifty_15m.csv", index=False)
    chain_df.to_csv(output_root / "options_chain.csv", index=False)
    missing_df = pd.DataFrame(missing_rows)
    missing_log_path = output_root / DATASET_MISSING_LOG_FILENAME
    missing_df.to_csv(missing_log_path, index=False)

    inspection = inspect_backtest_dataset(
        output_root,
        start=f"{start_date}T{monitor_start}:00",
        end=f"{end_date}T{monitor_end}:00",
        monitor_start=monitor_start,
        monitor_end=monitor_end,
        interval_minutes=interval_minutes,
    )
    missing_reason_counts = (
        missing_df["reason"].value_counts().sort_index().to_dict()
        if not missing_df.empty and "reason" in missing_df.columns
        else {}
    )
    metadata = {
        "rolling_root": str(rolling_root),
        "output_path": str(output_root),
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "monitor_start": monitor_start,
        "monitor_end": monitor_end,
        "max_staleness_minutes": max_staleness_minutes,
        "inspection": inspection.to_dict(),
        "missing_reason_counts": missing_reason_counts,
        "files": {
            "nifty_5m": str(output_root / "nifty_5m.csv"),
            "nifty_15m": str(output_root / "nifty_15m.csv"),
            "options_chain": str(output_root / "options_chain.csv"),
            "missing_log": str(missing_log_path),
        },
    }
    metadata_path = output_root / DATASET_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    notes: list[str] = []
    if inspection.warnings:
        notes.extend(inspection.warnings)
    return DenseDatasetBuildReport(
        rolling_root=str(rolling_root),
        output_path=str(output_root),
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        trading_days=inspection.trading_days,
        expected_timestamps_per_day=inspection.expected_timestamps_per_day,
        total_expected_timestamps=inspection.total_expected_timestamps,
        populated_timestamps=inspection.total_unique_decision_timestamps,
        density_ratio=inspection.density_ratio,
        per_day_timestamp_count_summary=inspection.per_day_timestamp_count_summary,
        days_with_fewer_than_40_timestamps=inspection.days_with_fewer_than_40_timestamps,
        days_ending_before_1445=inspection.days_ending_before_1445,
        missing_reason_counts={str(key): int(value) for key, value in missing_reason_counts.items()},
        metadata_path=str(metadata_path),
        missing_log_path=str(missing_log_path),
        notes=notes,
    )


def compare_benchmark_results(old_json: str | Path, new_json: str | Path) -> dict[str, object]:
    old_payload = json.loads(Path(old_json).read_text())
    new_payload = json.loads(Path(new_json).read_text())

    def _avg_pnl(payload: dict[str, object]) -> float:
        trades = float(payload.get("trades") or 0.0)
        pnl = float(payload.get("realized_pnl_rupees") or 0.0)
        return round(pnl / trades, 2) if trades else 0.0

    def _top_rejections(payload: dict[str, object]) -> dict[str, int]:
        funnel = payload.get("trade_funnel") or {}
        reasons = funnel.get("top_rejection_reasons") or []
        return {str(item[0]): int(item[1]) for item in reasons if isinstance(item, (list, tuple)) and len(item) == 2}

    report = {
        "old_json": str(old_json),
        "new_json": str(new_json),
        "signals": {
            "old": int(old_payload.get("signals") or 0),
            "new": int(new_payload.get("signals") or 0),
            "delta": int(new_payload.get("signals") or 0) - int(old_payload.get("signals") or 0),
        },
        "trades": {
            "old": int(old_payload.get("trades") or 0),
            "new": int(new_payload.get("trades") or 0),
            "delta": int(new_payload.get("trades") or 0) - int(old_payload.get("trades") or 0),
        },
        "wins_losses": {
            "old": {"wins": int(old_payload.get("wins") or 0), "losses": int(old_payload.get("losses") or 0)},
            "new": {"wins": int(new_payload.get("wins") or 0), "losses": int(new_payload.get("losses") or 0)},
        },
        "realized_pnl_rupees": {
            "old": round(float(old_payload.get("realized_pnl_rupees") or 0.0), 2),
            "new": round(float(new_payload.get("realized_pnl_rupees") or 0.0), 2),
            "delta": round(float(new_payload.get("realized_pnl_rupees") or 0.0) - float(old_payload.get("realized_pnl_rupees") or 0.0), 2),
        },
        "average_pnl_per_trade": {
            "old": _avg_pnl(old_payload),
            "new": _avg_pnl(new_payload),
            "delta": round(_avg_pnl(new_payload) - _avg_pnl(old_payload), 2),
        },
        "max_drawdown_rupees": {
            "old": round(float(old_payload.get("max_drawdown_rupees") or 0.0), 2),
            "new": round(float(new_payload.get("max_drawdown_rupees") or 0.0), 2),
            "delta": round(float(new_payload.get("max_drawdown_rupees") or 0.0) - float(old_payload.get("max_drawdown_rupees") or 0.0), 2),
        },
        "rejection_reasons": {
            "old": _top_rejections(old_payload),
            "new": _top_rejections(new_payload),
        },
        "dataset_metadata": {
            "old": old_payload.get("dataset_density"),
            "new": new_payload.get("dataset_density"),
        },
    }
    return report
