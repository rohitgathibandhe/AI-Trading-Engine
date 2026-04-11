from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from itertools import product
import json
from pathlib import Path
import shutil

import pandas as pd

from .backtest import run_backtest
from .data_models import AdaptiveParameters, AccountRiskLimits
from .dataset import _build_5m_bars, _filter_session_spot_outliers, _prepare_spot_series, _resolve_expiry_text
from .enrichment import derive_bid_ask_from_ltp, derive_delta_from_iv, float_or_none


DEFAULT_RISK_FREE_RATE = 0.06
# NSE shifted NIFTY weekly/monthly derivatives to Tuesday expiry for contracts
# expiring on or after 2025-09-01, so weekly-expiry intraday benchmarks should
# default to the first post-change session.
POST_TUESDAY_EXPIRY_START = "2025-09-02"
DEFAULT_BACKTEST_START = POST_TUESDAY_EXPIRY_START


def build_monitor_times(*, start: str = "09:30", end: str = "15:00", step_minutes: int = 5) -> tuple[str, ...]:
    start_ts = pd.Timestamp(f"2000-01-01 {start}:00")
    end_ts = pd.Timestamp(f"2000-01-01 {end}:00")
    if end_ts < start_ts:
        raise ValueError("end must not be earlier than start")
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")
    values: list[str] = []
    current = start_ts
    while current <= end_ts:
        values.append(current.strftime("%H:%M"))
        current += pd.Timedelta(minutes=step_minutes)
    return tuple(values)


@dataclass(frozen=True)
class ResearchDatasetReport:
    source_root: str
    output_root: str
    from_date: str | None
    to_date: str | None
    trading_days: int
    ohlcv_5m_rows: int
    ohlcv_15m_rows: int
    options_chain_rows: int
    derived_delta_rows: int
    observed_delta_rows: int
    derived_bid_ask_rows: int
    observed_bid_ask_rows: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class OptimizationResult:
    best_parameters: AdaptiveParameters
    best_summary: dict[str, object]
    runs: list[dict[str, object]]
    search_space: dict[str, list[float]]
    data_path: str
    start: str | None
    end: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "best_parameters": asdict(self.best_parameters),
            "best_summary": self.best_summary,
            "runs": self.runs,
            "search_space": self.search_space,
            "data_path": self.data_path,
            "start": self.start,
            "end": self.end,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class ResearchDatasetAppendReport:
    output_root: str
    start_date: str | None
    end_date: str | None
    appended_trading_days: int
    total_trading_days: int
    appended_chain_rows: int
    total_chain_rows: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def build_research_backtest_dataset(
    structured_root: str | Path,
    output_root: str | Path,
    *,
    from_date: str | None = DEFAULT_BACKTEST_START,
    to_date: str | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> ResearchDatasetReport:
    structured_root = Path(structured_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ohlcv_5m = pd.read_csv(structured_root / "nifty_5m.csv")
    ohlcv_5m["timestamp"] = pd.to_datetime(ohlcv_5m["timestamp"], errors="coerce")
    ohlcv_5m = ohlcv_5m.dropna(subset=["timestamp"]).sort_values("timestamp")
    if from_date:
        ohlcv_5m = ohlcv_5m.loc[ohlcv_5m["timestamp"].dt.date >= date.fromisoformat(from_date)]
    if to_date:
        ohlcv_5m = ohlcv_5m.loc[ohlcv_5m["timestamp"].dt.date <= date.fromisoformat(to_date)]

    ohlcv_15m = _build_15m_bars(ohlcv_5m)
    ohlcv_5m_out = output_root / "nifty_5m.csv"
    ohlcv_15m_out = output_root / "nifty_15m.csv"
    ohlcv_5m.to_csv(ohlcv_5m_out, index=False)
    ohlcv_15m.to_csv(ohlcv_15m_out, index=False)

    chain = pd.read_csv(structured_root / "option_chain_decision_times.csv")
    chain["timestamp"] = pd.to_datetime(chain["timestamp"], errors="coerce")
    chain = chain.dropna(subset=["timestamp", "ltp", "strike", "option_type", "spot"])
    if from_date:
        chain = chain.loc[chain["timestamp"].dt.date >= date.fromisoformat(from_date)]
    if to_date:
        chain = chain.loc[chain["timestamp"].dt.date <= date.fromisoformat(to_date)]
    chain = chain.loc[chain["expiry"].notna() & (chain["expiry"].astype(str).str.len() >= 10)].copy()
    chain["expiry"] = chain["expiry"].astype(str).str[:10]
    chain["delta_source"] = "observed"
    chain["bid_ask_source"] = "observed"

    observed_delta_rows = int(chain["delta"].notna().sum())
    observed_bid_ask_rows = int((chain["bid"].notna() & chain["ask"].notna()).sum())

    missing_delta = chain["delta"].isna()
    if missing_delta.any():
        chain.loc[missing_delta, "delta"] = chain.loc[missing_delta].apply(
            lambda row: derive_delta_from_iv(
                spot=float(row["spot"]),
                strike=float(row["strike"]),
                option_type=str(row["option_type"]),
                iv=float_or_none(row.get("iv")),
                as_of=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                expiry=date.fromisoformat(str(row["expiry"])),
                risk_free_rate=risk_free_rate,
            ),
            axis=1,
        )
        chain.loc[missing_delta & chain["delta"].notna(), "delta_source"] = "derived_bs"

    missing_bid_ask = chain["bid"].isna() | chain["ask"].isna() | (chain["bid"] <= 0) | (chain["ask"] <= 0)
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

    chain["margin_estimate_per_lot"] = ""
    options_out = output_root / "options_chain.csv"
    chain[
        [
            "timestamp",
            "expiry",
            "spot",
            "strike",
            "option_type",
            "bid",
            "ask",
            "ltp",
            "delta",
            "iv",
            "oi",
            "margin_estimate_per_lot",
            "delta_source",
            "bid_ask_source",
        ]
    ].to_csv(options_out, index=False)

    notes: list[str] = []
    if observed_bid_ask_rows == 0:
        notes.append("Historical bid/ask was absent; backtest spread costs use derived research spreads, not observed quotes.")
    if observed_delta_rows == 0:
        notes.append("Historical delta was absent; delta values were derived from spot/strike/IV/expiry using Black-Scholes.")

    return ResearchDatasetReport(
        source_root=str(structured_root),
        output_root=str(output_root),
        from_date=from_date,
        to_date=to_date,
        trading_days=int(ohlcv_5m["timestamp"].dt.date.nunique()),
        ohlcv_5m_rows=int(len(ohlcv_5m)),
        ohlcv_15m_rows=int(len(ohlcv_15m)),
        options_chain_rows=int(len(chain)),
        derived_delta_rows=int((chain["delta_source"] == "derived_bs").sum()),
        observed_delta_rows=observed_delta_rows,
        derived_bid_ask_rows=int((chain["bid_ask_source"] == "derived_liquidity_model").sum()),
        observed_bid_ask_rows=observed_bid_ask_rows,
        notes=notes,
    )


def build_research_backtest_dataset_from_rolling(
    rolling_root: str | Path,
    output_root: str | Path,
    *,
    from_date: str | None = DEFAULT_BACKTEST_START,
    to_date: str | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    monitor_times: tuple[str, ...] | None = None,
) -> ResearchDatasetReport:
    rolling_root = Path(rolling_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    monitor_times = monitor_times or build_monitor_times(start="09:30", end="15:00", step_minutes=5)
    ohlcv_5m_out = output_root / "nifty_5m.csv"
    ohlcv_15m_out = output_root / "nifty_15m.csv"
    options_out = output_root / "options_chain.csv"
    for path in (ohlcv_5m_out, ohlcv_15m_out, options_out):
        path.unlink(missing_ok=True)

    session_dirs = sorted(path for path in rolling_root.iterdir() if path.is_dir())
    if from_date:
        session_dirs = [path for path in session_dirs if path.name >= from_date]
    if to_date:
        session_dirs = [path for path in session_dirs if path.name <= to_date]

    trading_days = 0
    ohlcv_5m_rows = 0
    ohlcv_15m_rows = 0
    options_chain_rows = 0
    derived_delta_rows = 0
    observed_delta_rows = 0
    derived_bid_ask_rows = 0
    observed_bid_ask_rows = 0
    wrote_5m = False
    wrote_15m = False
    wrote_chain = False
    chain_columns = [
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
    ]
    for session_dir in session_dirs:
        parquet_path = session_dir / "rolling_option.parquet"
        if not parquet_path.exists():
            continue
        session_df = pd.read_parquet(parquet_path)
        if session_df.empty:
            continue
        spot_df = _prepare_spot_series(session_df)
        bars_5m = _build_5m_bars(spot_df)
        if not bars_5m.empty:
            trading_days += 1
            ohlcv_5m_rows += len(bars_5m)
            bars_5m.to_csv(ohlcv_5m_out, mode="a", header=not wrote_5m, index=False)
            wrote_5m = True
            bars_15m = _build_15m_bars(pd.DataFrame(bars_5m))
            ohlcv_15m_rows += len(bars_15m)
            bars_15m.to_csv(ohlcv_15m_out, mode="a", header=not wrote_15m, index=False)
            wrote_15m = True
        chain_snapshot_df = _build_monitor_chain(session_df, session_date=session_dir.name, monitor_times=monitor_times)
        if not chain_snapshot_df.empty:
            chain = chain_snapshot_df.copy()
            for column in chain_columns:
                if column not in chain.columns:
                    chain[column] = None
            chain["delta_source"] = "observed"
            chain["bid_ask_source"] = "observed"
            observed_delta_rows += int(chain["delta"].notna().sum())
            observed_bid_ask_rows += int((chain["bid"].notna() & chain["ask"].notna()).sum()) if "bid" in chain and "ask" in chain else 0

            missing_delta = chain["delta"].isna()
            if missing_delta.any():
                chain.loc[missing_delta, "delta"] = chain.loc[missing_delta].apply(
                    lambda row: derive_delta_from_iv(
                        spot=float(row["spot"]),
                        strike=float(row["strike"]),
                        option_type=str(row["option_type"]),
                        iv=float_or_none(row.get("iv")),
                        as_of=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                        expiry=date.fromisoformat(str(row["expiry"])),
                        risk_free_rate=risk_free_rate,
                    ),
                    axis=1,
                )
                derived_delta_rows += int((missing_delta & chain["delta"].notna()).sum())
                chain.loc[missing_delta & chain["delta"].notna(), "delta_source"] = "derived_bs"

            missing_bid_ask = chain["bid"].isna() | chain["ask"].isna() | (chain["bid"] <= 0) | (chain["ask"] <= 0)
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
                derived_bid_ask_rows += int(missing_bid_ask.sum())
                chain.loc[missing_bid_ask, "bid_ask_source"] = "derived_liquidity_model"

            chain = chain.sort_values(["option_type", "strike", "timestamp"]).reset_index(drop=True)
            grouping = ["expiry", "option_type", "strike"]
            chain["prev_bid"] = chain.groupby(grouping)["bid"].shift(1)
            chain["prev_ask"] = chain.groupby(grouping)["ask"].shift(1)
            chain["bid_change"] = chain["bid"] - chain["prev_bid"]
            chain["ask_change"] = chain["ask"] - chain["prev_ask"]
            chain["iv_change"] = chain["iv_change"].where(chain["iv_change"].notna(), chain["iv"] - chain.groupby(grouping)["iv"].shift(1))
            chain["oi_change"] = chain["oi_change"].where(chain["oi_change"].notna(), chain["oi"] - chain.groupby(grouping)["oi"].shift(1))
            chain["ltp_change"] = chain["ltp_change"].where(chain["ltp_change"].notna(), chain["ltp"] - chain.groupby(grouping)["ltp"].shift(1))
            chain["margin_estimate_per_lot"] = ""
            chain = chain[
                [
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
                ]
            ]
            options_chain_rows += len(chain)
            chain.to_csv(options_out, mode="a", header=not wrote_chain, index=False)
            wrote_chain = True

    notes: list[str] = []
    if observed_bid_ask_rows == 0:
        notes.append("Historical bid/ask was absent; monitoring snapshots use derived research spreads.")
    if observed_delta_rows == 0:
        notes.append("Historical delta was absent; monitoring snapshots use derived Black-Scholes deltas.")
    notes.append(f"Monitoring snapshots built at {', '.join(monitor_times)} while entry eligibility can remain narrower in backtests.")
    if not wrote_5m:
        pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]).to_csv(ohlcv_5m_out, index=False)
    if not wrote_15m:
        pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]).to_csv(ohlcv_15m_out, index=False)
    if not wrote_chain:
        pd.DataFrame(columns=chain_columns + ["margin_estimate_per_lot", "delta_source", "bid_ask_source"]).to_csv(options_out, index=False)
    return ResearchDatasetReport(
        source_root=str(rolling_root),
        output_root=str(output_root),
        from_date=from_date,
        to_date=to_date,
        trading_days=trading_days,
        ohlcv_5m_rows=ohlcv_5m_rows,
        ohlcv_15m_rows=ohlcv_15m_rows,
        options_chain_rows=options_chain_rows,
        derived_delta_rows=derived_delta_rows,
        observed_delta_rows=observed_delta_rows,
        derived_bid_ask_rows=derived_bid_ask_rows,
        observed_bid_ask_rows=observed_bid_ask_rows,
        notes=notes,
    )


def append_research_backtest_dataset_from_rolling(
    rolling_root: str | Path,
    output_root: str | Path,
    *,
    from_date: str | None = DEFAULT_BACKTEST_START,
    to_date: str | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    monitor_times: tuple[str, ...] | None = None,
) -> ResearchDatasetAppendReport:
    output_root = Path(output_root)
    chain_path = output_root / "options_chain.csv"
    bars_5m_path = output_root / "nifty_5m.csv"
    bars_15m_path = output_root / "nifty_15m.csv"
    if not chain_path.exists() or not bars_5m_path.exists() or not bars_15m_path.exists():
        report = build_research_backtest_dataset_from_rolling(
            rolling_root,
            output_root,
            from_date=from_date,
            to_date=to_date,
            risk_free_rate=risk_free_rate,
            monitor_times=monitor_times,
        )
        return ResearchDatasetAppendReport(
            output_root=str(output_root),
            start_date=report.from_date,
            end_date=report.to_date,
            appended_trading_days=report.trading_days,
            total_trading_days=report.trading_days,
            appended_chain_rows=report.options_chain_rows,
            total_chain_rows=report.options_chain_rows,
            notes=["Research dataset did not exist; built from scratch."] + list(report.notes),
        )

    max_dates: list[date] = []
    for path in (bars_5m_path, bars_15m_path, chain_path):
        df = pd.read_csv(path, usecols=["timestamp"])
        ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
        if ts.empty:
            continue
        max_dates.append(ts.max().date())
    if not max_dates:
        report = build_research_backtest_dataset_from_rolling(
            rolling_root,
            output_root,
            from_date=from_date,
            to_date=to_date,
            risk_free_rate=risk_free_rate,
            monitor_times=monitor_times,
        )
        return ResearchDatasetAppendReport(
            output_root=str(output_root),
            start_date=report.from_date,
            end_date=report.to_date,
            appended_trading_days=report.trading_days,
            total_trading_days=report.trading_days,
            appended_chain_rows=report.options_chain_rows,
            total_chain_rows=report.options_chain_rows,
            notes=["Research dataset timestamps were empty; rebuilt from scratch."] + list(report.notes),
        )

    existing_max = min(max_dates)
    next_date = existing_max + pd.Timedelta(days=1)
    next_date_value = pd.Timestamp(next_date).date()
    requested_from = pd.Timestamp(from_date).date() if from_date else next_date_value
    append_from = max(next_date_value, requested_from)
    requested_to = pd.Timestamp(to_date).date() if to_date else date.today()
    if append_from > requested_to:
        total_chain_rows = int(len(pd.read_csv(chain_path, usecols=["timestamp"])))
        total_days = int(pd.to_datetime(pd.read_csv(bars_5m_path, usecols=["timestamp"])["timestamp"], errors="coerce").dt.date.nunique())
        return ResearchDatasetAppendReport(
            output_root=str(output_root),
            start_date=None,
            end_date=None,
            appended_trading_days=0,
            total_trading_days=total_days,
            appended_chain_rows=0,
            total_chain_rows=total_chain_rows,
            notes=["Research dataset already covers the requested window."],
        )

    tmp_root = output_root.parent / f"{output_root.name}__append_tmp"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    report = build_research_backtest_dataset_from_rolling(
        rolling_root,
        tmp_root,
        from_date=append_from.isoformat(),
        to_date=requested_to.isoformat(),
        risk_free_rate=risk_free_rate,
        monitor_times=monitor_times,
    )
    if report.trading_days <= 0:
        shutil.rmtree(tmp_root, ignore_errors=True)
        total_chain_rows = int(len(pd.read_csv(chain_path, usecols=["timestamp"])))
        total_days = int(pd.to_datetime(pd.read_csv(bars_5m_path, usecols=["timestamp"])["timestamp"], errors="coerce").dt.date.nunique())
        return ResearchDatasetAppendReport(
            output_root=str(output_root),
            start_date=append_from.isoformat(),
            end_date=requested_to.isoformat(),
            appended_trading_days=0,
            total_trading_days=total_days,
            appended_chain_rows=0,
            total_chain_rows=total_chain_rows,
            notes=["No new rolling-option sessions were appended."] + list(report.notes),
        )

    for name in ("nifty_5m.csv", "nifty_15m.csv", "options_chain.csv"):
        src = tmp_root / name
        dst = output_root / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_csv(dst)
        appended = pd.read_csv(src)
        merged = pd.concat([existing, appended], ignore_index=True)
        merged.to_csv(dst, index=False)

    total_chain_rows = int(len(pd.read_csv(chain_path, usecols=["timestamp"])))
    total_days = int(pd.to_datetime(pd.read_csv(bars_5m_path, usecols=["timestamp"])["timestamp"], errors="coerce").dt.date.nunique())
    shutil.rmtree(tmp_root, ignore_errors=True)
    return ResearchDatasetAppendReport(
        output_root=str(output_root),
        start_date=append_from.isoformat(),
        end_date=requested_to.isoformat(),
        appended_trading_days=report.trading_days,
        total_trading_days=total_days,
        appended_chain_rows=report.options_chain_rows,
        total_chain_rows=total_chain_rows,
        notes=list(report.notes),
    )


def optimize_backtest_parameters(
    data_path: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    risk_limits: AccountRiskLimits | None = None,
    search_space: dict[str, list[float]] | None = None,
    db_dir: str | Path = "/tmp/intraday_defined_risk_opt",
) -> OptimizationResult:
    search_space = search_space or {
        "rv_mid_cutoff": [0.25, 0.30, 0.35],
        "rv_high_cutoff": [0.50, 0.60, 0.70],
        "directional_credit_width_ratio": [0.15, 0.20, 0.25, 0.30],
        "condor_credit_width_ratio": [0.20, 0.25, 0.30, 0.35],
    }
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    best_run: dict[str, object] | None = None
    for idx, (rv_mid, rv_high, directional_ratio, condor_ratio) in enumerate(
        product(
            search_space["rv_mid_cutoff"],
            search_space["rv_high_cutoff"],
            search_space["directional_credit_width_ratio"],
            search_space["condor_credit_width_ratio"],
        )
    ):
        if rv_high < rv_mid:
            continue
        params = AdaptiveParameters(
            rv_mid_cutoff=rv_mid,
            rv_high_cutoff=rv_high,
            directional_credit_width_ratio=directional_ratio,
            condor_credit_width_ratio=condor_ratio,
        ).clamped()
        db_path = db_dir / f"run_{idx:03d}.sqlite3"
        result = run_backtest(
            data_path,
            start=start,
            end=end,
            risk_limits=risk_limits,
            learning_db_path=db_path,
            parameters=params,
            reset_learning_db=True,
        )
        run_record = {
            "parameters": asdict(params),
            "summary": result["summary"],
        }
        runs.append(run_record)
        if best_run is None or float(run_record["summary"]["objective_score"]) > float(best_run["summary"]["objective_score"]):
            best_run = run_record

    assert best_run is not None
    return OptimizationResult(
        best_parameters=AdaptiveParameters(**best_run["parameters"]),
        best_summary=best_run["summary"],
        runs=sorted(runs, key=lambda run: float(run["summary"]["objective_score"]), reverse=True),
        search_space=search_space,
        data_path=str(data_path),
        start=start,
        end=end,
    )


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


def _build_monitor_chain(df: pd.DataFrame, *, session_date: str, monitor_times: tuple[str, ...]) -> pd.DataFrame:
    work = df.copy()
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
        return pd.DataFrame()
    work["expiryDate"] = work.get("expiryDate")
    available = sorted(work["timestamp"].dropna().unique())
    rows: list[dict[str, object]] = []
    previous_front_map: dict[tuple[int, str], dict[str, object]] = {}
    previous_front_timestamp: pd.Timestamp | None = None
    for monitor_time in monitor_times:
        target = pd.Timestamp(f"{session_date} {monitor_time}:00")
        nearest = _nearest_timestamp(available, target)
        if nearest is None or abs(nearest - target) > pd.Timedelta(minutes=5):
            continue
        snap = work.loc[work["timestamp"] == nearest].copy()
        if snap.empty:
            continue
        expiry_norm = snap["expiryDate"].map(lambda value: _resolve_expiry_text(value, session_date=session_date)[0])
        valid_expiries = sorted(expiry for expiry in expiry_norm.dropna().unique() if expiry >= session_date)
        if not valid_expiries:
            inferred_expiry, inferred = _resolve_expiry_text(None, session_date=session_date)
            if inferred and inferred_expiry >= session_date:
                valid_expiries = [inferred_expiry]
        if not valid_expiries:
            continue
        front_expiry = valid_expiries[0]
        if expiry_norm.notna().any():
            front = snap.loc[expiry_norm == front_expiry].copy()
        else:
            front = snap.copy()
        if front.empty:
            continue
        for record in front.to_dict(orient="records"):
            option_type = str(record.get("optionType") or "").upper()
            if option_type in {"CE", "CALL"}:
                option_type = "CALL"
            elif option_type in {"PE", "PUT"}:
                option_type = "PUT"
            else:
                continue
            strike = int(round(float(record["strikePrice"])))
            previous = previous_front_map.get((strike, option_type))
            rows.append(
                {
                    "timestamp": pd.Timestamp(nearest).strftime("%Y-%m-%dT%H:%M:%S"),
                    "previous_timestamp": previous_front_timestamp.strftime("%Y-%m-%dT%H:%M:%S") if previous_front_timestamp is not None else None,
                    "expiry": front_expiry,
                    "spot": record.get("spot"),
                    "strike": strike,
                    "option_type": option_type,
                    "bid": None,
                    "ask": None,
                    "ltp": record.get("ltp"),
                    "prev_ltp": previous.get("ltp") if previous else None,
                    "ltp_change": (float(record.get("ltp")) - float(previous.get("ltp"))) if previous and previous.get("ltp") is not None else None,
                    "delta": record.get("delta"),
                    "iv": record.get("iv"),
                    "prev_iv": previous.get("iv") if previous else None,
                    "iv_change": (float(record.get("iv")) - float(previous.get("iv"))) if previous and previous.get("iv") is not None and record.get("iv") is not None else None,
                    "oi": record.get("oi"),
                    "prev_oi": previous.get("oi") if previous else None,
                    "oi_change": (float(record.get("oi")) - float(previous.get("oi"))) if previous and previous.get("oi") is not None and record.get("oi") is not None else None,
                }
            )
        previous_front_map = {}
        for record in front.to_dict(orient="records"):
            option_type = str(record.get("optionType") or "").upper()
            if option_type in {"CE", "CALL"}:
                option_type = "CALL"
            elif option_type in {"PE", "PUT"}:
                option_type = "PUT"
            else:
                continue
            strike = int(round(float(record["strikePrice"])))
            previous_front_map[(strike, option_type)] = record
        previous_front_timestamp = pd.Timestamp(nearest)
    return pd.DataFrame(rows)


def _nearest_timestamp(values: list[pd.Timestamp], target: pd.Timestamp) -> pd.Timestamp | None:
    if not values:
        return None
    return min(values, key=lambda value: abs(value - target))
