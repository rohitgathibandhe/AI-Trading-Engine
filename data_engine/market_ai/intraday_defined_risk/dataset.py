from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
import csv
import json
import sqlite3

import pandas as pd

from .enrichment import derive_delta_from_iv, float_or_none
from ..utils.nifty_expiry_calendar import infer_nifty_weekly_expiry


REQUIRED_DECISION_TIMES = ("09:30", "10:00", "13:00")
MIN_TRAINING_DAYS = 60
TARGET_TRAINING_DAYS = 120
ALLOWED_SESSION_LABELS = {"trend_down", "trend_up", "range", "event_day"}
MAX_WEEKLY_EXPIRY_DAYS = 8
SESSION_SPOT_OUTLIER_PCT = 0.10


@dataclass(frozen=True)
class DatasetCoverageReport:
    status: str
    summary: str
    available_trading_days: int
    has_5m_ohlcv: bool
    has_chain_snapshots: bool
    has_execution_fills: bool
    has_session_labels: bool
    decision_time_coverage_pct: float
    delta_coverage_pct: float
    iv_coverage_pct: float
    bid_ask_coverage_pct: float
    label_coverage_pct: float
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class StructuredDatasetPaths:
    root: Path
    ohlcv_5m: Path
    option_chain_decision_times: Path
    execution_fills: Path
    session_labels: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "StructuredDatasetPaths":
        root_path = Path(root)
        return cls(
            root=root_path,
            ohlcv_5m=root_path / "nifty_5m.csv",
            option_chain_decision_times=root_path / "option_chain_decision_times.csv",
            execution_fills=root_path / "execution_fills.csv",
            session_labels=root_path / "session_labels.csv",
        )


@dataclass(frozen=True)
class BuildStructuredDatasetReport:
    data_root: str
    rolling_root: str
    processed_sessions: int
    ohlcv_rows: int
    chain_rows: int
    fill_rows: int
    label_rows: int
    used_option_chain_db: bool
    from_date: str | None = None
    to_date: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class StructuredTrainingRefreshReport:
    data_root: str
    rolling_root: str
    trading_days: int
    ohlcv_rows: int
    fill_rows: int
    label_rows: int
    preserved_chain_rows: int
    from_date: str | None = None
    to_date: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class DeltaEnrichmentReport:
    data_root: str
    total_rows: int
    rows_missing_delta: int
    rows_enriched: int
    delta_coverage_pct: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def assess_structured_dataset(root: str | Path) -> DatasetCoverageReport:
    paths = StructuredDatasetPaths.from_root(root)
    issues: list[str] = []
    recommendations: list[str] = []

    has_5m = paths.ohlcv_5m.exists()
    has_chain = paths.option_chain_decision_times.exists()
    has_fills = paths.execution_fills.exists()
    has_labels = paths.session_labels.exists()

    trading_days = _count_trading_days(paths.ohlcv_5m) if has_5m else 0
    if not has_5m:
        issues.append("Missing 5-minute OHLCV dataset: nifty_5m.csv")
    elif trading_days < MIN_TRAINING_DAYS:
        issues.append(f"Only {trading_days} trading days of 5-minute OHLCV are available; at least {MIN_TRAINING_DAYS} are required.")
    elif trading_days < TARGET_TRAINING_DAYS:
        recommendations.append(f"Expand 5-minute OHLCV coverage from {trading_days} to at least {TARGET_TRAINING_DAYS} trading days.")

    chain_stats = _assess_chain_snapshots(paths.option_chain_decision_times) if has_chain else None
    if not has_chain:
        issues.append("Missing option_chain_decision_times.csv with 09:30/10:00/13:00 decision-time chain snapshots.")
        decision_cov = delta_cov = iv_cov = bid_ask_cov = 0.0
    else:
        decision_cov = chain_stats["decision_time_coverage_pct"]
        delta_cov = chain_stats["delta_coverage_pct"]
        iv_cov = chain_stats["iv_coverage_pct"]
        bid_ask_cov = chain_stats["bid_ask_coverage_pct"]
        if decision_cov < 0.90:
            issues.append(f"Decision-time chain coverage is only {decision_cov:.1%}; need reliable 09:30/10:00/13:00 snapshots.")
        if bid_ask_cov < 0.95:
            issues.append(f"Bid/ask coverage is only {bid_ask_cov:.1%}; spread quality cannot be evaluated reliably.")
        if delta_cov < 0.75:
            recommendations.append(f"Delta coverage is only {delta_cov:.1%}; add greeks to improve strike selection.")
        if iv_cov < 0.75:
            recommendations.append(f"IV coverage is only {iv_cov:.1%}; dynamic IV is important for spread-credit evaluation.")

    fills_stats = _assess_execution_fills(paths.execution_fills) if has_fills else None
    if not has_fills:
        issues.append("Missing execution_fills.csv with realised fills/slippage history.")
    elif fills_stats["slippage_coverage_pct"] < 0.80:
        recommendations.append("Execution fills exist but slippage coverage is weak; log quoted-mid vs fill price on every order.")

    label_cov = _assess_session_labels(paths.session_labels, trading_days) if has_labels else 0.0
    if not has_labels:
        issues.append("Missing session_labels.csv with one label per trading day: trend_down/trend_up/range/event_day.")
    elif label_cov < 0.80:
        issues.append(f"Only {label_cov:.1%} of session days have labels; the regime learner needs denser supervision.")

    if trading_days >= TARGET_TRAINING_DAYS and has_chain and has_fills and has_labels and decision_cov >= 0.90 and bid_ask_cov >= 0.95 and label_cov >= 0.80:
        status = "READY"
        summary = "Dataset coverage is strong enough for guarded calibration and shadow deployment."
    elif trading_days >= MIN_TRAINING_DAYS and has_chain:
        status = "PARTIAL"
        summary = "Dataset is usable for research and paper calibration, but learning inputs are still incomplete."
    else:
        status = "INSUFFICIENT"
        summary = "Dataset coverage is not yet sufficient for a production adaptive learner."

    recommendations.extend([
        "Maintain 5-minute OHLCV for at least 60-120 sessions or more.",
        "Capture option-chain snapshots exactly at 09:30, 10:00, and 13:00 with bid/ask and preferably delta/IV.",
        "Store realised fills with quoted mid, fill price, order timestamp, and slippage.",
        "Label each session as trend_down, trend_up, range, or event_day.",
    ])

    return DatasetCoverageReport(
        status=status,
        summary=summary,
        available_trading_days=trading_days,
        has_5m_ohlcv=has_5m,
        has_chain_snapshots=has_chain,
        has_execution_fills=has_fills,
        has_session_labels=has_labels,
        decision_time_coverage_pct=decision_cov,
        delta_coverage_pct=delta_cov,
        iv_coverage_pct=iv_cov,
        bid_ask_coverage_pct=bid_ask_cov,
        label_coverage_pct=label_cov,
        issues=issues,
        recommendations=_dedupe(recommendations),
    )


def scaffold_dataset_layout(root: str | Path) -> StructuredDatasetPaths:
    paths = StructuredDatasetPaths.from_root(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    _write_if_missing(paths.ohlcv_5m, "timestamp,open,high,low,close,volume\n")
    _write_if_missing(
        paths.option_chain_decision_times,
        "session_date,timestamp,decision_time,expiry,spot,strike,option_type,bid,ask,ltp,delta,delta_raw,delta_source,iv,oi\n",
    )
    _write_if_missing(
        paths.execution_fills,
        "session_date,timestamp,strategy,side,strike,option_type,quoted_mid,fill_price,quantity,slippage_points,pnl_rupees\n",
    )
    _write_if_missing(paths.session_labels, "session_date,label\n")
    return paths


def build_structured_dataset_from_rolling(
    rolling_root: str | Path,
    data_root: str | Path,
    *,
    trade_blotter_path: str | Path | None = None,
    scrip_master_path: str | Path | None = None,
    option_chain_db_path: str | Path | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> BuildStructuredDatasetReport:
    paths = scaffold_dataset_layout(data_root)
    rolling_path = Path(rolling_root)
    session_dirs = sorted(p for p in rolling_path.iterdir() if p.is_dir())
    if from_date:
        session_dirs = [p for p in session_dirs if p.name >= from_date]
    if to_date:
        session_dirs = [p for p in session_dirs if p.name <= to_date]
    trading_days = {
        datetime.strptime(p.name, "%Y-%m-%d").date()
        for p in session_dirs
        if len(p.name) == 10
    }

    chain_lookup = _HistoricalChainLookup(option_chain_db_path)

    ohlcv_frames: list[pd.DataFrame] = []
    chain_frames: list[pd.DataFrame] = []
    labels: list[dict[str, str]] = []
    notes: list[str] = []
    processed_sessions = 0

    for session_dir in session_dirs:
        parquet_path = session_dir / "rolling_option.parquet"
        if not parquet_path.exists():
            continue
        session_df = pd.read_parquet(parquet_path)
        if session_df.empty:
            continue
        processed_sessions += 1

        spot_df = _prepare_spot_series(session_df)
        bars_5m = _build_5m_bars(spot_df)
        if not bars_5m.empty:
            ohlcv_frames.append(bars_5m)
            labels.append(
                {
                    "session_date": session_dir.name,
                    "label": _classify_session_label(bars_5m),
                }
            )

        chain_snapshot_df = _build_decision_time_chain(
            session_df,
            session_date=session_dir.name,
            chain_lookup=chain_lookup,
            trading_days=trading_days,
        )
        if not chain_snapshot_df.empty:
            chain_frames.append(chain_snapshot_df)

    if not processed_sessions:
        notes.append("No rolling-option parquet sessions matched the requested window.")

    ohlcv_df = _concat_frames(ohlcv_frames)
    if not ohlcv_df.empty:
        ohlcv_df.to_csv(paths.ohlcv_5m, index=False)

    chain_df = _concat_frames(chain_frames)
    if not chain_df.empty:
        chain_df.to_csv(paths.option_chain_decision_times, index=False)
        enrichment = enrich_structured_chain_deltas(paths.root)
        if enrichment.rows_enriched:
            notes.append(f"Derived Black-Scholes delta for {enrichment.rows_enriched} historical chain rows.")
        if chain_df["bid"].replace("", pd.NA).isna().all():
            notes.append("Decision-time chain snapshots were built from rolling-option history; bid/ask is still missing for most rows.")

    fills_df = _build_execution_fills(trade_blotter_path, scrip_master_path)
    if not fills_df.empty:
        fills_df.to_csv(paths.execution_fills, index=False)
    else:
        notes.append("Execution fills remain empty or partial; structured slippage history is not yet available.")

    labels_df = pd.DataFrame(labels)
    if not labels_df.empty:
        labels_df = labels_df.drop_duplicates(subset=["session_date"], keep="last").sort_values("session_date")
        labels_df.to_csv(paths.session_labels, index=False)

    return BuildStructuredDatasetReport(
        data_root=str(paths.root),
        rolling_root=str(rolling_path),
        processed_sessions=processed_sessions,
        ohlcv_rows=int(len(ohlcv_df)),
        chain_rows=int(len(chain_df)),
        fill_rows=int(len(fills_df)),
        label_rows=int(len(labels_df)),
        used_option_chain_db=chain_lookup.available,
        from_date=from_date,
        to_date=to_date,
        notes=notes,
    )


def refresh_structured_training_dataset_from_rolling(
    rolling_root: str | Path,
    data_root: str | Path,
    *,
    trade_blotter_path: str | Path | None = None,
    scrip_master_path: str | Path | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> StructuredTrainingRefreshReport:
    paths = scaffold_dataset_layout(data_root)
    rolling_path = Path(rolling_root)
    session_dirs = sorted(p for p in rolling_path.iterdir() if p.is_dir())
    if from_date:
        session_dirs = [p for p in session_dirs if p.name >= from_date]
    if to_date:
        session_dirs = [p for p in session_dirs if p.name <= to_date]

    ohlcv_frames: list[pd.DataFrame] = []
    labels: list[dict[str, str]] = []
    notes: list[str] = []

    for session_dir in session_dirs:
        parquet_path = session_dir / "rolling_option.parquet"
        if not parquet_path.exists():
            continue
        session_df = pd.read_parquet(parquet_path)
        if session_df.empty:
            continue
        spot_df = _prepare_spot_series(session_df)
        bars_5m = _build_5m_bars(spot_df)
        if bars_5m.empty:
            continue
        ohlcv_frames.append(bars_5m)
        labels.append(
            {
                "session_date": session_dir.name,
                "label": _classify_session_label(bars_5m),
            }
        )

    ohlcv_df = _concat_frames(ohlcv_frames)
    if not ohlcv_df.empty:
        ohlcv_df.to_csv(paths.ohlcv_5m, index=False)
    else:
        pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]).to_csv(paths.ohlcv_5m, index=False)
        notes.append("No rolling-option sessions matched the requested window for 5-minute OHLCV refresh.")

    labels_df = pd.DataFrame(labels).drop_duplicates(subset=["session_date"], keep="last") if labels else pd.DataFrame(columns=["session_date", "label"])
    labels_df.to_csv(paths.session_labels, index=False)

    fills_df = _build_execution_fills(trade_blotter_path, scrip_master_path)
    if fills_df.empty:
        fills_df = pd.DataFrame(
            columns=[
                "session_date",
                "timestamp",
                "strategy",
                "side",
                "strike",
                "option_type",
                "quoted_mid",
                "fill_price",
                "quantity",
                "slippage_points",
                "pnl_rupees",
            ]
        )
        notes.append("Execution fills refresh found no blotter-backed fills to write.")
    fills_df.to_csv(paths.execution_fills, index=False)

    preserved_chain_rows = 0
    if paths.option_chain_decision_times.exists():
        try:
            preserved_chain_rows = int(len(pd.read_csv(paths.option_chain_decision_times)))
        except Exception:
            preserved_chain_rows = 0

    return StructuredTrainingRefreshReport(
        data_root=str(paths.root),
        rolling_root=str(rolling_path),
        trading_days=int(ohlcv_df["timestamp"].astype(str).str[:10].nunique()) if not ohlcv_df.empty else 0,
        ohlcv_rows=int(len(ohlcv_df)),
        fill_rows=int(len(fills_df)),
        label_rows=int(len(labels_df)),
        preserved_chain_rows=preserved_chain_rows,
        from_date=from_date,
        to_date=to_date,
        notes=notes,
    )


def enrich_structured_chain_deltas(
    root: str | Path,
    *,
    risk_free_rate: float = 0.06,
) -> DeltaEnrichmentReport:
    paths = StructuredDatasetPaths.from_root(root)
    if not paths.option_chain_decision_times.exists():
        return DeltaEnrichmentReport(
            data_root=str(paths.root),
            total_rows=0,
            rows_missing_delta=0,
            rows_enriched=0,
            delta_coverage_pct=0.0,
            notes=["option_chain_decision_times.csv is missing."],
        )

    df = pd.read_csv(paths.option_chain_decision_times)
    if df.empty:
        return DeltaEnrichmentReport(
            data_root=str(paths.root),
            total_rows=0,
            rows_missing_delta=0,
            rows_enriched=0,
            delta_coverage_pct=0.0,
            notes=["option_chain_decision_times.csv is empty."],
        )

    if "delta_raw" not in df.columns:
        df["delta_raw"] = df.get("delta")
    if "delta_source" not in df.columns:
        df["delta_source"] = ""
    df["delta_raw"] = df["delta_raw"].astype(object)
    df["delta_source"] = df["delta_source"].astype(object)

    original_delta = df["delta"].copy()
    total_rows = int(len(df))
    missing_mask = original_delta.isna() | (original_delta.astype(str).str.strip() == "")
    rows_missing_delta = int(missing_mask.sum())
    rows_enriched = 0

    if rows_missing_delta:
        work = df.loc[missing_mask, ["spot", "strike", "iv", "option_type", "timestamp", "expiry"]].copy()
        work["_timestamp_dt"] = pd.to_datetime(work["timestamp"], errors="coerce")
        work["_expiry_dt"] = pd.to_datetime(work["expiry"], errors="coerce")
    else:
        work = pd.DataFrame()

    for idx, row in work.iterrows():
        derived_delta, inferred_expiry = _derive_delta_row(
            row,
            risk_free_rate=risk_free_rate,
            timestamp_value=row.get("_timestamp_dt"),
            expiry_value=row.get("_expiry_dt"),
            session_date_value=df.at[idx, "session_date"] if "session_date" in df.columns else None,
        )
        if derived_delta is None:
            continue
        df.at[idx, "delta"] = round(derived_delta, 6)
        df.at[idx, "delta_source"] = "BS_IV_DERIVED_INFERRED_EXPIRY" if inferred_expiry else "BS_IV_DERIVED"
        rows_enriched += 1

    provided_mask = ~(original_delta.isna() | (original_delta.astype(str).str.strip() == ""))
    if provided_mask.any():
        df.loc[provided_mask & (df["delta_source"].astype(str).str.strip() == ""), "delta_source"] = "PROVIDED"

    delta_present_mask = ~(df["delta"].isna() | (df["delta"].astype(str).str.strip() == ""))
    df.to_csv(paths.option_chain_decision_times, index=False)

    notes: list[str] = []
    if rows_missing_delta and not rows_enriched:
        notes.append("No missing delta rows could be derived from IV/spot/strike/time-to-expiry.")
    return DeltaEnrichmentReport(
        data_root=str(paths.root),
        total_rows=total_rows,
        rows_missing_delta=rows_missing_delta,
        rows_enriched=rows_enriched,
        delta_coverage_pct=float(delta_present_mask.mean()) if total_rows else 0.0,
        notes=notes,
    )


def _count_trading_days(path: Path) -> int:
    days: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = row.get("timestamp")
            if not ts:
                continue
            days.add(str(ts)[:10])
    return len(days)


def _assess_chain_snapshots(path: Path) -> dict[str, float]:
    by_day_time: dict[tuple[str, str], int] = {}
    rows = 0
    rows_with_delta = 0
    rows_with_iv = 0
    rows_with_bid_ask = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            session_date = row.get("session_date") or str(row.get("timestamp", ""))[:10]
            decision_time = row.get("decision_time") or str(row.get("timestamp", ""))[11:16]
            if session_date and decision_time in REQUIRED_DECISION_TIMES:
                by_day_time[(session_date, decision_time)] = by_day_time.get((session_date, decision_time), 0) + 1
            if row.get("delta") not in {None, ""}:
                rows_with_delta += 1
            if row.get("iv") not in {None, ""}:
                rows_with_iv += 1
            if row.get("bid") not in {None, ""} and row.get("ask") not in {None, ""}:
                rows_with_bid_ask += 1
    session_days = len({day for day, _ in by_day_time})
    expected_slots = max(session_days * len(REQUIRED_DECISION_TIMES), 1)
    covered_slots = sum(1 for count in by_day_time.values() if count > 0)
    return {
        "decision_time_coverage_pct": covered_slots / expected_slots,
        "delta_coverage_pct": rows_with_delta / rows if rows else 0.0,
        "iv_coverage_pct": rows_with_iv / rows if rows else 0.0,
        "bid_ask_coverage_pct": rows_with_bid_ask / rows if rows else 0.0,
    }


def _assess_execution_fills(path: Path) -> dict[str, float]:
    rows = 0
    with_slippage = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            if row.get("slippage_points") not in {None, ""}:
                with_slippage += 1
    return {"slippage_coverage_pct": with_slippage / rows if rows else 0.0}


def _assess_session_labels(path: Path, trading_days: int) -> float:
    labelled_days: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            session_date = row.get("session_date")
            label = (row.get("label") or "").strip()
            if session_date and label in ALLOWED_SESSION_LABELS:
                labelled_days.add(session_date)
    return len(labelled_days) / trading_days if trading_days else 0.0


def _write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content)


def _prepare_spot_series(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(
        work["tradeDate"].astype(str) + " " + work["tradeTime"].astype(str),
        errors="coerce",
    )
    work["spot"] = pd.to_numeric(work.get("spot"), errors="coerce")
    work["proxy_volume"] = pd.to_numeric(work.get("volume"), errors="coerce").fillna(0.0)
    work = work.dropna(subset=["timestamp", "spot"])
    work = _filter_session_spot_outliers(work)
    if work.empty:
        return pd.DataFrame(columns=["timestamp", "spot", "proxy_volume"])
    spot = (
        work.groupby("timestamp", as_index=False)
        .agg(
            spot=("spot", "last"),
            proxy_volume=("proxy_volume", "sum"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return spot


def _build_5m_bars(spot_df: pd.DataFrame) -> pd.DataFrame:
    if spot_df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    work = spot_df.copy().set_index("timestamp").sort_index()
    spot = work["spot"].astype(float)
    volume = work["proxy_volume"].astype(float) if "proxy_volume" in work else pd.Series(1.0, index=work.index)
    bars = spot.resample("5min").ohlc().dropna(how="any")
    if bars.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    bars["volume"] = volume.resample("5min").sum().reindex(bars.index).fillna(0.0)
    if (bars["volume"] <= 0).all():
        bars["volume"] = 1.0
    bars = bars.reset_index()
    bars["timestamp"] = bars["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return bars[["timestamp", "open", "high", "low", "close", "volume"]]


def _build_decision_time_chain(
    df: pd.DataFrame,
    *,
    session_date: str,
    chain_lookup: "_HistoricalChainLookup",
    trading_days: set[date] | None = None,
) -> pd.DataFrame:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(
        work["tradeDate"].astype(str) + " " + work["tradeTime"].astype(str),
        errors="coerce",
    )
    work["strikePrice"] = pd.to_numeric(work.get("strikePrice"), errors="coerce")
    work["ltp"] = pd.to_numeric(work.get("ltp"), errors="coerce")
    work["delta"] = pd.to_numeric(work.get("delta"), errors="coerce")
    work["iv"] = pd.to_numeric(work.get("iv"), errors="coerce")
    work["oi"] = pd.to_numeric(work.get("oi"), errors="coerce")
    work["spot"] = pd.to_numeric(work.get("spot"), errors="coerce")
    work = work.dropna(subset=["timestamp", "strikePrice"])
    work = _filter_session_spot_outliers(work)
    if work.empty:
        return pd.DataFrame()

    available = sorted(work["timestamp"].dropna().unique())
    rows: list[dict[str, object]] = []
    for decision_time in REQUIRED_DECISION_TIMES:
        target = pd.Timestamp(f"{session_date} {decision_time}:00")
        nearest = _nearest_timestamp(available, target)
        if nearest is None or abs(nearest - target) > pd.Timedelta(minutes=5):
            continue
        snap = work.loc[work["timestamp"] == nearest].copy()
        inferred_expiry = infer_nifty_weekly_expiry(session_date, trading_days=trading_days)
        for record in snap.to_dict(orient="records"):
            expiry = inferred_expiry.isoformat() if inferred_expiry is not None else _resolve_expiry_text(
                record.get("expiryDate"),
                session_date=session_date,
            )[0]
            option_type = _normalize_option_type(record.get("optionType"))
            if not option_type:
                continue
            strike = float(record["strikePrice"])
            bid, ask = chain_lookup.lookup(
                as_of_date=session_date,
                expiry=expiry,
                strike=strike,
                option_type=option_type,
            )
            rows.append(
                {
                    "session_date": session_date,
                    "timestamp": pd.Timestamp(nearest).strftime("%Y-%m-%dT%H:%M:%S"),
                    "decision_time": decision_time,
                    "expiry": expiry,
                    "spot": record.get("spot"),
                    "strike": int(round(strike)),
                    "option_type": option_type,
                    "bid": bid,
                    "ask": ask,
                    "ltp": record.get("ltp"),
                    "delta": record.get("delta"),
                    "delta_raw": record.get("delta"),
                    "delta_source": "PROVIDED" if pd.notna(record.get("delta")) else "",
                    "iv": record.get("iv"),
                    "oi": record.get("oi"),
                }
            )
    return pd.DataFrame(rows)


def _build_execution_fills(
    trade_blotter_path: str | Path | None,
    scrip_master_path: str | Path | None,
) -> pd.DataFrame:
    if not trade_blotter_path:
        return pd.DataFrame(
            columns=[
                "session_date",
                "timestamp",
                "strategy",
                "side",
                "strike",
                "option_type",
                "quoted_mid",
                "fill_price",
                "quantity",
                "slippage_points",
                "pnl_rupees",
            ]
        )
    blotter_path = Path(trade_blotter_path)
    if not blotter_path.exists():
        return pd.DataFrame()

    fills = pd.read_csv(blotter_path)
    if fills.empty:
        return pd.DataFrame()
    fills["timestamp"] = pd.to_datetime(fills.get("timestamp"), errors="coerce")
    fills["price"] = pd.to_numeric(fills.get("price"), errors="coerce")
    fills["quantity"] = pd.to_numeric(fills.get("quantity"), errors="coerce")
    fills["security_id"] = pd.to_numeric(fills.get("security_id"), errors="coerce").astype("Int64")
    fills = fills.dropna(subset=["timestamp", "price", "quantity"])
    if fills.empty:
        return pd.DataFrame()

    option_type_map = _load_option_type_map(scrip_master_path)
    fills["option_type"] = fills["security_id"].map(option_type_map).fillna("")
    out = pd.DataFrame(
        {
            "session_date": fills["timestamp"].dt.strftime("%Y-%m-%d"),
            "timestamp": fills["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "strategy": fills.get("tag", "").fillna(""),
            "side": fills.get("side", "").fillna(""),
            "strike": pd.to_numeric(fills.get("strike"), errors="coerce"),
            "option_type": fills["option_type"],
            "quoted_mid": "",
            "fill_price": fills["price"],
            "quantity": fills["quantity"],
            "slippage_points": "",
            "pnl_rupees": "",
        }
    )
    return out


def _load_option_type_map(scrip_master_path: str | Path | None) -> dict[int, str]:
    if not scrip_master_path:
        return {}
    path = Path(scrip_master_path)
    if not path.exists():
        return {}
    cols = ["SEM_SMST_SECURITY_ID", "SEM_OPTION_TYPE"]
    master = pd.read_csv(path, usecols=cols)
    option_type = master["SEM_OPTION_TYPE"].astype(str).str.upper().map({"CE": "CALL", "PE": "PUT"})
    keys = pd.to_numeric(master["SEM_SMST_SECURITY_ID"], errors="coerce").dropna().astype(int)
    return dict(zip(keys.tolist(), option_type.fillna("").tolist()))


def _nearest_timestamp(values: list[pd.Timestamp], target: pd.Timestamp) -> pd.Timestamp | None:
    if not values:
        return None
    return min(values, key=lambda value: abs(pd.Timestamp(value) - target))


def _normalize_option_type(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"CE", "CALL"}:
        return "CALL"
    if text in {"PE", "PUT"}:
        return "PUT"
    return ""


def _derive_delta_row(
    row: pd.Series,
    *,
    risk_free_rate: float,
    timestamp_value: object | None = None,
    expiry_value: object | None = None,
    session_date_value: object | None = None,
) -> tuple[float | None, bool]:
    spot = float_or_none(row.get("spot"))
    strike = float_or_none(row.get("strike"))
    iv = float_or_none(row.get("iv"))
    option_type = _normalize_option_type(row.get("option_type"))
    timestamp = pd.to_datetime(timestamp_value if timestamp_value is not None else row.get("timestamp"), errors="coerce")
    expiry, inferred_expiry = _resolve_expiry_date(
        expiry_value if expiry_value is not None else row.get("expiry"),
        session_date=session_date_value if session_date_value is not None else row.get("session_date"),
    )
    if not option_type or pd.isna(timestamp) or expiry is None:
        return None, inferred_expiry
    if spot is None or strike is None or iv is None:
        return None, inferred_expiry
    return derive_delta_from_iv(
        spot=spot,
        strike=strike,
        option_type=option_type,
        iv=iv,
        as_of=timestamp.to_pydatetime(),
        expiry=expiry,
        risk_free_rate=risk_free_rate,
    ), inferred_expiry


def _resolve_expiry_text(expiry_value: object, *, session_date: object | None = None) -> tuple[str, bool]:
    expiry, inferred = _resolve_expiry_date(expiry_value, session_date=session_date)
    return (expiry.isoformat() if expiry else "", inferred)


def _resolve_expiry_date(expiry_value: object, *, session_date: object | None = None) -> tuple[date | None, bool]:
    expiry = pd.to_datetime(expiry_value, errors="coerce")
    inferred = _infer_weekly_expiry(session_date)
    if not pd.isna(expiry):
        expiry_date = expiry.date()
        if inferred is not None:
            session_dt = pd.to_datetime(session_date, errors="coerce")
            if not pd.isna(session_dt):
                session_day = session_dt.date()
                days_to_expiry = (expiry_date - session_day).days
                # Rolling intraday option history can contain dummy far-expiry values.
                # This module is weekly-expiry only, so treat far-dated expiries as invalid.
                if days_to_expiry < 0 or days_to_expiry > MAX_WEEKLY_EXPIRY_DAYS:
                    return inferred, True
        return expiry_date, False
    return inferred, inferred is not None


def _infer_weekly_expiry(session_date_value: object) -> date | None:
    return infer_nifty_weekly_expiry(session_date_value)


def _filter_session_spot_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if "spot" not in df.columns or df.empty:
        return df
    work = df.copy()
    spot = pd.to_numeric(work["spot"], errors="coerce")
    median_spot = float(spot.median()) if spot.notna().any() else 0.0
    if median_spot <= 0:
        return work
    lower = median_spot * (1.0 - SESSION_SPOT_OUTLIER_PCT)
    upper = median_spot * (1.0 + SESSION_SPOT_OUTLIER_PCT)
    filtered = work.loc[spot.between(lower, upper)].copy()
    return filtered if not filtered.empty else work


def _classify_session_label(bars_5m: pd.DataFrame) -> str:
    if bars_5m.empty:
        return "range"
    work = bars_5m.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
    if len(work) < 6:
        return "range"

    open_price = float(work.iloc[0]["open"])
    close_price = float(work.iloc[-1]["close"])
    day_high = float(work["high"].max())
    day_low = float(work["low"].min())
    day_range = max(day_high - day_low, 0.0)
    if open_price <= 0 or day_range <= 0:
        return "range"

    or_window = work.loc[work["timestamp"].dt.time <= datetime.strptime("09:45", "%H:%M").time()]
    if or_window.empty:
        or_window = work.head(min(6, len(work)))
    or_high = float(or_window["high"].max())
    or_low = float(or_window["low"].min())
    later = work.loc[work["timestamp"].dt.time >= datetime.strptime("10:00", "%H:%M").time()]
    if later.empty:
        later = work

    close_above = float((later["close"] > or_high).mean())
    close_below = float((later["close"] < or_low).mean())
    close_inside = float(((later["close"] >= or_low) & (later["close"] <= or_high)).mean())
    range_pct = day_range / open_price

    if close_above >= 0.55 and close_price > open_price:
        return "trend_up"
    if close_below >= 0.55 and close_price < open_price:
        return "trend_down"
    if close_inside >= 0.60 and range_pct <= 0.0125:
        return "range"
    return "event_day"


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


class _HistoricalChainLookup:
    def __init__(self, db_path: str | Path | None):
        self.db_path = Path(db_path) if db_path else None
        self.available = bool(self.db_path and self.db_path.exists())
        self._cache: dict[tuple[str, str, float, str], tuple[float | str, float | str]] = {}

    def lookup(
        self,
        *,
        as_of_date: str,
        expiry: str,
        strike: float,
        option_type: str,
    ) -> tuple[float | str, float | str]:
        if not self.available or not expiry:
            return "", ""
        key = (as_of_date, expiry, round(float(strike), 2), option_type)
        if key in self._cache:
            return self._cache[key]
        with sqlite3.connect(str(self.db_path)) as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT ce_json, pe_json
                FROM option_chain
                WHERE as_of_date = ? AND expiry = ? AND ABS(strike - ?) < 0.01
                LIMIT 1
                """,
                (as_of_date, expiry, float(strike)),
            )
            row = cur.fetchone()
        if not row:
            self._cache[key] = ("", "")
            return self._cache[key]
        payload = json.loads((row[0] if option_type == "CALL" else row[1]) or "{}")
        bid = payload.get("top_bid_price", "")
        ask = payload.get("top_ask_price", "")
        self._cache[key] = (bid, ask)
        return self._cache[key]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
