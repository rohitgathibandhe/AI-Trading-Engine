from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dt_time, timedelta
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from market_ai.modules.data_fetch.dhan_rolling_option import RollingOptionConfig, RollingOptionIngestor

from .collector import IST, _is_trading_day, _load_dhan_creds, capture_decision_time_snapshot
from .data_pipeline_health import enrich_status_payload
from .dataset import (
    DatasetCoverageReport,
    DeltaEnrichmentReport,
    StructuredTrainingRefreshReport,
    assess_structured_dataset,
    enrich_structured_chain_deltas,
    refresh_structured_training_dataset_from_rolling,
)
from .research import (
    ResearchDatasetAppendReport,
    append_research_backtest_dataset_from_rolling,
    build_monitor_times,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingDataRefreshReport:
    as_of: str
    rolling_fetch_start: str | None
    rolling_fetch_end: str | None
    rolling_days_written: int
    structured_refresh: dict[str, object]
    delta_enrichment: dict[str, object]
    research_append: dict[str, object]
    coverage: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def refresh_training_data(
    *,
    rolling_root: str | Path,
    structured_root: str | Path,
    research_root: str | Path,
    trade_blotter_path: str | Path | None = None,
    scrip_master_path: str | Path | None = None,
    option_chain_db_path: str | Path | None = None,
    raw_snapshot_root: str | Path | None = None,
    status_path: str | Path | None = None,
    start_date: str = "2025-01-01",
    to_date: str | None = None,
    underlying: str = "NIFTY",
    security_id: int = 13,
    segment: str = "NSE_FNO",
    interval: str = "1",
    monitor_times: Sequence[str] | None = None,
    creds_path: str | Path | None = None,
    tolerate_fetch_errors: bool = False,
) -> TrainingDataRefreshReport:
    resolved_to_date = to_date or datetime.now(IST).date().isoformat()
    rolling_start: str | None = None
    rolling_end: str | None = None
    rolling_days_written = 0
    fetch_error: str | None = None
    try:
        try:
            rolling_start, rolling_end, rolling_days_written = _fetch_missing_rolling_history(
                rolling_root=rolling_root,
                start_date=start_date,
                to_date=resolved_to_date,
                underlying=underlying,
                security_id=security_id,
                segment=segment,
                interval=interval,
                creds_path=creds_path,
            )
        except Exception as exc:
            if not tolerate_fetch_errors:
                raise
            fetch_error = str(exc)
            logger.warning("Rolling history refresh skipped due to fetch error: %s", exc)
        structured_report = refresh_structured_training_dataset_from_rolling(
            rolling_root,
            structured_root,
            trade_blotter_path=trade_blotter_path,
            scrip_master_path=scrip_master_path,
            from_date=start_date,
            to_date=resolved_to_date,
        )
        delta_report = enrich_structured_chain_deltas(structured_root)
        research_report = append_research_backtest_dataset_from_rolling(
            rolling_root,
            research_root,
            from_date=start_date,
            to_date=resolved_to_date,
            monitor_times=tuple(monitor_times) if monitor_times else None,
        )
        coverage = assess_structured_dataset(structured_root)
        report = TrainingDataRefreshReport(
            as_of=datetime.now(IST).isoformat(),
            rolling_fetch_start=rolling_start,
            rolling_fetch_end=rolling_end,
            rolling_days_written=rolling_days_written,
            structured_refresh=structured_report.to_dict(),
            delta_enrichment=delta_report.to_dict(),
            research_append=research_report.to_dict(),
            coverage=coverage.to_dict(),
        )
        if status_path:
            payload = {"status": "REFRESHED", **report.to_dict()}
            if fetch_error:
                payload["fetch_error"] = fetch_error
            _write_status(Path(status_path), payload)
        return report
    except Exception as exc:
        if status_path:
            _write_status(
                Path(status_path),
                {
                    "status": "REFRESH_ERROR",
                    "as_of": datetime.now(IST).isoformat(),
                    "to_date": resolved_to_date,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            )
        raise


def run_training_data_pipeline(
    *,
    rolling_root: str | Path,
    structured_root: str | Path,
    research_root: str | Path,
    trade_blotter_path: str | Path | None = None,
    scrip_master_path: str | Path | None = None,
    option_chain_db_path: str | Path | None = None,
    raw_snapshot_root: str | Path | None = None,
    status_path: str | Path | None = None,
    start_date: str = "2025-01-01",
    to_date: str | None = None,
    underlying: str = "NIFTY",
    security_id: int = 13,
    underlying_seg: str = "IDX_I",
    rolling_segment: str = "NSE_FNO",
    interval: str = "1",
    monitor_start: str = "09:30",
    monitor_end: str = "15:00",
    monitor_step_minutes: int = 5,
    refresh_after: str = "15:20",
    poll_seconds: float = 30.0,
    creds_path: str | Path | None = None,
    now_provider: Any | None = None,
    max_days: int | None = None,
) -> list[TrainingDataRefreshReport]:
    now_fn = now_provider or (lambda: datetime.now(IST))
    decision_times = build_monitor_times(start=monitor_start, end=monitor_end, step_minutes=monitor_step_minutes)
    status_file = Path(status_path) if status_path else None
    reports: list[TrainingDataRefreshReport] = []
    current_day: date | None = None
    captured_times: set[str] = set()
    refreshed_days: set[date] = set()

    while True:
        now = now_fn().astimezone(IST)
        today = now.date()
        if current_day != today:
            current_day = today
            captured_times = _load_recorded_times(structured_root, today.isoformat())

        if status_file:
            pending_times = [item for item in decision_times if item not in captured_times]
            if not _is_trading_day(today):
                idle_status = {
                    "status": "IDLE_NON_TRADING_DAY",
                    "as_of": now.isoformat(),
                    "session_date": today.isoformat(),
                }
            elif now.strftime("%H:%M") < "09:15":
                idle_status = {
                    "status": "PREMARKET_WAIT",
                    "as_of": now.isoformat(),
                    "session_date": today.isoformat(),
                    "captured_times": sorted(captured_times),
                    "pending_times": pending_times,
                }
            elif now.strftime("%H:%M") <= monitor_end:
                idle_status = {
                    "status": "WAITING_TO_CAPTURE",
                    "as_of": now.isoformat(),
                    "session_date": today.isoformat(),
                    "captured_times": sorted(captured_times),
                    "pending_times": pending_times,
                }
            elif today in refreshed_days:
                idle_status = {
                    "status": "REFRESHED_FOR_SESSION",
                    "as_of": now.isoformat(),
                    "session_date": today.isoformat(),
                    "captured_times": sorted(captured_times),
                    "pending_times": pending_times,
                }
            else:
                idle_status = {
                    "status": "WAITING_FOR_REFRESH_WINDOW",
                    "as_of": now.isoformat(),
                    "session_date": today.isoformat(),
                    "captured_times": sorted(captured_times),
                    "pending_times": pending_times,
                    "refresh_after": refresh_after,
                }
            _write_status(status_file, idle_status)

        if _is_trading_day(today):
            if "09:15" <= now.strftime("%H:%M") <= monitor_end:
                for decision_time in decision_times:
                    if decision_time in captured_times or now.strftime("%H:%M") < decision_time:
                        continue
                    try:
                        capture_decision_time_snapshot(
                            structured_root,
                            underlying_id=security_id,
                            underlying_seg=underlying_seg,
                            decision_time=decision_time,
                            timestamp=now,
                            raw_snapshot_root=raw_snapshot_root,
                            creds_path=creds_path,
                        )
                        captured_times.add(decision_time)
                        if status_file:
                            _write_status(
                                status_file,
                                {
                                    "status": "CAPTURING",
                                    "as_of": now.isoformat(),
                                    "session_date": today.isoformat(),
                                    "captured_times": sorted(captured_times),
                                    "pending_times": [item for item in decision_times if item not in captured_times],
                                },
                            )
                    except Exception as exc:
                        logger.warning("Chain snapshot capture failed for %s %s: %s", today.isoformat(), decision_time, exc)
                        if status_file:
                            _write_status(
                                status_file,
                                {
                                    "status": "CAPTURE_ERROR",
                                    "as_of": now.isoformat(),
                                    "session_date": today.isoformat(),
                                    "decision_time": decision_time,
                                    "error": str(exc),
                                    "captured_times": sorted(captured_times),
                                },
                            )
            if now.strftime("%H:%M") >= refresh_after and today not in refreshed_days:
                try:
                    report = refresh_training_data(
                        rolling_root=rolling_root,
                        structured_root=structured_root,
                        research_root=research_root,
                        trade_blotter_path=trade_blotter_path,
                        scrip_master_path=scrip_master_path,
                        option_chain_db_path=option_chain_db_path,
                        start_date=start_date,
                        to_date=min(today.isoformat(), to_date) if to_date else today.isoformat(),
                        underlying=underlying,
                        security_id=security_id,
                        segment=rolling_segment,
                        interval=interval,
                        monitor_times=decision_times,
                        creds_path=creds_path,
                        tolerate_fetch_errors=True,
                    )
                    reports.append(report)
                    refreshed_days.add(today)
                    if status_file:
                        payload = report.to_dict()
                        payload["status"] = "REFRESHED"
                        _write_status(status_file, payload)
                    if max_days is not None and len(reports) >= max_days:
                        break
                except Exception as exc:
                    logger.warning("Training data refresh failed for %s: %s", today.isoformat(), exc)
                    if status_file:
                        _write_status(
                            status_file,
                            {
                                "status": "REFRESH_ERROR",
                                "as_of": now.isoformat(),
                                "session_date": today.isoformat(),
                                "error": str(exc),
                            },
                        )
        time.sleep(max(float(poll_seconds), 1.0))
    return reports


def _fetch_missing_rolling_history(
    *,
    rolling_root: str | Path,
    start_date: str,
    to_date: str,
    underlying: str,
    security_id: int,
    segment: str,
    interval: str,
    creds_path: str | Path | None = None,
) -> tuple[str | None, str | None, int]:
    rolling_path = Path(rolling_root)
    rolling_path.mkdir(parents=True, exist_ok=True)
    _load_dhan_creds(creds_path)
    day_dirs = sorted(path.name for path in rolling_path.iterdir() if path.is_dir())
    if day_dirs:
        last_existing = day_dirs[-1]
        fetch_start = max(pd_date(start_date), pd_date(last_existing) + timedelta(days=1))
    else:
        fetch_start = pd_date(start_date)
    fetch_end = pd_date(to_date)
    if fetch_start > fetch_end:
        return None, None, 0
    ingestor = RollingOptionIngestor(out_dir=rolling_path)
    cfg = RollingOptionConfig(
        underlying=underlying,
        segment=segment,
        security_id=security_id,
        start=datetime.combine(fetch_start, dt_time(9, 15)),
        end=datetime.combine(fetch_end, dt_time(15, 30)),
        interval=interval,
    )
    written = ingestor.fetch_range(cfg)
    written_days = sorted({path.parent.name for path in written})
    return fetch_start.isoformat(), fetch_end.isoformat(), len(written_days)


def _load_recorded_times(data_root: str | Path, session_date: str) -> set[str]:
    chain_path = Path(data_root) / "option_chain_decision_times.csv"
    if not chain_path.exists():
        return set()
    try:
        frame = pd.read_csv(chain_path, usecols=["session_date", "decision_time"])
    except Exception:
        return set()
    frame["session_date"] = frame["session_date"].astype(str)
    frame["decision_time"] = frame["decision_time"].astype(str)
    day_frame = frame.loc[frame["session_date"] == session_date]
    return set(day_frame["decision_time"].dropna().tolist())


def _write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_root = path.parent
    enriched = enrich_status_payload(
        payload,
        state_root=state_root,
        structured_root=state_root / "intraday_structured_dataset",
        research_root=state_root / "intraday_research_dataset_2025_01_01_2026_03_30_5m",
    )
    path.write_text(json.dumps(enriched, indent=2, sort_keys=True))


def pd_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()
