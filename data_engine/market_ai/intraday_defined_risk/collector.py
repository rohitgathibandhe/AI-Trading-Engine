from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from market_ai.modules.data_fetch.dhan_api import (
    DhanError,
    SimpleDhanClient,
    _post_json_with_backoff,
    get_expiry_list_for_underlying,
)

from .dataset import REQUIRED_DECISION_TIMES, scaffold_dataset_layout


IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class ChainCaptureReport:
    data_root: str
    expiry: str
    timestamp: str
    decision_time: str
    rows_written: int
    raw_json_path: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def run_daily_chain_schedule(
    data_root: str | Path,
    *,
    decision_times: Sequence[str] = REQUIRED_DECISION_TIMES,
    underlying_id: int = 13,
    underlying_seg: str = "IDX_I",
    expiry: str | None = None,
    raw_snapshot_root: str | Path | None = None,
    creds_path: str | Path | None = None,
    poll_seconds: float = 15.0,
    now_provider: Any | None = None,
    max_reports: int | None = None,
    max_days: int | None = None,
) -> list[ChainCaptureReport]:
    now_fn = now_provider or (lambda: datetime.now(IST))
    schedule = sorted({t.strip() for t in decision_times if t and t.strip()})
    reports: list[ChainCaptureReport] = []
    current_day: date | None = None
    pending: list[str] = []
    day_finished = False
    completed_days = 0

    while True:
        now = now_fn().astimezone(IST)
        today = now.date()
        if today != current_day:
            current_day = today
            pending = list(schedule)
            day_finished = False

        if not _is_trading_day(today):
            time.sleep(max(float(poll_seconds), 1.0))
            continue

        current_hhmm = now.strftime("%H:%M")
        if current_hhmm >= "15:20" and not day_finished:
            pending = []
            day_finished = True
            completed_days += 1
            if max_days is not None and completed_days >= max_days:
                break
            time.sleep(max(float(poll_seconds), 1.0))
            continue

        if current_hhmm < "09:15":
            time.sleep(max(float(poll_seconds), 1.0))
            continue

        for target in list(pending):
            if current_hhmm >= target:
                report = capture_decision_time_snapshot(
                    data_root,
                    underlying_id=underlying_id,
                    underlying_seg=underlying_seg,
                    expiry=expiry,
                    decision_time=target,
                    timestamp=now,
                    raw_snapshot_root=raw_snapshot_root,
                    creds_path=creds_path,
                )
                reports.append(report)
                pending.remove(target)
                if max_reports is not None and len(reports) >= max_reports:
                    return reports

        if not pending and not day_finished and current_hhmm >= "15:15":
            day_finished = True
            completed_days += 1
            if max_days is not None and completed_days >= max_days:
                break

        time.sleep(max(float(poll_seconds), 1.0))

    return reports


def capture_decision_time_snapshot(
    data_root: str | Path,
    *,
    underlying_id: int = 13,
    underlying_seg: str = "IDX_I",
    expiry: str | None = None,
    decision_time: str | None = None,
    timestamp: datetime | None = None,
    raw_snapshot_root: str | Path | None = None,
    creds_path: str | Path | None = None,
) -> ChainCaptureReport:
    ts = timestamp.astimezone(IST) if timestamp else datetime.now(IST)
    decision = decision_time or ts.strftime("%H:%M")
    resolved_expiry = expiry or _resolve_front_expiry(underlying_id, underlying_seg, creds_path=creds_path, now_ist=ts)
    payload = _fetch_option_chain_payload(
        underlying_id=underlying_id,
        underlying_seg=underlying_seg,
        expiry=resolved_expiry,
        creds_path=creds_path,
    )
    rows = _flatten_option_chain_payload(
        payload,
        timestamp=ts,
        decision_time=decision,
        expiry=resolved_expiry,
    )
    dataset_paths = scaffold_dataset_layout(data_root)
    _append_rows(dataset_paths.option_chain_decision_times, rows)

    raw_path = None
    if raw_snapshot_root:
        raw_path = _write_raw_snapshot(raw_snapshot_root, resolved_expiry, ts, payload)

    return ChainCaptureReport(
        data_root=str(dataset_paths.root),
        expiry=resolved_expiry,
        timestamp=ts.isoformat(),
        decision_time=decision,
        rows_written=len(rows),
        raw_json_path=str(raw_path) if raw_path else None,
    )


def run_decision_time_schedule(
    data_root: str | Path,
    *,
    decision_times: Sequence[str] = REQUIRED_DECISION_TIMES,
    underlying_id: int = 13,
    underlying_seg: str = "IDX_I",
    expiry: str | None = None,
    raw_snapshot_root: str | Path | None = None,
    creds_path: str | Path | None = None,
    poll_seconds: float = 15.0,
    now_provider: Any | None = None,
) -> list[ChainCaptureReport]:
    now_fn = now_provider or (lambda: datetime.now(IST))
    pending = sorted({t.strip() for t in decision_times if t and t.strip()})
    reports: list[ChainCaptureReport] = []
    done: set[str] = set()

    while pending:
        now = now_fn().astimezone(IST)
        current_hhmm = now.strftime("%H:%M")
        for target in list(pending):
            if current_hhmm >= target and target not in done:
                report = capture_decision_time_snapshot(
                    data_root,
                    underlying_id=underlying_id,
                    underlying_seg=underlying_seg,
                    expiry=expiry,
                    decision_time=target,
                    timestamp=now,
                    raw_snapshot_root=raw_snapshot_root,
                    creds_path=creds_path,
                )
                reports.append(report)
                done.add(target)
                pending.remove(target)
        if not pending:
            break
        time.sleep(max(float(poll_seconds), 1.0))
    return reports


def _default_dhan_creds_candidates() -> list[Path]:
    market_ai_root = Path(__file__).resolve().parents[1]
    data_engine_root = market_ai_root.parent
    repo_root = data_engine_root.parent
    return [
        market_ai_root / "state" / "creds.json",
        data_engine_root / "state" / "creds.json",
        repo_root / "state" / "creds.json",
    ]


def _load_dhan_creds(creds_path: str | Path | None = None) -> tuple[str, str]:
    if creds_path:
        candidates = [Path(creds_path)]
    else:
        candidates = _default_dhan_creds_candidates()

    # Prefer persisted UI/runtime credentials over inherited shell env so
    # long-running terminals do not keep using a stale token after the UI
    # updates state/creds.json.
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text())
        except Exception:
            continue
        client_id = (str(data.get("client_id") or "")).strip()
        token = (str(data.get("access_token") or "")).strip()
        if client_id and token:
            os.environ["DHAN_CLIENT_ID"] = client_id
            os.environ["DHAN_ACCESS_TOKEN"] = token
            return client_id, token

    client_id = (os.environ.get("DHAN_CLIENT_ID") or "").strip()
    token = (os.environ.get("DHAN_ACCESS_TOKEN") or "").strip()
    if client_id and token:
        return client_id, token
    raise DhanError("Missing DHAN credentials: update state/creds.json or export DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN")


def _make_client(creds_path: str | Path | None = None) -> SimpleDhanClient:
    client_id, token = _load_dhan_creds(creds_path)
    return SimpleDhanClient(client_id=client_id, access_token=token)


def _resolve_front_expiry(
    underlying_id: int,
    underlying_seg: str,
    *,
    creds_path: str | Path | None = None,
    now_ist: datetime | None = None,
) -> str:
    client = _make_client(creds_path)
    expiries = get_expiry_list_for_underlying(client, underlying_id, underlying_seg, use_cache=False)
    today = (now_ist or datetime.now(IST)).date()
    valid: list[date] = []
    for item in expiries:
        try:
            d = datetime.strptime(str(item), "%Y-%m-%d").date()
        except Exception:
            continue
        if d >= today:
            valid.append(d)
    if not valid:
        raise DhanError("No future NIFTY expiries returned by Dhan")
    return sorted(valid)[0].isoformat()


def _fetch_option_chain_payload(
    *,
    underlying_id: int,
    underlying_seg: str,
    expiry: str,
    creds_path: str | Path | None = None,
) -> dict[str, Any]:
    client = _make_client(creds_path)
    payload = {
        "UnderlyingScrip": int(underlying_id),
        "UnderlyingSeg": str(underlying_seg),
        "Expiry": str(expiry),
    }
    return _post_json_with_backoff(client, "/v2/optionchain", payload)


def _flatten_option_chain_payload(
    payload: dict[str, Any],
    *,
    timestamp: datetime,
    decision_time: str,
    expiry: str,
) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data") or payload
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        data = data.get("data") or data
    oc = data.get("oc") if isinstance(data, dict) else None
    if not isinstance(oc, dict):
        return []
    spot = _num(data.get("last_price") or data.get("lastPrice") or data.get("spot"))
    session_date = timestamp.strftime("%Y-%m-%d")
    stamp = timestamp.strftime("%Y-%m-%dT%H:%M:%S")

    rows: list[dict[str, object]] = []
    for strike_key, node in oc.items():
        try:
            strike = int(round(float(strike_key)))
        except Exception:
            continue
        for leg_key, option_type in (("ce", "CALL"), ("pe", "PUT")):
            leg = node.get(leg_key) or {}
            greeks = leg.get("greeks") or {}
            rows.append(
                {
                    "session_date": session_date,
                    "timestamp": stamp,
                    "decision_time": decision_time,
                    "expiry": expiry,
                    "spot": spot if spot is not None else "",
                    "strike": strike,
                    "option_type": option_type,
                    "bid": _num(leg.get("top_bid_price")),
                    "ask": _num(leg.get("top_ask_price")),
                    "ltp": _num(leg.get("last_price") or leg.get("ltp")),
                    "delta": _num(greeks.get("delta") or leg.get("delta")),
                    "delta_raw": _num(greeks.get("delta") or leg.get("delta")),
                    "delta_source": "PROVIDED" if _num(greeks.get("delta") or leg.get("delta")) is not None else "",
                    "iv": _num(leg.get("implied_volatility") or leg.get("iv")),
                    "oi": _num(leg.get("oi")),
                }
            )
    return rows


def _append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception:
            existing = pd.DataFrame(columns=df.columns)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["session_date", "timestamp", "decision_time", "expiry", "strike", "option_type"],
            keep="last",
        )
    else:
        combined = df
    combined.to_csv(path, index=False)


def _write_raw_snapshot(root: str | Path, expiry: str, timestamp: datetime, payload: dict[str, Any]) -> Path:
    raw_root = Path(root)
    folder = raw_root / timestamp.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{timestamp.strftime('%H%M%S')}_{expiry}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out


def _num(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _is_trading_day(day: date) -> bool:
    return day.weekday() < 5
