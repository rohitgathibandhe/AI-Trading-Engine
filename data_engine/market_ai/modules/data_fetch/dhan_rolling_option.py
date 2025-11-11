# -*- coding: utf-8 -*-
"""
Client for Dhan's /v2/charts/rollingoption endpoint.

This module fetches historical rolling option data and persists it as
parquet files partitioned by trade date (state/rolling_option/YYYY-MM-DD).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests

from .dhan_api import SimpleDhanClient, _raise_for_status

LOG = logging.getLogger(__name__)

ROLLING_OPTION_PATH = "/v2/charts/rollingoption"
DEFAULT_REQUIRED_DATA: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "oi",
    "iv",
    "spot",
    "strike",
    "timestamp",
)


@dataclass
class RollingOptionConfig:
    underlying: str
    segment: str = "NSE_FNO"
    security_id: Optional[int] = None
    instrument: str = "OPTIDX"
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    expiry: Optional[str] = None
    interval: str = "1"
    expiry_flag: str = "MONTH"
    expiry_code: Optional[int] = None
    strike_selectors: Sequence[str] = field(default_factory=lambda: ("ATM",))
    option_types: Sequence[str] = field(default_factory=lambda: ("CALL", "PUT"))
    required_data: Sequence[str] = field(default_factory=lambda: DEFAULT_REQUIRED_DATA)
    limit_per_page: int = 500  # retained for backward compatibility (unused)


class RollingOptionIngestor:
    def __init__(self, client: Optional[SimpleDhanClient] = None, out_dir: Optional[Path] = None):
        self.client = client or SimpleDhanClient()
        self.out_dir = Path(out_dir or Path(__file__).resolve().parents[2] / "state" / "rolling_option")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def fetch_range(self, cfg: RollingOptionConfig) -> List[Path]:
        if not cfg.security_id:
            raise ValueError("RollingOptionConfig requires security_id")
        if not cfg.start or not cfg.end:
            raise ValueError("RollingOptionConfig requires start and end datetimes")

        expiry_code = cfg.expiry_code or self._infer_expiry_code(cfg.start.date(), cfg.expiry)
        base_payload = {
            "exchangeSegment": cfg.segment,
            "instrument": cfg.instrument,
            "securityId": int(cfg.security_id),
            "interval": cfg.interval,
            "expiryFlag": cfg.expiry_flag,
            "expiryCode": int(expiry_code),
            "fromDate": cfg.start.strftime("%Y-%m-%d"),
            "toDate": cfg.end.strftime("%Y-%m-%d"),
        }

        records: List[Dict[str, Any]] = []
        strikes = cfg.strike_selectors or ("ATM",)
        option_types = cfg.option_types or ("CALL", "PUT")
        required = list(cfg.required_data or DEFAULT_REQUIRED_DATA)

        for strike in strikes:
            for option_type in option_types:
                payload = {
                    **base_payload,
                    "strike": strike,
                    "drvOptionType": option_type,
                    "requiredData": required,
                }
                LOG.info(
                    "Fetching rollingoption strike=%s option=%s range=%s→%s",
                    strike,
                    option_type,
                    base_payload["fromDate"],
                    base_payload["toDate"],
                )
                resp = requests.post(
                    self.client._url(ROLLING_OPTION_PATH),
                    headers=self.client.headers,
                    json=payload,
                    timeout=self.client.timeout,
                )
                _raise_for_status(resp)
                data = resp.json()
                records.extend(
                    self._normalize_payload(
                        data,
                        option_type=option_type,
                        selector=strike,
                        expiry_hint=cfg.expiry,
                    )
                )

        return self._write_records(records)

    @staticmethod
    def _infer_expiry_code(start_day: date, expiry_str: Optional[str]) -> int:
        if not expiry_str:
            return 1
        try:
            expiry_day = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        except Exception:
            return 1
        diff = (expiry_day.year - start_day.year) * 12 + (expiry_day.month - start_day.month)
        if diff < 0:
            return 1
        return min(3, diff + 1)

    @staticmethod
    def _normalize_payload(
        payload: Dict[str, Any],
        *,
        option_type: str,
        selector: str,
        expiry_hint: Optional[str],
    ) -> List[Dict[str, Any]]:
        block_key = "ce" if option_type.upper() == "CALL" else "pe"
        node = (payload or {}).get("data") or {}
        block = node.get(block_key) or {}
        timestamps = block.get("timestamp") or []
        if not timestamps:
            return []

        def _pick(seq: Iterable[Any], idx: int) -> Any:
            if isinstance(seq, list):
                return seq[idx] if idx < len(seq) else None
            return None

        rows: List[Dict[str, Any]] = []
        for idx, raw_ts in enumerate(timestamps):
            try:
                stamp = int(raw_ts)
                dt = datetime.fromtimestamp(stamp)
            except Exception:
                continue
            trade_date = dt.date().isoformat()
            trade_time = dt.time().isoformat(timespec="seconds")
            strike_price = _pick(block.get("strike") or [], idx)
            expiry_date = _pick(block.get("expiryDate") or [], idx) or expiry_hint
            close_px = _pick(block.get("close") or [], idx)
            open_px = _pick(block.get("open") or [], idx)
            record = {
                "tradeDate": trade_date,
                "tradeTime": trade_time,
                "expiryDate": expiry_date,
                "strikePrice": strike_price,
                "optionType": "CE" if option_type.upper() == "CALL" else "PE",
                "ltp": close_px if close_px is not None else open_px,
                "open": open_px,
                "high": _pick(block.get("high") or [], idx),
                "low": _pick(block.get("low") or [], idx),
                "close": close_px,
                "volume": _pick(block.get("volume") or [], idx),
                "oi": _pick(block.get("oi") or [], idx),
                "iv": _pick(block.get("iv") or [], idx),
                "spot": _pick(block.get("spot") or [], idx),
                "delta": _pick(block.get("delta") or [], idx),
                "gamma": _pick(block.get("gamma") or [], idx),
                "theta": _pick(block.get("theta") or [], idx),
                "vega": _pick(block.get("vega") or [], idx),
                "selector": selector,
                "raw_option_type": option_type,
            }
            rows.append(record)
        return rows

    def _write_records(self, records: Iterable[Dict[str, Any]]) -> List[Path]:
        records = list(records)
        if not records:
            return []
        df = pd.DataFrame(records)
        written: List[Path] = []
        for trade_date, chunk in df.groupby("tradeDate", dropna=True):
            folder = self.out_dir / str(trade_date)
            folder.mkdir(parents=True, exist_ok=True)
            json_path = folder / "rolling_option_raw.jsonl"
            with json_path.open("a") as fh:
                for row in chunk.to_dict(orient="records"):
                    fh.write(json.dumps(row) + "\n")
            try:
                merged = pd.read_json(json_path, lines=True)
            except ValueError:
                merged = chunk.copy()
            parquet_path = folder / "rolling_option.parquet"
            merged.to_parquet(parquet_path, index=False)
            written.append(parquet_path)
        return written
