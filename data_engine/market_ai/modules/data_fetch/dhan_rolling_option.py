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
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests

from .dhan_api import SimpleDhanClient, _raise_for_status
from .dhan_api import get_expiry_list_for_underlying
import time

LOG = logging.getLogger(__name__)

ROLLING_OPTION_PATH = "/v2/charts/rollingoption"
MAX_ROLLING_WINDOW_DAYS = 30
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
    "expiryDate",
    "delta",
    "gamma",
    "theta",
    "vega",
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
    # Default to the maximum supported range for index options: ATM±10.
    strike_selectors: Sequence[str] = field(
        default_factory=lambda: tuple(
            ["ATM"]
            + [f"ATM+{k}" for k in range(1, 11)]
            + [f"ATM-{k}" for k in range(1, 11)]
        )
    )
    option_types: Sequence[str] = field(default_factory=lambda: ("CALL", "PUT"))
    required_data: Sequence[str] = field(default_factory=lambda: DEFAULT_REQUIRED_DATA)
    limit_per_page: int = 500  # retained for backward compatibility (unused)


class RollingOptionIngestor:
    def __init__(self, client: Optional[SimpleDhanClient] = None, out_dir: Optional[Path] = None):
        self.client = client or SimpleDhanClient()
        self.out_dir = Path(out_dir or Path(__file__).resolve().parents[2] / "state" / "rolling_option")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Throttle between requests to avoid DH-904 rate limits
        try:
            self.throttle_s = float(os.environ.get("DHAN_ROLLING_THROTTLE", "6.0"))
        except Exception:
            self.throttle_s = 6.0

    def fetch_range(self, cfg: RollingOptionConfig) -> List[Path]:
        if not cfg.security_id:
            raise ValueError("RollingOptionConfig requires security_id")
        if not cfg.start or not cfg.end:
            raise ValueError("RollingOptionConfig requires start and end datetimes")

        expiry_code = cfg.expiry_code or self._infer_expiry_code(cfg.start.date(), cfg.expiry)
        base_start = cfg.start.date()
        base_end = cfg.end.date()
        base_payload = {
            "exchangeSegment": cfg.segment,
            "instrument": cfg.instrument,
            "securityId": int(cfg.security_id),
            "interval": cfg.interval,
            "expiryFlag": cfg.expiry_flag,
            "expiryCode": int(expiry_code),
        }

        records: List[Dict[str, Any]] = []
        strikes = cfg.strike_selectors or ("ATM",)
        # Enforce Dhan limits: index options ATM±10, others ATM±3
        def _within_limit(selector: str) -> bool:
            if selector.upper() == "ATM":
                return True
            if not selector.upper().startswith("ATM"):
                return False
            try:
                off = selector.upper().replace("ATM", "")
                sign = 1 if "+" in off else -1
                mag = int(off.replace("+", "").replace("-", "")) * sign
            except Exception:
                return False
            limit = 10 if cfg.instrument.upper() == "OPTIDX" else 3
            return abs(mag) <= limit
        strikes = tuple(s for s in strikes if _within_limit(s))
        option_types = cfg.option_types or ("CALL", "PUT")
        required = list(cfg.required_data or DEFAULT_REQUIRED_DATA)

        # Pre-fetch expiry list so we can stamp expiry when the payload omits it.
        try:
            expiry_list = get_expiry_list_for_underlying(self.client, int(cfg.security_id), cfg.segment, use_cache=True)
            expiry_list = sorted(expiry_list)
        except Exception as exc:
            LOG.warning("Could not fetch expiry list; expiry stamping may be missing: %s", exc)
            expiry_list = []

        chunk_start = base_start
        while chunk_start <= base_end:
            chunk_end = min(chunk_start + timedelta(days=MAX_ROLLING_WINDOW_DAYS - 1), base_end)
            chunk_payload = {
                **base_payload,
                "fromDate": chunk_start.isoformat(),
                "toDate": chunk_end.isoformat(),
            }
            for strike in strikes:
                for option_type in option_types:
                    payload = {
                        **chunk_payload,
                        "strike": strike,
                        "drvOptionType": option_type,
                        "requiredData": required,
                    }
                    LOG.info(
                        "Fetching rollingoption strike=%s option=%s range=%s→%s",
                        strike,
                        option_type,
                        payload["fromDate"],
                        payload["toDate"],
                    )
                    resp = requests.post(
                    self.client._url(ROLLING_OPTION_PATH),
                    headers=self.client.headers,
                    json=payload,
                    timeout=self.client.timeout,
                )
                    if resp.status_code == 429:
                        wait = min(max(self.throttle_s, 2.0 * attempt), 12.0) if 'attempt' in locals() else self.throttle_s
                        LOG.warning("rollingoption 429 rate limit; sleeping %.1fs and retrying once", wait)
                        time.sleep(wait)
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
                            expiry_hint=self._choose_expiry(expiry_list, dt_hint=chunk_start),
                        )
                    )
                    # Backoff between requests to respect rate limits
                    time.sleep(self.throttle_s)
            chunk_start = chunk_end + timedelta(days=1)

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

    @staticmethod
    def _choose_expiry(expiries: Sequence[str], dt_hint: date) -> Optional[str]:
        """
        Pick the nearest expiry on/after the trade date when API omits expiryDate.
        """
        if not expiries:
            return None
        for exp in expiries:
            try:
                d = datetime.strptime(exp, "%Y-%m-%d").date()
            except Exception:
                continue
            if d >= dt_hint:
                return d.isoformat()
        return expiries[-1] if expiries else None

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
            subset_cols = [c for c in ("tradeDate", "tradeTime", "optionType", "selector", "strikePrice") if c in merged.columns]
            if subset_cols:
                merged = merged.drop_duplicates(subset=subset_cols)
            parquet_path = folder / "rolling_option.parquet"
            merged.to_parquet(parquet_path, index=False)
            written.append(parquet_path)
        return written
