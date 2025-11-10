# -*- coding: utf-8 -*-
"""
Client for Dhan's /v2/charts/rollingoption endpoint.

This module fetches historical rolling option data and persists it as
parquet files partitioned by trade date (state/rolling_option/YYYY-MM-DD).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

from .dhan_api import SimpleDhanClient, _raise_for_status

LOG = logging.getLogger(__name__)

ROLLING_OPTION_PATH = "/v2/charts/rollingoption"


@dataclass
class RollingOptionConfig:
    underlying: str
    segment: str = "NSE_FNO"
    security_id: Optional[int] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    expiry: Optional[str] = None
    limit_per_page: int = 500
    interval: str = "1"


class RollingOptionIngestor:
    def __init__(self, client: Optional[SimpleDhanClient] = None, out_dir: Optional[Path] = None):
        self.client = client or SimpleDhanClient()
        self.out_dir = Path(out_dir or Path(__file__).resolve().parents[2] / "state" / "rolling_option")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def fetch_range(self, cfg: RollingOptionConfig) -> List[Path]:
        if not cfg.security_id:
            raise ValueError("RollingOptionConfig requires security_id")

        payload = {
            "underlyingScrip": cfg.underlying,
            "underlyingSeg": cfg.segment,
            "exchangeSegment": cfg.segment,
            "instrument": cfg.underlying,
            "securityId": int(cfg.security_id),
            "pageSize": cfg.limit_per_page,
            "interval": cfg.interval,
        }
        if cfg.expiry:
            payload["expiry"] = cfg.expiry
        if cfg.start:
            payload["startDate"] = cfg.start.strftime("%Y-%m-%d")
        if cfg.end:
            payload["endDate"] = cfg.end.strftime("%Y-%m-%d")

        page = 1
        written: List[Path] = []
        while True:
            payload["pageNo"] = page
            LOG.info("Fetching rollingoption page=%s params=%s", page, {k: payload.get(k) for k in ("startDate", "endDate", "expiry")})
            resp = requests.post(
                self.client._url(ROLLING_OPTION_PATH),
                headers=self.client.headers,
                json=payload,
                timeout=self.client.timeout,
            )
            _raise_for_status(resp)
            data = resp.json()
            body = data.get("data") or []
            if not body:
                break
            written.extend(self._write_page(body))
            if not data.get("hasMore"):
                break
            page += 1
        return written

    def _write_page(self, rows: Iterable[Dict[str, Any]]) -> List[Path]:
        rows = list(rows)
        paths: List[Path] = []
        for row in rows:
            try:
                trade_date = datetime.fromisoformat(row.get("tradeDate")).date()
            except Exception:
                continue
            folder = self.out_dir / trade_date.isoformat()
            folder.mkdir(parents=True, exist_ok=True)
            json_path = folder / "rolling_option_raw.jsonl"
            with json_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

        # convert to parquet per date
        day_dirs = set()
        for r in rows:
            trade_date = r.get("tradeDate")
            if not trade_date:
                continue
            try:
                day_dirs.add(self.out_dir / datetime.fromisoformat(trade_date).date().isoformat())
            except Exception:
                continue

        for day_dir in day_dirs:
            raw_path = day_dir / "rolling_option_raw.jsonl"
            if not raw_path.exists():
                continue
            df = pd.read_json(raw_path, lines=True)
            parquet_path = day_dir / "rolling_option.parquet"
            df.to_parquet(parquet_path, index=False)
            paths.append(parquet_path)
        return paths
