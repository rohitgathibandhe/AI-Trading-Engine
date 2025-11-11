"""Quality and coverage summaries for rolling option datasets."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


_FIELDS_OF_INTEREST = ("iv", "oi", "delta", "gamma", "theta", "vega", "spot", "ltp")


@dataclass
class CoverageSummary:
    root_dir: Path
    lookback_days: int = 365

    def window(self) -> tuple[date, date]:
        end = date.today()
        start = end - timedelta(days=self.lookback_days - 1)
        return start, end

    def summarise(self) -> Dict[str, Any]:
        start_day, end_day = self.window()
        expected_days = []
        avail_days = {}
        total_rows = 0
        field_present = Counter()
        field_totals = Counter()
        selector_counts = Counter()
        option_counts = Counter()
        per_day_counts = {}
        errors: List[str] = []

        dt = start_day
        while dt <= end_day:
            if dt.weekday() < 5:  # weekdays only
                expected_days.append(dt)
                day_dir = self.root_dir / dt.isoformat()
                parquet_path = day_dir / "rolling_option.parquet"
                if parquet_path.exists():
                    stats = self._summarise_day(parquet_path)
                    if stats.get("error"):
                        errors.append(f"{dt}: {stats['error']}")
                    else:
                        rows = stats.get("rows", 0)
                        total_rows += rows
                        per_day_counts[dt.isoformat()] = rows
                        option_counts.update(stats.get("option_counts", {}))
                        selector_counts.update(stats.get("selector_counts", {}))
                        for field, count in stats.get("field_populated", {}).items():
                            field_present[field] += count
                            field_totals[field] += rows
                        avail_days[dt] = parquet_path.stat().st_mtime
            dt += timedelta(days=1)

        missing_days = [d.isoformat() for d in expected_days if d not in avail_days]
        latest_day = max(avail_days.keys(), default=None)
        latest_updated = None
        if latest_day is not None:
            latest_updated = datetime.fromtimestamp(avail_days[latest_day]).isoformat()

        coverage_ratio = {
            field: (field_present[field] / field_totals[field]) if field_totals[field] else 0.0
            for field in _FIELDS_OF_INTEREST
        }

        summary: Dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
            "root_dir": str(self.root_dir),
            "lookback_days": self.lookback_days,
            "window_start": start_day.isoformat(),
            "window_end": end_day.isoformat(),
            "weekdays_expected": len(expected_days),
            "days_with_data": len(avail_days),
            "days_missing": len(missing_days),
            "missing_dates": missing_days[:50],  # surface first 50 for UI
            "latest_day": latest_day.isoformat() if latest_day else None,
            "latest_file_updated": latest_updated,
            "total_rows": total_rows,
            "average_rows_per_day": (total_rows / len(avail_days)) if avail_days else 0.0,
            "coverage_ratio": coverage_ratio,
            "selector_counts": selector_counts.most_common(15),
            "option_counts": option_counts,
            "per_day_counts": _tail_dict(per_day_counts, limit=120),
            "errors": errors[:20],
        }
        return summary

    def _summarise_day(self, parquet_path: Path) -> Dict[str, Any]:
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            return {"rows": 0, "error": str(exc)}

        rows = len(df)
        field_populated = {}
        for field in _FIELDS_OF_INTEREST:
            if field in df.columns:
                field_populated[field] = int(df[field].notna().sum())
            else:
                field_populated[field] = 0
        selector_counts = Counter()
        if "selector" in df.columns:
            selector_counts.update(df["selector"].astype(str).fillna("NA"))
        option_counts = Counter()
        if "optionType" in df.columns:
            option_counts.update(df["optionType"].astype(str).fillna("NA"))

        return {
            "rows": rows,
            "field_populated": field_populated,
            "selector_counts": selector_counts,
            "option_counts": option_counts,
        }


def _tail_dict(data: Dict[str, int], limit: int = 120) -> Dict[str, int]:
    if not data:
        return {}
    ordered = sorted(data.items(), key=lambda x: x[0], reverse=True)
    trimmed = ordered[:limit]
    return dict(reversed(trimmed))


def write_summary(
    root_dir: Path,
    summary_path: Optional[Path] = None,
    lookback_days: int = 365,
) -> Dict[str, Any]:
    summary = CoverageSummary(root_dir=root_dir, lookback_days=lookback_days).summarise()
    if summary_path is None:
        summary_path = root_dir.parent / "rolling_option_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
