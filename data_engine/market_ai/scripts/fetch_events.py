#!/usr/bin/env python3
"""
Fetch external events (holidays, macro calendar, regulatory RSS) and store them locally.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import sys

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from market_ai.modules.events import EventService  # noqa: E402
from market_ai.modules.events.collectors import (  # noqa: E402
    collect_events,
    collect_macro_forexfactory,
    collect_nse_holidays,
    collect_rbi_press_releases,
    collect_sebi_press_releases,
)


COLLECTOR_MAP = {
    "nse_holidays": collect_nse_holidays,
    "rbi": collect_rbi_press_releases,
    "sebi": collect_sebi_press_releases,
    "macro": collect_macro_forexfactory,
    "all": collect_events,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch external events and persist them locally.")
    parser.add_argument(
        "--collectors",
        default="all",
        help="Comma separated list of collectors (all,nse_holidays,rbi,sebi,macro).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("state/events"))
    parser.add_argument("--year", type=int, help="Year for NSE holiday fetch.")
    parser.add_argument("--countries", default="IN,US", help="Countries for macro collector.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = EventService(base_dir=args.output_dir)
    collector_keys = [key.strip() for key in args.collectors.split(",") if key.strip()]
    if not collector_keys:
        collector_keys = ["all"]
    total = 0
    summary: List[str] = []
    for key in collector_keys:
        func = COLLECTOR_MAP.get(key)
        if not func:
            continue
        if key == "nse_holidays":
            events = func(year=args.year)
        elif key == "macro":
            countries = [c.strip() for c in args.countries.split(",") if c.strip()]
            events = func(countries=countries)
        elif key == "all":
            events = func()
        else:
            events = func()
        inserted = service.add_events(events)
        total += inserted
        summary.append(f"{key}: {inserted}")
    now = datetime.now(timezone.utc).isoformat()
    print(f"[{now}] stored {total} events ({', '.join(summary)}) in {args.output_dir}")


if __name__ == "__main__":
    main()
