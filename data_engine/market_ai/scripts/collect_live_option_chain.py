#!/usr/bin/env python3
"""Collect live option chain snapshots and store them locally."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from market_ai.modules.data_fetch.dhan_api import make_client  # noqa: E402
from market_ai.modules.data_fetch.dhan_option_chain import OptionChain  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch live option chain snapshot and store JSON/CSV.")
    parser.add_argument("--underlying-id", type=int, default=13, help="Dhan security ID (13 = NIFTY).")
    parser.add_argument("--underlying-seg", default="IDX_I", help="Segment for option chain API.")
    parser.add_argument("--expiries", default="", help="Comma separated expiries (YYYY-MM-DD). Blank fetches front.")
    parser.add_argument("--out-dir", type=Path, default=Path("state/option_chain_history"), help="Where to store snapshots.")
    parser.add_argument("--symbols", default="NIFTY", help="Comma separated symbols tags to annotate rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = make_client()
    oc = OptionChain(client, default_seg=args.underlying_seg)
    expiries = [e.strip() for e in args.expiries.split(",") if e.strip()]
    if not expiries:
        expiries = [None]
    ts = datetime.now()
    date_tag = ts.strftime("%Y-%m-%d")
    time_tag = ts.strftime("%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    for expiry in expiries:
        chain = oc.get_option_chain(args.underlying_id, expiry_or_tag=expiry, underlying_seg=args.underlying_seg)
        if not chain:
            continue
        dir_path = args.out_dir / date_tag
        dir_path.mkdir(parents=True, exist_ok=True)
        name_suffix = expiry or "front"
        json_path = dir_path / f"{time_tag}_{name_suffix}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(chain, handle, indent=2)
        csv_path = dir_path / f"{time_tag}_{name_suffix}.csv"
        rows = []
        for strike, legs in chain.items():
            strike_val = float(strike)
            for option_type in ("ce", "pe"):
                leg = legs.get(option_type)
                if not leg:
                    continue
                rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "symbol": symbols[0] if symbols else "",
                        "expiry": expiry or "",
                        "option_type": "CALL" if option_type == "ce" else "PUT",
                        "strike": strike_val,
                        "last_price": leg.get("last_price"),
                        "open": leg.get("open"),
                        "high": leg.get("high"),
                        "low": leg.get("low"),
                        "close": leg.get("close"),
                        "volume": leg.get("volume"),
                        "iv": leg.get("iv"),
                        "oi": leg.get("oi"),
                    }
                )
        try:
            import pandas as pd
            import numpy as np  # noqa: F401
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        except Exception:
            pass
        print(f"[{ts.isoformat()}] stored option chain snapshot -> {json_path}")


if __name__ == "__main__":
    os.environ.setdefault("STREAMLIT_ENV", "CLI")
    main()
