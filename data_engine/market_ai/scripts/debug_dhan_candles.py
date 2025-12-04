#!/usr/bin/env python3
"""
Debug helper: call Dhan /charts/historical and /charts/intraday and show raw payload/response.
Adjust securityId / exchangeSegment / instrument as needed.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

# Ensure repo root on sys.path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_engine.market_ai.dhan_wrapper import DhanWrapper
from data_engine.market_ai.signals.monthly_signals import _candles_response_to_df, _candles_list_to_df


# Defaults (NIFTY) – IDX_I segment works for historical/intraday
SECURITY_ID = "13"
EXCHANGE_SEGMENT = "IDX_I"
INSTRUMENT = "INDEX"


def main() -> None:
    dw = DhanWrapper()
    client = dw.http  # raw DhanHTTP (has .post)

    today = dt.date.today()
    from_date = today.replace(day=1)
    to_date = today + dt.timedelta(days=1)

    out_dir = Path("data_engine/market_ai/state/debug_candles")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Testing segment={EXCHANGE_SEGMENT} instrument={INSTRUMENT} ===")
    try:
        daily_raw = dw.get_daily_candles(
            int(SECURITY_ID),
            EXCHANGE_SEGMENT,
            INSTRUMENT,
            from_date=from_date.isoformat(),
            to_date=to_date.isoformat(),
        )
    except Exception as exc:
        print("Daily fetch failed:", exc)
        daily_raw = None
    print("Daily raw type:", type(daily_raw), "sample:", str(daily_raw)[:200])
    daily_df = _candles_list_to_df(daily_raw) if daily_raw is not None else None
    print(f"Daily rows: {0 if daily_df is None else len(daily_df)}")

    try:
        intra_raw = dw.get_intraday_candles(
            int(SECURITY_ID),
            EXCHANGE_SEGMENT,
            INSTRUMENT,
            interval=15,
            from_date=(today - dt.timedelta(days=2)).isoformat(),
            to_date=to_date.isoformat(),
        )
    except Exception as exc:
        print("Intraday fetch failed:", exc)
        intra_raw = None
    print("Intraday raw type:", type(intra_raw), "sample:", str(intra_raw)[:200])
    intra_df = _candles_list_to_df(intra_raw) if intra_raw is not None else None
    print(f"Intraday rows: {0 if intra_df is None else len(intra_df)}")

    Path(out_dir / f"historical_{EXCHANGE_SEGMENT}_{INSTRUMENT}.json").write_text(json.dumps(daily_raw, indent=2))
    Path(out_dir / f"intraday_{EXCHANGE_SEGMENT}_{INSTRUMENT}.json").write_text(json.dumps(intra_raw, indent=2))
    if daily_df is not None:
        daily_df.to_csv(out_dir / f"historical_{EXCHANGE_SEGMENT}_{INSTRUMENT}.csv")
    if intra_df is not None:
        intra_df.to_csv(out_dir / f"intraday_{EXCHANGE_SEGMENT}_{INSTRUMENT}.csv")

    print(f"\nSaved responses under {out_dir}")


if __name__ == "__main__":
    main()
