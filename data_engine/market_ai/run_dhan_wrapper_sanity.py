from datetime import datetime
import os
import time

from data_engine.market_ai.dhan_wrapper import DhanWrapper

def main():
    # Basic env check
    cid = os.getenv("DHAN_CLIENT_ID")
    tok = os.getenv("DHAN_ACCESS_TOKEN")
    if not cid or not tok:
        print("❌ Missing DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN")
        return

    dw = DhanWrapper()

    # 1) Funds
    print("\n=== FUNDS ===")
    funds = dw.get_funds()
    print(funds)

    # 2) Expiry list for NIFTY (IDX_I, 13)
    print("\n=== EXPIRY LIST (NIFTY) ===")
    exp = dw.get_expiry_list(underlying_security_id=13, underlying_seg="IDX_I")
    print(exp)
    # optionally pick the nearest expiry from response
    expiry = None
    try:
        # Adapt to your API shape if needed
        dates = exp.get("data") or exp  # sometimes exp is already the data dict
        if isinstance(dates, dict):
            # e.g. {"Expiry":["2025-10-30", ...]}
            for k in ("Expiry", "expiries", "expiry", "data"):
                if k in dates:
                    v = dates[k]
                    expiry = v[0] if isinstance(v, list) and v else None
                    break
    except Exception:
        pass
    print("Picked Expiry:", expiry)

    # 3) One-shot LTP for NIFTY spot
    print("\n=== LTP ONCE (NIFTY spot: IDX_I/13) ===")
    ltp = dw.get_ltp_once(exchange_seg="IDX_I", security_id=13)
    print("NIFTY LTP:", ltp)

    # 4) Stream LTP for 10 seconds to prove updates work
    print("\n=== START LTP STREAM (10s) ===")
    ticks = []

    def on_tick(sym, last, ts):
        print(f"[{ts}] {sym} {last}")
        ticks.append((sym, last, ts))

    dw.start_ltp_stream(symbol="NIFTY", exchange_seg="IDX_I", security_id=13,
                        interval_sec=2.0, on_update=on_tick, poll_when_closed=True)

    time.sleep(10)
    dw.stop_ltp_stream()

    # 5) Final cached value
    print("\n=== CACHED LTP ===")
    last, ts = dw.get_cached_ltp("NIFTY")
    print("Cached:", last, ts)

if __name__ == "__main__":
    main()
