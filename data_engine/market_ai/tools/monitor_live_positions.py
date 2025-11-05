# tools/monitor_live_positions.py
"""
Simple monitor for live open positions (e.g. monthly strangle).
It polls Dhan account endpoints and logs open option positions, P&L, and snapshots option chain.
Run it in background or scheduler; extend to place hedge orders if desired.
"""
from __future__ import annotations
import time
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from market_ai.modules.data_fetch.dhan_api import DhanApiClient  # adapt to your actual client class
from market_ai.modules.data_fetch.dhan_option_chain import OptionChain  # adapt import if needed

LOG = logging.getLogger("monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--poll-interval", type=int, default=60, help="Polling interval seconds")
    p.add_argument("--out-dir", default="live_monitor", help="Directory to write snapshots")
    p.add_argument("--underlying", type=int, default=26000)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Create your Dhan client - adapt this to your code
    dh_client = DhanApiClient()  # <--- replace with your constructor
    oc = OptionChain(dh_client)

    LOG.info("Starting monitor for underlying %s", args.underlying)

    while True:
        try:
            # fetch account positions (replace with actual function names)
            positions = getattr(dh_client, "get_positions", None)
            if positions is None:
                # if your wrapper uses different call, update here
                LOG.warning("Dhan client has no get_positions() - implement this call")
                positions = []

            # write positions snapshot
            ts = datetime.now().isoformat()
            fname = outdir / f"positions_{ts}.json"
            with open(fname, "w") as fh:
                json.dump({"ts": ts, "positions": positions}, fh, default=str, indent=2)

            LOG.info("Wrote positions snapshot: %s", fname)

            # If you want to monitor a particular strategy / monthly strangle, filter positions here
            # Example: look for option legs with expiry equal to monthly expiry & strikes matching your record.
            # Also snapshot underlying option chain for same expiry:
            expiries = oc.expiry_list(args.underlying, "NSE_FNO")
            chosen_expiry = None
            if expiries:
                # pick last Tuesday style monthly expiry. For now choose the next > today with weekday==1 (Tuesday)
                from datetime import date
                today = date.today()
                for e in expiries:
                    try:
                        d = datetime.fromisoformat(e).date()
                    except Exception:
                        continue
                    if d >= today:
                        # naive pick first future expiry as "current"
                        chosen_expiry = e
                        break
            oc_resp = oc.option_chain(args.underlying, "NSE_FNO", chosen_expiry)
            snapshot_file = outdir / f"optionchain_{chosen_expiry or 'all'}_{ts}.json"
            with open(snapshot_file, "w") as fh:
                json.dump(oc_resp, fh, indent=2, default=str)
            LOG.info("Saved optionchain snapshot: %s", snapshot_file)

            # optional: compute simple P&L for option legs if positions available (user must map fields)
            # For a real monitoring/automation, add safety checks and paper-run first.

        except Exception as e:
            LOG.exception("Monitor loop exception: %s", e)

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
