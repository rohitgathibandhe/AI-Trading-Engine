# tools/snapshot_option_chains.py
"""
Snapshot option chains (one JSON per day per underlying/expiry) so you can compute historical IV / OI.

Usage:
  python3 tools/snapshot_option_chains.py --underlying 26000 --out snapshots/nifty
Schedule daily (cron / systemd / launchd) before market close to collect OI/IV history.

Requires your dh wrapper imports to work. Adjust imports at top if needed.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
from datetime import date
from pathlib import Path
from market_ai.modules.data_fetch.dhan_option_chain import OptionChain  # adjust if path different
from market_ai.modules.data_fetch.dhan_api import DhanContext  # replace with your actual context factory

LOG = logging.getLogger("snapshotter")
logging.basicConfig(level=logging.INFO)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--underlying", type=int, required=True, help="Dhan underlying id (e.g. 26000 for NIFTY)")
    p.add_argument("--out", default="snapshots", help="Output base directory")
    p.add_argument("--expiry", default=None, help="Optional expiry (YYYY-MM-DD). If omitted, snapshots all expiries")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # create dh context (adapt to your code)
    # Example: if you have a function create_dhan_context(...) do that here.
    # Below is placeholder — replace with your code to create Dhan HTTP/context object:
    try:
        dh_ctx = DhanContext()  # <<< replace this if your constructor differs
    except Exception:
        # If your module is named differently, instruct user to edit here
        raise RuntimeError("Please adapt snapshot_option_chains.py to create your DhanContext (see file top).")

    oc = OptionChain(dh_ctx)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    expiries = [args.expiry] if args.expiry else oc.expiry_list(args.underlying, "NSE_FNO") or [None]

    for e in expiries:
        try:
            resp = oc.option_chain(args.underlying, "NSE_FNO", e)
            fname = outdir / f"{args.underlying}_{e or 'all'}_{today}.json"
            with open(fname, "w") as fh:
                json.dump(resp, fh, indent=2, default=str)
            LOG.info("Saved snapshot to %s", fname)
        except Exception as ex:
            LOG.exception("Failed to snapshot expiry=%s: %s", e, ex)


if __name__ == "__main__":
    main()
