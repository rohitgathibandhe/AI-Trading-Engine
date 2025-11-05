from __future__ import annotations

import os
import time
import argparse
import logging
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo  # built-in on Python 3.9+

from market_ai.modules.data_fetch.live_option_chain import LiveOptionChainMarket, _k_key
from market_ai.modules.paper_broker import PaperBroker

LOG = logging.getLogger("paper_trader")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

IST = ZoneInfo("Asia/Kolkata")

def _pick_delta_strike(chain: Dict[str, Any], target: float, is_call: bool) -> Optional[Tuple[float, Dict[str, Any]]]:
    best = None; best_gap = 1e9
    for k_str, row in chain.items():
        leg = row["ce"] if is_call else row["pe"]
        d = leg.get("greeks", {}).get("delta"); ltp = leg.get("last_price")
        if d is None or ltp is None or ltp <= 0: continue
        gap = abs(abs(d) - target)
        if gap < best_gap:
            best_gap = gap; best = (float(k_str), leg)
    return best

def _market_open_now() -> bool:
    now = datetime.now(IST).time()
    return dtime(9, 15) <= now <= dtime(15, 30)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying-scrip", type=int, default=13, help="DHAN UnderlyingScrip (security id) for NIFTY index")
    ap.add_argument("--expiry", required=True, help="Expiry yyyy-mm-dd (e.g., next monthly)")
    ap.add_argument("--lots", type=int, default=1)
    ap.add_argument("--poll-sec", type=float, default=30.0)
    ap.add_argument("--profit-target", type=float, default=0.50)
    ap.add_argument("--stop-loss", type=float, default=1.50)
    ap.add_argument("--delta-target", type=float, default=0.15)
    ap.add_argument("--lot-size", type=int, default=50)
    ap.add_argument("--gate-market-hours", action="store_true",
                    help="If set, pause MTM when NSE is closed (09:15–15:30 IST only).")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug: logging.getLogger().setLevel(logging.DEBUG)

    cid = os.environ.get("DHAN_CLIENT_ID"); tok = os.environ.get("DHAN_ACCESS_TOKEN")
    LOG.info("=== PAPER TRADER START ===  Using DHAN_CLIENT_ID=%s… token=%s",
             (cid or "MISSING")[:6], "SET" if tok else "MISSING")

    market = LiveOptionChainMarket(client_id=cid, access_token=tok, underlying_seg="IDX_I", iv_fallback=0.20)
    broker = PaperBroker(lot_size=args.lot_size)

    LOG.info("Fetching live chain: UnderlyingScrip=%d, requested Expiry=%s", args.underlying_scrip, args.expiry)
    chain = market.get_live_chain(args.underlying_scrip, args.expiry)
    if not chain:
        LOG.error("EMPTY chain. Check UnderlyingScrip / Expiry on /v2/optionchain docs.")
        return 2

    ce_pick = _pick_delta_strike(chain, args.delta_target, is_call=True)
    pe_pick = _pick_delta_strike(chain, args.delta_target, is_call=False)
    if not ce_pick or not pe_pick:
        LOG.error("Could not find usable strikes near Δ=%.2f. Try a nearer expiry or verify greeks/ltp fields.", args.delta_target)
        return 3

    ceK, celeg = ce_pick; peK, peleg = pe_pick
    ceK_key = _k_key(ceK); peK_key = _k_key(peK)
    ce_px = float(celeg["last_price"]); pe_px = float(peleg["last_price"])
    LOG.info("ENTRY: SELL CE %s @ %.2f (Δ=%.3f) | SELL PE %s @ %.2f (Δ=%.3f)",
             ceK_key, ce_px, celeg["greeks"]["delta"] or float("nan"),
             peK_key, pe_px, peleg["greeks"]["delta"] or float("nan"))

    broker.place("SELL", "CE", float(ceK_key), args.lots, ce_px, args.expiry)
    broker.place("SELL", "PE", float(peK_key), args.lots, pe_px, args.expiry)
    entry_credit = (ce_px + pe_px) * args.lot_size * args.lots
    LOG.info("Net credit (paper): %.2f", entry_credit)

    # last known LTPs (local cache for stale ticks)
    last_ce_ltp = ce_px
    last_pe_ltp = pe_px

    credit_target = entry_credit * (1.0 - args.profit_target)
    credit_stop   = entry_credit * (1.0 + args.stop_loss)

    try:
        while True:
            # use the correct argparse attribute name:
            if args.gate_market_hours and not _market_open_now():
                now = datetime.now(IST).strftime("%H:%M:%S")
                LOG.info("Market closed (%s IST). Sleeping %ds; use --gate-market-hours off to compute with stale LTP.", now, int(args.poll_sec))
                time.sleep(args.poll_sec)
                continue

            chain = market.get_live_chain(args.underlying_scrip, args.expiry)
            ce_leg = chain.get(ceK_key, {}).get("ce", {})
            pe_leg = chain.get(peK_key, {}).get("pe", {})
            ce_ltp = ce_leg.get("last_price"); pe_ltp = pe_leg.get("last_price")

            # fall back to last known if this poll is missing
            if ce_ltp is None:
                ce_ltp = last_ce_ltp
            else:
                last_ce_ltp = ce_ltp
            if pe_ltp is None:
                pe_ltp = last_pe_ltp
            else:
                last_pe_ltp = pe_ltp

            if ce_ltp is None or pe_ltp is None:
                LOG.warning("Still no price for one leg (even after fallback). Waiting…")
                time.sleep(args.poll_sec); continue

            payback_now = (ce_ltp + pe_ltp) * args.lot_size * args.lots
            mtm = entry_credit - payback_now
            LOG.info("MTM: %.2f | payback: %.2f | targets: <= %.2f (PT) / >= %.2f (SL) | LTPs CE=%.2f PE=%.2f",
                     mtm, payback_now, credit_target, credit_stop, ce_ltp, pe_ltp)

            if payback_now <= credit_target:
                LOG.info("🎯 Profit target reached (paper)."); break
            if payback_now >= credit_stop:
                LOG.info("🛑 Stop-loss reached (paper)."); break

            time.sleep(args.poll_sec)
    except KeyboardInterrupt:
        LOG.info("Interrupted by user.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
