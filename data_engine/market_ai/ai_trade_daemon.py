from __future__ import annotations

import os, time, csv, logging, argparse, inspect
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from market_ai.modules.data_fetch.live_option_chain import LiveOptionChainMarket
from market_ai.modules.broker_factory import get_broker
from market_ai.modules.risk_manager import RiskLimits, RiskManager
from market_ai.modules.strategies.managed_strangle import (
    enter_strangle, compute_payback, roll_up_call, roll_down_put, roll_both_in, add_hedge, Legs
)
from market_ai.agents.adaptive_agent import AdaptiveAgent, Context
from market_ai.modules.kill_switch import KillSwitch
from market_ai.modules.notifier import Notifier

IST = ZoneInfo("Asia/Kolkata")
LOG = logging.getLogger("trade_daemon")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

def market_open() -> bool:
    now = datetime.now(IST).time()
    return dtime(9,15) <= now <= dtime(15,30)

def _build_ctx(**kwargs) -> Context:
    """Only pass kwargs accepted by Context (defensive if stale version is loaded)."""
    sig = inspect.signature(Context)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    missing = [k for k in ("since_entry_sec","cycles_since_entry","min_hold_sec","min_hold_cycles") if k not in sig.parameters]
    if missing:
        LOG.warning("Context missing warm-up fields %s; warm-up guard limited this run.", missing)
    return Context(**filtered)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper","live"], default=os.environ.get("TRADE_MODE","paper"))
    ap.add_argument("--underlying-scrip", type=int, default=int(os.environ.get("UNDERLYING_SCRIP","13")))
    ap.add_argument("--expiry", default=os.environ.get("EXPIRY","2025-10-30"))
    ap.add_argument("--lot-size", type=int, default=int(os.environ.get("LOT_SIZE","75")))
    ap.add_argument("--lots", type=int, default=int(os.environ.get("LOTS","1")))
    ap.add_argument("--delta-target", type=float, default=float(os.environ.get("DELTA_TARGET","0.15")))
    ap.add_argument("--poll-sec", type=float, default=float(os.environ.get("POLL_SEC","30")))
    ap.add_argument("--breach-buffer", type=float, default=float(os.environ.get("BREACH_BUFFER","100")))
    # risk
    ap.add_argument("--max-daily-loss", type=float, default=float(os.environ.get("MAX_DAILY_LOSS","10000")))
    ap.add_argument("--per-leg-sl-mult", type=float, default=float(os.environ.get("PER_LEG_SL_MULT","2.0")))
    ap.add_argument("--max-rolls", type=int, default=int(os.environ.get("MAX_ROLLS","3")))
    ap.add_argument("--notify", default=os.environ.get("NOTIFY","console"))
    # warm-up
    ap.add_argument("--min-hold-sec", type=int, default=int(os.environ.get("MIN_HOLD_SEC","300")))
    ap.add_argument("--min-hold-cycles", type=int, default=int(os.environ.get("MIN_HOLD_CYCLES","3")))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cid = os.environ.get("DHAN_CLIENT_ID")
    tok = os.environ.get("DHAN_ACCESS_TOKEN")

    LOG.info("=== TRADE DAEMON START === mode=%s", args.mode)
    market = LiveOptionChainMarket(client_id=cid, access_token=tok, underlying_seg="IDX_I")
    broker = get_broker(args.mode, args.lot_size, cid, tok)
    agent  = AdaptiveAgent(seed=77); agent.load()
    limits = RiskLimits(max_daily_loss=args.max_daily_loss, per_leg_sl_mult=args.per_leg_sl_mult, max_rolls_per_day=args.max_rolls)
    risk   = RiskManager(limits)
    kill   = KillSwitch()
    notify = Notifier(channel=args.notify)

    # CSV log
    os.makedirs("logs", exist_ok=True)
    trade_csv = f"logs/managed_trades_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(trade_csv,"w",newline="") as f:
        csv.writer(f).writerow(["ts","action","spot","ceK","peK","payback","entry_credit","delta_ce","delta_pe","ivp","trend","notes"])

    def log_row(action, spot, payback, delta_ce, delta_pe, ivp, trend, notes=""):
        with open(trade_csv,"a",newline="") as f:
            csv.writer(f).writerow([
                datetime.now(IST).isoformat(timespec="seconds"), action, f"{(spot or 0):.2f}",
                int(legs.ceK) if 'legs' in locals() else -1,
                int(legs.peK) if 'legs' in locals() else -1,
                f"{payback:.2f}" if payback is not None else "NA",
                f"{legs.entry_credit:.2f}" if 'legs' in locals() else "NA",
                f"{(delta_ce or 0):.3f}", f"{(delta_pe or 0):.3f}", f"{(ivp or 0):.2f}", f"{(trend or 0):.2f}", notes
            ])

    # ----- Entry (retry until success) -----
    entry_ts = None
    cycles_since_entry = 0
    backoff = 10
    while True:
        try:
            if kill.tripped():
                LOG.warning("Kill-switch tripped before entry; exiting."); return 0
            chain = market.get_live_chain(args.underlying_scrip, args.expiry)
            legs  = enter_strangle(chain, args.delta_target, args.lots, args.lot_size, args.expiry)
            if not legs: raise RuntimeError("No CE/PE targets in this snapshot.")
            ce_leg = chain.get(str(int(legs.ceK)), {}).get("ce", {})
            pe_leg = chain.get(str(int(legs.peK)), {}).get("pe", {})
            LOG.info("ENTRY: %s SELL CE %d @ %.2f (Δ=%s) | SELL PE %d @ %.2f (Δ=%s)",
                     args.mode.upper(), int(legs.ceK), legs.ce_px,
                     f"{(ce_leg.get('greeks') or {}).get('delta'):.3f}" if (ce_leg.get('greeks') or {}).get('delta') is not None else "NA",
                     int(legs.peK), legs.pe_px,
                     f"{(pe_leg.get('greeks') or {}).get('delta'):.3f}" if (pe_leg.get('greeks') or {}).get('delta') is not None else "NA")
            broker.place("SELL","CE", legs.ceK, legs.lots, legs.ce_px, legs.expiry, underlying_scrip=args.underlying_scrip)
            broker.place("SELL","PE", legs.peK, legs.lots, legs.pe_px, legs.expiry, underlying_scrip=args.underlying_scrip)
            notify.send("Entry", f"{args.mode.upper()} short strangle: CE {int(legs.ceK)} / PE {int(legs.peK)}")
            entry_ts = datetime.now(IST)
            cycles_since_entry = 0
            break
        except Exception as e:
            LOG.warning("Entry attempt failed (%s). Retrying in %ds ...", e, backoff)
            time.sleep(backoff); backoff = min(60, backoff + 5)

    # ----- Manage loop -----
    last_mtm = last_ce_delta = last_pe_delta = None
    while True:
        try:
            if kill.tripped():
                LOG.warning("Kill-switch tripped; exiting loop."); notify.send("Kill","Kill file detected."); break
            if not market_open():
                LOG.info("Market closed; sleeping %ds", int(args.poll_sec)); time.sleep(args.poll_sec); continue

            chain = market.get_live_chain(args.underlying_scrip, args.expiry)
            payback = compute_payback(chain, legs)
            if payback is None:
                LOG.warning("Missing LTP temporarily; waiting."); time.sleep(args.poll_sec); continue

            any_row = next(iter(chain.values()))
            spot = any_row["ce"]["spot"] or any_row["pe"]["spot"] or 0.0
            ce_delta = float((chain.get(str(int(legs.ceK)), {}).get("ce", {}).get("greeks") or {}).get("delta") or 0.0)
            pe_delta = float((chain.get(str(int(legs.peK)), {}).get("pe", {}).get("greeks") or {}).get("delta") or 0.0)

            atm_key = min(chain.keys(), key=lambda k: abs(float(k) - float(spot)))
            atm_iv  = (chain[atm_key]["ce"]["iv"] or chain[atm_key]["pe"]["iv"] or 0.20)
            ivp     = min(1.0, max(0.0, (float(atm_iv) - 0.10) / (0.45 - 0.10)))
            trend   = max(-1.0, min(1.0, (ce_delta + pe_delta)))
            tte     = max(1.0, (datetime.fromisoformat(legs.expiry) - datetime.now()).days)
            mtm     = legs.entry_credit - payback
            risk.on_mtm(mtm)

            ce_ltp_now = (chain.get(str(int(legs.ceK)), {}).get("ce", {}) or {}).get("last_price") or legs.ce_px
            pe_ltp_now = (chain.get(str(int(legs.peK)), {}).get("pe", {}) or {}).get("last_price") or legs.pe_px
            hit_ce = risk.per_leg_stop_hit(legs.ce_px, ce_ltp_now)
            hit_pe = risk.per_leg_stop_hit(legs.pe_px, pe_ltp_now)

            since_entry_sec = (datetime.now(IST) - entry_ts).total_seconds() if entry_ts else 0.0

            ctx = _build_ctx(
                spot=spot, ceK=legs.ceK, peK=legs.peK,
                ce_delta=abs(ce_delta), pe_delta=abs(pe_delta),
                entry_credit=legs.entry_credit, payback_now=payback,
                time_to_expiry_days=tte, iv_percentile=ivp, trend_score=trend,
                breach_buffer_pts=args.breach_buffer,
                since_entry_sec=since_entry_sec, cycles_since_entry=cycles_since_entry,
                min_hold_sec=args.min_hold_sec, min_hold_cycles=args.min_hold_cycles
            )

            # Decision
            action = "HOLD"
            if not risk.allow_trade():
                action = "BOOK_PROFIT" if mtm > 0 else "HOLD"
            elif hit_ce and hit_pe:
                action = "ROLL_BOTH_IN"
            elif hit_ce:
                action = "ROLL_UP_CALL"
            elif hit_pe:
                action = "ROLL_DOWN_PUT"
            else:
                mtm_small = (last_mtm is not None) and (abs(mtm - last_mtm) < 100.0)
                deltas_small = (last_ce_delta is not None and last_pe_delta is not None and
                                abs(abs(ce_delta) - abs(last_ce_delta)) < 0.01 and
                                abs(abs(pe_delta) - abs(last_pe_delta)) < 0.01)
                if not (mtm_small and deltas_small):
                    action = agent.decide(ctx)

            LOG.info("Context: spot=%.1f | payback=%.0f | mtm=%.0f | |ΔCE|=%.3f | |ΔPE|=%.3f | IVp=%.2f | trend=%.2f | action=%s",
                     float(spot), float(payback), float(mtm), abs(ce_delta), abs(pe_delta), ivp, trend, action)

            before = payback; cost = 0.0
            if action == "HOLD":
                pass
            elif action == "BOOK_PROFIT":
                ce_ltp = (chain.get(str(int(legs.ceK)), {}).get("ce", {}) or {}).get("last_price") or ce_ltp_now
                pe_ltp = (chain.get(str(int(legs.peK)), {}).get("pe", {}) or {}).get("last_price") or pe_ltp_now
                broker.close("CE", legs.ceK, legs.lots, float(ce_ltp), legs.expiry, underlying_scrip=args.underlying_scrip)
                broker.close("PE", legs.peK, legs.lots, float(pe_ltp), legs.expiry, underlying_scrip=args.underlying_scrip)
                log_row("BOOK_PROFIT", spot, payback, abs(ce_delta), abs(pe_delta), ivp, trend, "final exit")
                agent.save(); notify.send("Exit", f"Booked. MTM={mtm:.0f}"); break
            elif action == "ROLL_UP_CALL":
                legs2, pnl = roll_up_call(chain, legs, args.delta_target, broker)
                if legs2: legs = legs2; cost = -pnl; risk.on_roll(); notify.send("Roll", f"Up call → {int(legs.ceK)}")
            elif action == "ROLL_DOWN_PUT":
                legs2, pnl = roll_down_put(chain, legs, args.delta_target, broker)
                if legs2: legs = legs2; cost = -pnl; risk.on_roll(); notify.send("Roll", f"Down put → {int(legs.peK)}")
            elif action == "ROLL_BOTH_IN":
                legs2, pnl = roll_both_in(chain, legs, args.delta_target, broker)
                if legs2: legs = legs2; cost = -pnl; risk.on_roll(); notify.send("Roll", f"Both in; CE {int(legs.ceK)} PE {int(legs.peK)}")
            elif action == "ADD_HEDGE_CALL":
                cost = -add_hedge(chain, legs, True, broker);  notify.send("Hedge", "Added call wing")
            elif action == "ADD_HEDGE_PUT":
                cost = -add_hedge(chain, legs, False, broker); notify.send("Hedge", "Added put wing")

            chain2 = market.get_live_chain(args.underlying_scrip, args.expiry)
            after  = compute_payback(chain2, legs) or before
            agent.learn(action, before, after, cost); agent.save()
            log_row(action, spot, after, abs(ce_delta), abs(pe_delta), ivp, trend, f"cost={cost:.2f}")

            cycles_since_entry += 1
            last_mtm, last_ce_delta, last_pe_delta = mtm, ce_delta, pe_delta
            time.sleep(args.poll_sec)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            LOG.exception("Manage loop error: %s", e)
            time.sleep(max(5, int(args.poll_sec)))
            continue

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
