# adaptive_agent.py
from __future__ import annotations
import json, time, threading, signal, math, os, tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timezone

import logging
log = logging.getLogger("AdaptiveAgent")
log.propagate = False
if not log.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [AdaptiveAgent] %(message)s")
    h.setFormatter(fmt)
    log.addHandler(h)
log.setLevel(logging.INFO)

# --- Cautious imports to match your tree --------------------------------------
try:
    from risk_manager import RiskManager, EntryGateDecision
except Exception:
    # Minimal fallback to prevent hard crashes if import path differs.
    @dataclass
    class EntryGateDecision:
        allow: bool
        reason: str
    class RiskManager:
        def __init__(self, cfg: Dict[str, Any]): self.cfg = cfg
        def check_entry_gates(self, features: Dict[str, Any]) -> EntryGateDecision:
            return EntryGateDecision(True, "OK")
        def check_hard_stops(self, pnl_trade_abs: float, pnl_day_abs: float,
                             deployed_capital: float, account_equity: float) -> Tuple[bool, Optional[str]]:
            return False, None
        def set_pause(self, v: bool): pass
        def paused(self) -> bool: return False

try:
    from dhan_wrapper import DhanWrapper, OrderSide, OrderType, Leg, ReconciliationReport
except Exception:
    class OrderSide: BUY="BUY"; SELL="SELL"
    class OrderType: MARKET="MARKET"; LIMIT="LIMIT"
    @dataclass
    class Leg:
        symbol: str
        qty: int
        side: str
        order_type: str = OrderType.MARKET
        client_order_id: Optional[str] = None
        price: Optional[float] = None
    @dataclass
    class ReconciliationReport:
        fixed: bool; notes: str
    class DhanWrapper:
        def positions(self) -> List[Dict[str, Any]]: return []
        def place_legs_idem(self, legs: List[Leg]) -> List[Dict[str, Any]]: return []
        def reconcile_positions(self, intended_legs: List[Leg]) -> ReconciliationReport:
            return ReconciliationReport(True, "noop")

# -----------------------------------------------------------------------------

@dataclass
class AgentConfig:
    state_path: str = "./state/trade_state.json"
    heartbeat_sec: int = 30
    save_interval_sec: int = 30
    # soft gates (used by your scheduler/top-down regime):
    trading_start_hour_ist: int = 9
    trading_end_hour_ist: int = 15
    # optional: identifier to tag client_order_id prefixes
    client_tag: str = "alog"

@dataclass
class OpenPosition:
    # minimal persistent shape; extend as needed
    entry_ts: str
    entry_spot: float
    short_pe: str
    short_ce: str
    wings: List[str]
    lots: int
    net_credit_per_lot: float
    deployed_margin_per_lot: float
    unrealized: float = 0.0
    exit_reason: str = ""

@dataclass
class AgentState:
    account_equity: float
    day_pnl: float
    open_position: Optional[OpenPosition]
    pause_mode: bool
    last_heartbeat_ts: str

class AdaptiveAgent:
    """
    Add persistence + recovery to your existing loop:
    - JSON snapshot every N seconds
    - Cold-start recovery: load → broker reconcile → resume monitors
    - Heartbeat logging
    """
    def __init__(self,
                 config: AgentConfig,
                 risk_cfg: Dict[str, Any],
                 dhan: Optional[DhanWrapper] = None):
        self.cfg = config
        self.risk = RiskManager(risk_cfg)
        self.dhan = dhan or DhanWrapper()
        self._stop = threading.Event()
        self._lock = threading.RLock()

        self._state_path = Path(self.cfg.state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        # runtime vars
        self.account_equity = float(risk_cfg.get("account_equity", 500000.0))
        self.day_pnl = 0.0
        self.open_position: Optional[OpenPosition] = None
        self.loop_count = 0
        self._last_saved = 0.0

        signal.signal(signal.SIGINT, self._sig_stop)
        signal.signal(signal.SIGTERM, self._sig_stop)

    # ---------- Persistence ----------
    def _snapshot(self) -> AgentState:
        return AgentState(
            account_equity=self.account_equity,
            day_pnl=self.day_pnl,
            open_position=self.open_position,
            pause_mode=self.risk.paused(),
            last_heartbeat_ts=datetime.now(timezone.utc).isoformat()
        )

    def save_state_json(self):
        snap = self._snapshot()
        data = json.dumps(asdict(snap), ensure_ascii=False, separators=(",", ":"))
        tmp = self._state_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._state_path)

    def load_state_json(self) -> Optional[AgentState]:
        if not self._state_path.exists():
            return None
        with open(self._state_path, "r") as f:
            d = json.load(f)
        # rebuild nested dataclasses
        op = d.get("open_position")
        open_position = OpenPosition(**op) if op else None
        return AgentState(
            account_equity=float(d.get("account_equity", 0.0)),
            day_pnl=float(d.get("day_pnl", 0.0)),
            open_position=open_position,
            pause_mode=bool(d.get("pause_mode", False)),
            last_heartbeat_ts=d.get("last_heartbeat_ts", "")
        )

    # ---------- Recovery / Reconciliation ----------
    def cold_start_recovery(self) -> None:
        """
        On process start:
         1) Load last state file (if present)
         2) Pull broker positions
         3) Generate intended legs from state.open_position
         4) Ask DhanWrapper to reconcile
         5) Re-arm monitors
        """
        st = self.load_state_json()
        if not st:
            log.info("No prior state found — starting clean.")
            return

        self.account_equity = st.account_equity
        self.day_pnl = st.day_pnl
        if st.pause_mode:
            self.risk.set_pause(True)

        self.open_position = st.open_position
        if not self.open_position:
            log.info("Recovered prior state (no open position).")
            return

        intended = self._intended_legs_from_open(self.open_position)
        rep = self.dhan.reconcile_positions(intended)
        log.info("Recovery reconcile: fixed=%s notes=%s", rep.fixed, rep.notes)
        self._resume_monitors()

    def _intended_legs_from_open(self, op: OpenPosition) -> List[Leg]:
        # Build minimal picture of intended legs (short strangle + wings)
        legs: List[Leg] = []
        lot_qty = self._lot_qty()

        legs.append(Leg(symbol=op.short_ce, qty=lot_qty, side=OrderSide.SELL))
        legs.append(Leg(symbol=op.short_pe, qty=lot_qty, side=OrderSide.SELL))
        for w in op.wings:
            legs.append(Leg(symbol=w, qty=lot_qty, side=OrderSide.BUY))
        return legs

    def _resume_monitors(self):
        # Placeholders: in your real loop, re-arm delta/vega watchers, timers, etc.
        log.info("Monitors resumed for existing position.")

    # ---------- Utility ----------
    def _lot_qty(self) -> int:
        # Plug in your instrument lot size if needed; default 50 (Nifty)
        return 50

    def _now_ist_hour(self) -> int:
        # quick IST hour: IST = UTC+5:30
        return (datetime.utcnow().hour + 5 + ((datetime.utcnow().minute+30)//60)) % 24

    # ---------- Public lifecycle ----------
    def start(self):
        log.info("==============================================")
        log.info("🚀 Agent started at %s", datetime.now().isoformat())
        log.info("Using state: %s", self._state_path)
        log.info("==============================================")
        self.cold_start_recovery()
        self._loop()

    def stop(self):
        self._stop.set()

    def _sig_stop(self, *_):
        log.info("🛑 Stop signal received — exiting cleanly...")
        self.stop()

    # ---------- Main loop skeleton (call your strategy inside) ----------
    def _loop(self):
        while not self._stop.is_set():
            try:
                self.loop_count += 1
                hour = self._now_ist_hour()

                # Example gate (keep your own “topdown_regime” too)
                if not (self.cfg.trading_start_hour_ist <= hour <= self.cfg.trading_end_hour_ist):
                    log.info("[IDLE] Gates blocked: ['time_window']")
                    self._heartbeat_and_maybe_save()
                    time.sleep(10)
                    continue

                # 1) Compute features (replace with your pipeline)
                features = self._mock_features()

                # 2) Hard entry gates
                gate = self.risk.check_entry_gates(features)
                if self.risk.paused() or not gate.allow:
                    why = "pause_mode" if self.risk.paused() else gate.reason
                    log.info("[IDLE] Gates blocked: ['%s']", why)
                    self._heartbeat_and_maybe_save()
                    time.sleep(10)
                    continue

                # 3) If no open position, enter via your strategy (placeholder)
                if self.open_position is None:
                    self._maybe_open_position(features)
                else:
                    # 4) Manage open position (delta, exits, etc.)
                    self._manage_position(features)

                self._heartbeat_and_maybe_save()
                time.sleep(10)

            except Exception as e:
                log.exception("Loop error: %s", e)
                time.sleep(2)

        # On stop, persist final snapshot
        self.save_state_json()
        log.info("✅ Agent stopped gracefully.")

    # ---------- Strategy placeholders (replace with your real ones) ----------
    def _maybe_open_position(self, features: Dict[str, Any]):
        # Example entry creating OpenPosition; replace with your strike selection
        spot = float(features.get("spot", 24000))
        short_ce = features.get("short_ce_symbol", "NIFTY24OCT24000CE")
        short_pe = features.get("short_pe_symbol", "NIFTY24OCT24000PE")
        wings = [
            features.get("wing_ce_symbol", "NIFTY24OCT24300CE"),
            features.get("wing_pe_symbol", "NIFTY24OCT23700PE"),
        ]
        lots = int(features.get("lots", 1))
        net_credit = float(features.get("net_credit_per_lot", 5000.0))
        margin_per_lot = float(features.get("deployed_margin_per_lot", 180000.0))

        legs = self._intended_legs_from_open(
            OpenPosition(
                entry_ts=datetime.now(timezone.utc).isoformat(),
                entry_spot=spot,
                short_pe=short_pe,
                short_ce=short_ce,
                wings=wings,
                lots=lots,
                net_credit_per_lot=net_credit,
                deployed_margin_per_lot=margin_per_lot
            )
        )
        # Idempotent placement
        self.dhan.place_legs_idem(legs)

        self.open_position = OpenPosition(
            entry_ts=datetime.now(timezone.utc).isoformat(),
            entry_spot=spot,
            short_pe=short_pe,
            short_ce=short_ce,
            wings=wings,
            lots=lots,
            net_credit_per_lot=net_credit,
            deployed_margin_per_lot=margin_per_lot
        )
        log.info("Opened position: CE=%s PE=%s wings=%s lots=%s", short_ce, short_pe, wings, lots)

    def _manage_position(self, features: Dict[str, Any]):
        # Simplified MTM & guards: hook into your Greeks/payoff
        move = abs((float(features.get("spot", 24000)) - self.open_position.entry_spot) / max(self.open_position.entry_spot, 1))
        mtm = self.open_position.net_credit_per_lot * self.open_position.lots * (0.02 - 5.0 * max(0.0, move - 0.01))
        self.open_position.unrealized += mtm
        self.day_pnl += mtm

        # Hard stops
        used = self.open_position.deployed_margin_per_lot * self.open_position.lots
        hit, reason = self.risk.check_hard_stops(
            pnl_trade_abs=abs(self.open_position.unrealized),
            pnl_day_abs=abs(self.day_pnl),
            deployed_capital=used,
            account_equity=self.account_equity
        )
        if hit:
            self._exit_all(reason)
            return

        # Profit booking target (~70% of credit) — you likely already have this
        target = 0.7 * self.open_position.net_credit_per_lot * self.open_position.lots
        if self.open_position.unrealized >= target:
            self._exit_all("target_hit")

    def _exit_all(self, reason: str):
        # Build opposite legs to flatten; here we just mark exit
        self.open_position.exit_reason = reason
        log.info("Exiting all legs — reason=%s", reason)
        # In your real code: send close legs idempotently; reconcile; confirm flat
        self.open_position = None
        # cool-down if daily circuit
        if reason == "daily_circuit":
            self.risk.set_pause(True)

    # ---------- Housekeeping ----------
    def _heartbeat_and_maybe_save(self):
        if self.loop_count % 30 == 0:
            log.info("Heartbeat — still running (%d loops)", self.loop_count)
        now = time.time()
        if now - self._last_saved >= self.cfg.save_interval_sec:
            with self._lock:
                self.save_state_json()
                self._last_saved = now

    # ---------- Mock features (replace) ----------
    def _mock_features(self) -> Dict[str, Any]:
        return {
            "spot": 24000.0,
            "iv_percentile": 35,
            "vix": 14.5,
            "overnight_move_pct": 0.2,
            "event_day": False,
            "lots": 1,
            "net_credit_per_lot": 5000.0,
            "deployed_margin_per_lot": 180000.0
        }

if __name__ == "__main__":
    cfg = AgentConfig()
    risk_cfg = {
        "account_equity": 500000.0,
        "max_loss_pct_trade": 0.015,
        "max_loss_pct_day": 0.025,
        "ivpct_min": 10,
        "ivpct_max": 85,
        "vix_min": 10.0,
        "vix_max": 25.0,
        "gap_thr_pct": 0.5
    }
    agent = AdaptiveAgent(cfg, risk_cfg)
    agent.start()
