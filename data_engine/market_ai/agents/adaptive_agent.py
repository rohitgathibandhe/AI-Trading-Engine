# -*- coding: utf-8 -*-
"""
Adaptive Agent — polls UI commands and reacts.

Drop this at:
data_engine/market_ai/agents/adaptive_agent.py

The Streamlit UI can communicate with the agent by appending JSON
commands to: data_engine/market_ai/runtime/ui_commands.jsonl

Each line is a JSON object like:
{"cmd":"set_mode","mode":"paper"}                   # "paper" | "live"
{"cmd":"set_indices","indices":["NIFTY","BANKNIFTY"]}
{"cmd":"start"}                                     # start monitoring/trading
{"cmd":"stop"}                                      # stop monitoring/trading
{"cmd":"shutdown"}                                  # agent exits
{"cmd":"trade","what":"initiate","symbol":"NIFTY"}  # placeholder

Agent writes current state to:
data_engine/market_ai/runtime/agent_state.json

And logs to:
data_engine/market_ai/runtime/agent_log.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

# ---------- paths
ROOT = os.path.dirname(os.path.dirname(__file__))
RUNTIME_DIR = os.path.join(ROOT, "runtime")
os.makedirs(RUNTIME_DIR, exist_ok=True)
CMD_FILE = os.path.join(RUNTIME_DIR, "ui_commands.jsonl")
STATE_FILE = os.path.join(RUNTIME_DIR, "agent_state.json")
LOG_FILE = os.path.join(RUNTIME_DIR, "agent_log.jsonl")
PAPER_STATE_FILE = os.path.join(RUNTIME_DIR, "paper_state.json")

# ---------- logger
def _log(ev: str, meta: Optional[dict] = None) -> None:
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": ev,
        "meta": meta or {},
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        # also print to stderr
        print("[agent-log-failed]", rec, file=sys.stderr)

# ---------- utils
def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")

# ---------- optional imports (Dhan + Paper)
# Dhan wrapper (your latest one)
try:
    from data_engine.market_ai.dhan_wrapper import DhanWrapper  # absolute
    from data_engine.market_ai.dhan_wrapper import Leg, OrderSide
except Exception:
    # fallback to local import if PYTHONPATH differs or agent loaded differently
    try:
        import importlib
        mod = importlib.import_module("dhan_wrapper")
        DhanWrapper, Leg, OrderSide = mod.DhanWrapper, mod.Leg, mod.OrderSide
    except Exception as e:
        DhanWrapper = None  # type: ignore
        Leg = None          # type: ignore
        OrderSide = None    # type: ignore
        _log("warn", {"msg": "DhanWrapper not available", "err": str(e)})

# Minimal PaperBroker (works even if your infra module differs)
DEFAULT_VIRTUAL_CASH = 500_000.0

class PaperBroker:
    def __init__(self, state_file: str = PAPER_STATE_FILE, initial_cash: float = DEFAULT_VIRTUAL_CASH):
        self.path = state_file
        st = _read_json(self.path, None)
        if not st:
            st = {"cash": float(initial_cash), "positions": []}
            _write_json(self.path, st)
        self.state = st

    def funds(self) -> Dict[str, float]:
        # Very simple model: cash = available = withdrawable, collateral/utilized = 0
        s = _read_json(self.path, self.state)
        return {
            "availableBalance": float(s.get("cash", 0.0)),
            "collateral": 0.0,
            "utilized": 0.0,
            "withdrawable": float(s.get("cash", 0.0)),
        }

    def place_dummy_trade(self, symbol: str, notional: float = 10_000.0) -> Dict[str, Any]:
        s = _read_json(self.path, self.state)
        cash = float(s.get("cash", 0.0))
        if cash < notional:
            return {"status": "failure", "error": "Insufficient virtual cash"}
        cash -= notional
        pos = {
            "id": f"paper-{int(time.time())}",
            "symbol": symbol,
            "notional": notional,
            "ts": _now(),
            "side": "BUY",
        }
        s["cash"] = cash
        s.setdefault("positions", []).append(pos)
        _write_json(self.path, s)
        self.state = s
        return {"status": "success", "position": pos, "cash": cash}

# ---------- agent state
@dataclass
class AgentState:
    mode: str = "paper"                   # "paper" | "live"
    running: bool = False
    indices: List[str] = None            # ["NIFTY", "BANKNIFTY"]
    last_funds: Dict[str, Any] = None
    last_prices: Dict[str, Optional[float]] = None
    last_error: Optional[str] = None
    last_updated: str = _now()

    def to_dict(self):
        d = asdict(self)
        if d["indices"] is None:
            d["indices"] = []
        if d["last_prices"] is None:
            d["last_prices"] = {}
        if d["last_funds"] is None:
            d["last_funds"] = {}
        return d

# ---------- AdaptiveAgent
class AdaptiveAgent:
    def __init__(self, cfg: Optional[dict] = None, risk_cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.risk_cfg = risk_cfg or {}
        self.state = AgentState(indices=[], last_prices={})
        self.paper = PaperBroker()
        self.dhan: Optional[DhanWrapper] = None  # lazy until credentials arrive
        self._last_cmd_pos = 0  # file read position
        self._symbols = {
            "NIFTY": {"exchange_seg": "IDX_I", "security_id": 13},
            "BANKNIFTY": {"exchange_seg": "IDX_I", "security_id": 25},
        }

    # ---- lifecycle
    def start(self):
        _log("agent_start")
        self._persist_state()

        # MAIN LOOP — polls UI commands and reacts
        while True:
            try:
                for cmd in self._drain_commands():
                    self._handle_command(cmd)

                if self.state.running:
                    # periodic work: fetch funds + prices
                    self._tick()
                else:
                    time.sleep(0.2)

            except KeyboardInterrupt:
                _log("agent_keyboard_interrupt")
                break
            except SystemExit:
                break
            except Exception as e:
                tb = traceback.format_exc()
                self.state.last_error = str(e)
                _log("agent_uncaught_error", {"err": str(e), "trace": tb})
                self._persist_state()
                time.sleep(1.0)

        _log("agent_exit")

    # ---- command handling
    def _drain_commands(self) -> List[dict]:
        cmds: List[dict] = []
        # ensure file exists
        if not os.path.exists(CMD_FILE):
            open(CMD_FILE, "a", encoding="utf-8").close()

        with open(CMD_FILE, "r", encoding="utf-8") as f:
            f.seek(self._last_cmd_pos)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cmds.append(json.loads(line))
                except Exception as e:
                    _log("bad_command_json", {"line": line, "err": str(e)})
            self._last_cmd_pos = f.tell()
        return cmds

    def _handle_command(self, cmd: dict) -> None:
        kind = cmd.get("cmd")
        _log("cmd", {"cmd": kind, "payload": cmd})

        if kind == "shutdown":
            _log("shutdown_requested")
            self._persist_state()
            sys.exit(0)

        if kind == "start":
            self.state.running = True
            self.state.last_error = None
            self._persist_state()
            return

        if kind == "stop":
            self.state.running = False
            self._persist_state()
            return

        if kind == "set_mode":
            mode = str(cmd.get("mode", "paper")).lower()
            if mode not in ("paper", "live"):
                self.state.last_error = f"invalid mode: {mode}"
            else:
                self.state.mode = mode
                self.state.last_error = None
            self._persist_state()
            return

        if kind == "set_indices":
            items = cmd.get("indices") or []
            valid = [s for s in items if s in self._symbols]
            self.state.indices = valid
            self._persist_state()
            return

        if kind == "set_credentials":
            cid = (cmd.get("client_id") or "").strip()
            tok = (cmd.get("access_token") or "").strip()
            if not cid or not tok:
                self.state.last_error = "Missing client_id/access_token"
                self._persist_state()
                return
            self._init_dhan(cid, tok)
            return

        if kind == "trade":
            what = cmd.get("what")
            symbol = cmd.get("symbol")
            if what == "initiate" and symbol in self._symbols:
                self._initiate_trade(symbol)
            else:
                self.state.last_error = f"invalid trade cmd: {cmd}"
                self._persist_state()
            return

        # unknown
        self.state.last_error = f"unknown cmd: {cmd}"
        self._persist_state()

    # ---- periodic tick
    def _tick(self) -> None:
        # funds
        self.state.last_funds = self._fetch_funds()
        # prices
        prices: Dict[str, Optional[float]] = {}
        for sym in (self.state.indices or []):
            prices[sym] = self._fetch_ltp(sym)
        self.state.last_prices = prices
        self.state.last_updated = _now()
        self._persist_state()
        time.sleep(0.8)

    # ---- helpers
    def _persist_state(self) -> None:
        _write_json(STATE_FILE, self.state.to_dict())

    def _init_dhan(self, client_id: str, access_token: str):
        if DhanWrapper is None:
            self.state.last_error = "DhanWrapper not available"
            self._persist_state()
            return
        try:
            self.dhan = DhanWrapper(client_id, access_token)
            self.state.last_error = None
            _log("dhan_ready")
            self._persist_state()
        except Exception as e:
            self.state.last_error = f"Dhan init failed: {e}"
            _log("dhan_init_error", {"err": str(e)})
            self._persist_state()

    def _fetch_funds(self) -> Dict[str, Any]:
        if self.state.mode == "paper":
            return self.paper.funds()
        if not self.dhan:
            return {"error": "no_dhan"}
        try:
            data = self.dhan.get_funds()
            # normalize keys expected by UI
            return {
                "availableBalance": data.get("availableBalance") or data.get("availabelBalance") or 0.0,
                "collateral": data.get("collateral", 0.0),
                "utilized": data.get("utilized", 0.0),
                "withdrawable": data.get("withdrawable", data.get("availableBalance", 0.0)),
                "_raw": data,
            }
        except Exception as e:
            _log("funds_error", {"err": str(e)})
            return {"error": str(e)}

    def _fetch_ltp(self, symbol: str) -> Optional[float]:
        meta = self._symbols[symbol]
        if not self.dhan:
            return None
        try:
            return self.dhan.get_ltp_once(meta["exchange_seg"], meta["security_id"])
        except Exception as e:
            _log("ltp_error", {"symbol": symbol, "err": str(e)})
            return None

    def _initiate_trade(self, symbol: str) -> None:
        if self.state.mode == "paper":
            res = self.paper.place_dummy_trade(symbol)
            if res.get("status") == "success":
                _log("paper_trade_ok", {"symbol": symbol, "pos": res.get("position")})
                self.state.last_error = None
            else:
                _log("paper_trade_fail", {"symbol": symbol, "err": res.get("error")})
                self.state.last_error = res.get("error")
            self._persist_state()
            return

        # live mode (placeholder — wire with your real strategy)
        if not self.dhan:
            self.state.last_error = "No Dhan credentials"
            self._persist_state()
            return
        # TODO: call your live execution here
        _log("live_trade_stub", {"symbol": symbol})
        self.state.last_error = None
        self._persist_state()

# ---------- loader for start_agent.py compatibility
def _wire_dhan_wrapper():
    """
    start_agent.py dynamically imports this module and expects
    DhanWrapper, Leg, OrderSide names to exist at module level.
    We already expose those via the optional imports above.
    """
    pass

# ---------- main
if __name__ == "__main__":
    # Optional cfg load — keep trivial for now
    cfg = {}
    risk_cfg = {}
    agent = AdaptiveAgent(cfg, risk_cfg)
    _log("boot", {"cfg": cfg})
    agent.start()
