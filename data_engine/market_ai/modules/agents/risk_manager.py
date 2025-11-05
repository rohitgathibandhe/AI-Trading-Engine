# modules/agents/risk_manager.py
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import logging

log = logging.getLogger("RiskManager")
log.propagate = False
if not log.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [RiskManager] %(message)s")
    h.setFormatter(fmt)
    log.addHandler(h)
log.setLevel(logging.INFO)

@dataclass
class EntryGateDecision:
    allow: bool
    reason: str

@dataclass
class RiskConfig:
    # Hard loss caps
    max_loss_pct_trade: float = 0.015     # 1.5% of deployed capital (per trade)
    max_loss_pct_day: float = 0.025       # 2.5% of account equity (per day)
    # Vol filters
    ivpct_min: int = 10
    ivpct_max: int = 85
    vix_min: float = 10.0
    vix_max: float = 25.0
    # Gap / event gates
    gap_thr_pct: float = 0.5              # skip if overnight move > 0.5%
    block_event_days: bool = True
    # Fallback equity if adapter not available
    default_account_equity: float = 500000.0

def _filter_to_risk_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return only keys that exist in RiskConfig, from possibly nested dicts."""
    allowed = {f.name for f in fields(RiskConfig)}
    # If config is sectioned (e.g., {'general':..., 'risk':{...}}), prefer 'risk'
    if "risk" in data and isinstance(data["risk"], dict):
        source = data["risk"]
    else:
        source = data
    return {k: v for k, v in source.items() if k in allowed}

class RiskManager:
    """
    Single source of truth for risk gates & hard stops.

    Backward-compatible constructor:
      RiskManager(cfg: Dict[str, Any] | RiskConfig,
                  account_adapter: Optional[Any] = None,
                  **kwargs)

    - Accepts full YAML-loaded config dicts with sections (e.g., 'general', 'risk', ...).
    - Silently ignores unknown keys.
    - Optionally uses account_adapter for equity/day-PnL if caller doesn't pass them.
    """
    def __init__(self,
                 cfg: Dict[str, Any] | RiskConfig,
                 account_adapter: Optional[Any] = None,
                 **kwargs):
        if isinstance(cfg, dict):
            filtered = _filter_to_risk_fields(cfg)
            self.cfg = RiskConfig(**filtered)
        else:
            self.cfg = cfg
        self.account_adapter = account_adapter  # may be None
        self._pause_mode = False
        self._day_stamp = None
        self._day_realized_loss = 0.0

        log.info("Risk config loaded: %s", self.cfg)

    # ----- Pause mode (daily circuit, operator override) -----
    def set_pause(self, v: bool):
        self._pause_mode = bool(v)
        log.info("Pause mode = %s", self._pause_mode)

    def paused(self) -> bool:
        return self._pause_mode

    # ----- Entry Gates -----
    def check_entry_gates(self, features: Dict[str, Any]) -> EntryGateDecision:
        """
        features expects:
          iv_percentile, vix, overnight_move_pct, event_day (bool)
        """
        if self.paused():
            return EntryGateDecision(False, "pause_mode")

        ivp = float(features.get("iv_percentile", 50))
        if not (self.cfg.ivpct_min <= ivp <= self.cfg.ivpct_max):
            return EntryGateDecision(False, f"ivpct_gate:{ivp:.0f}")

        vix = float(features.get("vix", 14.0))
        if not (self.cfg.vix_min <= vix <= self.cfg.vix_max):
            return EntryGateDecision(False, f"vix_gate:{vix:.1f}")

        gap = float(features.get("overnight_move_pct", 0.0))
        if gap > self.cfg.gap_thr_pct:
            return EntryGateDecision(False, f"gap_gate:{gap:.2f}%")

        if self.cfg.block_event_days and bool(features.get("event_day", False)):
            return EntryGateDecision(False, "event_day_gate")

        return EntryGateDecision(True, "OK")

    # ----- Day tracking (optional) -----
    def _roll_day(self):
        today = datetime.now().date()
        if self._day_stamp != today:
            self._day_stamp = today
            self._day_realized_loss = 0.0

    def record_realized_pnl(self, pnl: float):
        self._roll_day()
        self._day_realized_loss += min(0.0, pnl)

    # ----- Adapter helpers -----
    def _resolve_account_equity(self, fallback_equity: Optional[float]) -> float:
        if fallback_equity is not None:
            return float(fallback_equity)
        if self.account_adapter and hasattr(self.account_adapter, "get_equity"):
            try:
                return float(self.account_adapter.get_equity())
            except Exception as e:
                log.warning("account_adapter.get_equity() failed: %s", e)
        return float(self.cfg.default_account_equity)

    def _resolve_day_loss_abs(self, fallback_day_abs: Optional[float]) -> float:
        if fallback_day_abs is not None:
            return float(fallback_day_abs)
        if self.account_adapter and hasattr(self.account_adapter, "get_intraday_realized_pnl_abs"):
            try:
                return float(self.account_adapter.get_intraday_realized_pnl_abs())
            except Exception as e:
                log.warning("account_adapter.get_intraday_realized_pnl_abs() failed: %s", e)
        self._roll_day()
        return abs(self._day_realized_loss)

    # ----- Hard Stops -----
    def check_hard_stops(self,
                         pnl_trade_abs: Optional[float] = None,
                         pnl_day_abs: Optional[float] = None,
                         deployed_capital: Optional[float] = None,
                         account_equity: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """
        Return (hit, reason) if any hard stop triggers.
        Any parameter may be None; we resolve what we can.
        """
        acct_equity = self._resolve_account_equity(account_equity)
        day_abs = self._resolve_day_loss_abs(pnl_day_abs)

        # per-trade hard stop (only if both provided)
        if pnl_trade_abs is not None and deployed_capital is not None:
            max_trade_loss = self.cfg.max_loss_pct_trade * float(deployed_capital)
            if float(pnl_trade_abs) >= max_trade_loss:
                log.warning("Trade hard stop hit: trade_loss=%.2f >= %.2f", float(pnl_trade_abs), max_trade_loss)
                return True, "trade_max_loss"

        # daily circuit breaker
        max_day_loss = self.cfg.max_loss_pct_day * acct_equity
        if day_abs >= max_day_loss:
            log.error("Daily circuit hit: day_loss=%.2f >= %.2f", day_abs, max_day_loss)
            self.set_pause(True)
            return True, "daily_circuit"

        return False, None
