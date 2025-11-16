"""
Watch the live monitor snapshot and trigger alerts / auto-actions when risk flags change.

Actions:
- On any position with risk_flag == "exit": write manual_action_status.json requesting flatten,
  and append an alert line to state/live_monitor/alerts.jsonl.
- On risk_flag == "caution": append an alert noting caution.

This is intentionally conservative: it does NOT place orders directly. The existing strategy
loop can observe manual_action_status.json and react (e.g., flatten/hedge).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import MANUAL_ACTION_STATUS

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "state"
LIVE_MONITOR_DIR = STATE_DIR / "live_monitor"
ALERTS = LIVE_MONITOR_DIR / "alerts.jsonl"
LAST_FLAGS = LIVE_MONITOR_DIR / "last_flags.json"


def _load_snapshot() -> Dict[str, Any]:
    snap = LIVE_MONITOR_DIR / "latest.json"
    if not snap.exists():
        return {}
    try:
        return json.loads(snap.read_text())
    except Exception:
        return {}


def _load_last_flags() -> Dict[str, str]:
    if not LAST_FLAGS.exists():
        return {}
    try:
        return json.loads(LAST_FLAGS.read_text())
    except Exception:
        return {}


def _save_last_flags(flags: Dict[str, str]) -> None:
    LAST_FLAGS.parent.mkdir(parents=True, exist_ok=True)
    LAST_FLAGS.write_text(json.dumps(flags, indent=2))


def _append_alert(record: Dict[str, Any]) -> None:
    ALERTS.parent.mkdir(parents=True, exist_ok=True)
    with ALERTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def _write_manual_action(action: str, message: str) -> None:
    payload = {
        "action": action,
        "status": "requested",
        "message": message,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    MANUAL_ACTION_STATUS.write_text(json.dumps(payload, indent=2))


def run_once() -> None:
    snap = _load_snapshot()
    positions = snap.get("positions") or []
    gap_pct = snap.get("gap_pct")
    spot = snap.get("spot")
    nearest_res = None
    nearest_sup = None
    zones = snap.get("zones") or []
    events = snap.get("events") or []
    try:
        if zones and spot is not None:
            supports = [z for z in zones if (z.get("side") == "support" and z.get("price") is not None)]
            resistances = [z for z in zones if (z.get("side") == "resistance" and z.get("price") is not None)]
            if supports:
                nearest_sup = min(supports, key=lambda z: abs(float(z.get("price")) - float(spot)))
            if resistances:
                nearest_res = min(resistances, key=lambda z: abs(float(z.get("price")) - float(spot)))
    except Exception:
        pass
    last_flags = _load_last_flags()
    new_flags: Dict[str, str] = {}
    # Event-based caution: upcoming holiday or high importance regulatory/macro in +/-1 day
    req_caution = False
    now = datetime.now()
    for ev in events:
        try:
            ts = ev.get("timestamp")
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if abs((dt - now).total_seconds()) <= 36 * 3600 and int(ev.get("importance", 1)) >= 3:
                    req_caution = True
                    break
        except Exception:
            continue

    for pos in positions:
        symbol = str(pos.get("symbol") or pos.get("security_id") or "unknown")
        flag = str(pos.get("risk_flag") or "safe")
        new_flags[symbol] = flag
        prev = last_flags.get(symbol)
        if prev == flag:
            continue
        rec = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "symbol": symbol,
            "from": prev,
            "to": flag,
            "pnl": pos.get("pnl"),
            "pnl_pct": pos.get("pnl_pct"),
            "notes": pos.get("notes"),
        }
        if flag == "exit":
            _write_manual_action("flatten_all", f"Auto-alert: {symbol} risk_flag exit")
            rec["action"] = "flatten_all_requested"
        elif flag == "caution":
            rec["action"] = "caution"
        # Gap/resistance awareness: if gap-up into resistance, request hedge refresh
        if gap_pct is not None and gap_pct > 0.003 and nearest_res and spot:
            res_px = float(nearest_res.get("price"))
            buffer_pct = (res_px - float(spot)) / max(float(spot), 1.0)
            if buffer_pct >= -0.001:  # at or below resistance by ~0.1%
                _write_manual_action("force_hedge_refresh", f"Gap-up near resistance {res_px:.1f}")
                rec["action"] = rec.get("action") or "hedge_refresh"
        if req_caution and rec.get("action") is None:
            rec["action"] = "event_caution"
        _append_alert(rec)

    _save_last_flags(new_flags)


def main():
    run_once()


if __name__ == "__main__":
    main()
