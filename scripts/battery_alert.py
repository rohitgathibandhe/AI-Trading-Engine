#!/usr/bin/env python3
"""Telegram alert when the Mac battery is below a threshold and discharging (owner request 2026-10-18).
Run every 5 min by launchd (com.algoagent.battery_alert). Re-alerts at most every 30 min while still low."""
import json, re, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data_engine" / "market_ai" / "state" / "battery_alert_state.json"
import os
THRESHOLD = int(os.environ.get("BATTERY_ALERT_THRESHOLD", "25"))
out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
m = re.search(r"(\d+)%;\s*(\w+)", out)
if not m:
    sys.exit(0)
pct, status = int(m.group(1)), m.group(2)
st = json.loads(STATE.read_text()) if STATE.exists() else {}
if pct < THRESHOLD and status != "charging" and "AC Power" not in out:
    if time.time() - float(st.get("last_alert", 0)) >= 1800:
        msg = (f"Mac battery {pct}% and {status} — plug in the charger, or the trading agent will stop when it dies.")
        subprocess.run(["/Users/Rohit/.venvs/data_engine_py310/bin/python", str(ROOT / "scripts" / "telegram_alerts.py"), "send", msg],
                       cwd=str(ROOT), env={"PYTHONPATH": str(ROOT / "data_engine"), "HOME": str(Path.home()), "PATH": "/usr/bin:/bin"},
                       capture_output=True, timeout=60)
        st["last_alert"] = time.time()
else:
    st["last_alert"] = 0          # reset once it's charging / above threshold
st["last_pct"] = pct
STATE.write_text(json.dumps(st))
