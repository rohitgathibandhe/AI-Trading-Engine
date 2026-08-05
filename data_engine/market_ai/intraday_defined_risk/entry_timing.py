"""ENTRY TIMING — wait for a better PRICE, not a better clock time.

The agent's directional entries were consistently made at the worst price of the move. Measured on
every debit trade with live captures, entry landed at 0-14% of the day's range so far — i.e. it
bought puts AT THE LOW, right after the fall it was reacting to:

    date      entry   spot     range so far    position   better price later
    07-15     13:00  24,052   24,049-24,215        2%        +38 pts
    07-16     09:35  24,128   24,128-24,146        0%        +55 pts
    07-21     11:35  24,177   24,167-24,253       12%        +14 pts
    07-22     10:32  24,002   23,977-24,150       14%        +46 pts
    07-23     09:48  23,907   23,905-23,949        6%        +82 pts

In EVERY case a materially better entry existed later in the session. This is the "signal fires ->
buy immediately" reflex, and it costs the trade its edge before it starts: on 2026-07-23 the agent
paid 69.00 for the spread at 09:49 when 58.90 was available at 10:04.

THE RULE (price action, not a fixed time): once a directional signal fires, do not chase. Wait for
price to RETRACE AGAINST the trade — for a bearish trade, wait for a bounce — and enter into that
retrace at a better price. Priced on the two days whose reconstruction validates against actual
realized P&L:

    07-23  signal 09:49 debit 69.00 -> +588   |  wait +20pt 09:59 debit 63.62 -> +938 (+349, 11 min)
                                              |  wait +30pt 10:04 debit 58.90 -> +1,245 (+657, 16 min)
    07-22  signal 10:32 debit 70.90 ->  -96   |  wait +20pt 10:57 debit 66.97 -> +159 (+255, 25 min)
                                              |  wait +30pt 11:22 debit 65.20 -> +275 (+371, 50 min)

CRITICAL — the TIMEOUT is not a detail, it is what makes this safe. On a genuine trend day price
never retraces; it just goes. A pure "wait for a pullback" rule would sit out exactly the runaway
days the debit exists to catch (2026-07-16 never gave back 20pts for 77 minutes and still made
+1,638). So if the pullback does not arrive within the window, we enter anyway at market. The rule
can only ever IMPROVE the fill or leave it unchanged — it can never cause a missed trade.

Applies to DIRECTIONAL DEBIT entries only. A premium SELLER wants the opposite (sell into the spike,
richer premium), and a neutral fly wants spot near the ATM, so both are left alone here until there
is forward evidence for their own entry rule.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

_STATE = Path(__file__).resolve().parent.parent / "state" / "entry_pending.json"

# Retrace required before entering, as a FRACTION OF SPOT so it scales with the index level and
# volatility regime rather than being a hardcoded point count. 0.0009 ~= 21pts at Nifty 24,000,
# the level validated above.
_RETRACE_PCT = float(os.environ.get("ENTRY_RETRACE_PCT", "0.0009") or 0.0009)
# How long to wait before giving up and taking the signal price. 45 min covers both validated
# pullbacks (11 and 25 min) with room, and bounds the theta paid while waiting.
_TIMEOUT_MIN = float(os.environ.get("ENTRY_WAIT_TIMEOUT_MIN", "45") or 45)
_ENABLED = os.environ.get("ENTRY_WAIT_ENABLED", "1") == "1"

_BEARISH = {"PUT_DEBIT_SPREAD"}
_BULLISH = {"CALL_DEBIT_SPREAD"}

# Credit SELLERS reuse the SAME timing mechanic (wait for spot to move a set amount before entering)
# for a different REASON. A debit buyer waits for a retrace to get a cheaper fill; a credit seller
# waits for the thesis to CONFIRM so it doesn't sell into a local extreme that instantly reverses.
# 2026-07-30: a bull-put entered at 24,277 the instant its up-signal fired — right before a dip to
# VWAP that stopped it out (-331), then the tape recovered and closed higher; the shadow book shows
# the same structure made +401. Waiting for confirmation would have entered AFTER that dip.
# The wait DIRECTION coincides with the debits: a BULL_PUT (bullish) waits for spot UP exactly like a
# PUT_DEBIT; a BEAR_CALL (bearish) waits for spot DOWN exactly like a CALL_DEBIT. Only the rationale
# differs. n=1 so far — SELLER_ENTRY_WAIT_ENABLED=0 disables independently while the forward record
# (selection_log + exit_shadow) judges it.
_SELLER_ENABLED = os.environ.get("SELLER_ENTRY_WAIT_ENABLED", "1") == "1"
_SELL_WAIT_UP = {"BULL_PUT_CREDIT_SPREAD"}      # bullish seller — confirm up before selling puts
_SELL_WAIT_DOWN = {"BEAR_CALL_CREDIT_SPREAD"}   # bearish seller — confirm down before selling calls


def _read() -> dict:
    try:
        return json.loads(_STATE.read_text())
    except (OSError, ValueError):
        return {}


def _write(d: dict) -> None:
    try:
        _STATE.write_text(json.dumps(d))
    except OSError:
        pass


def clear() -> None:
    try:
        _STATE.unlink()
    except OSError:
        pass


def should_wait(strategy_name: str, spot: float, now: datetime) -> tuple[bool, str]:
    """Called when a directional-debit signal is ready to fire.

    Returns (wait, reason). wait=True means HOLD this cycle — the signal stays live and we re-check
    next cycle. wait=False means enter now, either because the retrace arrived (better price) or
    because the window expired (don't miss a trend).

    Fails OPEN: on any unexpected condition it returns (False, ...) so a bug here can never stop the
    agent trading. That failure mode — silent machinery discarding a good decision — has already
    cost this project twice.
    """
    try:
        # math.isfinite first: a NaN spot passes `<= 0` (every NaN comparison is False), so it would
        # arm and then wait out the whole window without ever "reaching" the target — a fail-CLOSED
        # path in a module whose entire contract is to fail open.
        if not _ENABLED or not math.isfinite(spot) or spot <= 0:
            return False, "ENTRY_WAIT_DISABLED"
        is_seller = strategy_name in _SELL_WAIT_UP or strategy_name in _SELL_WAIT_DOWN
        if is_seller and not _SELLER_ENABLED:
            return False, "SELLER_ENTRY_WAIT_DISABLED"
        direction = (
            1 if (strategy_name in _BEARISH or strategy_name in _SELL_WAIT_UP)
            else -1 if (strategy_name in _BULLISH or strategy_name in _SELL_WAIT_DOWN)
            else 0
        )
        if direction == 0:
            return False, "NOT_TIMED_ENTRY"      # neutral sellers (fly/condor/strangle/straddle): unchanged

        st = _read()
        key = f"{now.date().isoformat()}|{strategy_name}"
        if st.get("key") != key:
            st = {"key": key, "armed_at": now.isoformat(timespec="seconds"), "signal_spot": round(spot, 2)}
            _write(st)
            _verb = "confirm" if is_seller else "retrace"
            return True, (f"ENTRY_WAIT_ARMED: signal at {spot:,.0f}; want spot "
                          f"{'+' if direction > 0 else '-'}{spot * _RETRACE_PCT:.0f}pts to {_verb} before entry")

        s0 = float(st.get("signal_spot") or spot)
        armed = datetime.fromisoformat(st["armed_at"])
        waited = (now - armed).total_seconds() / 60.0
        target = s0 * (1.0 + direction * _RETRACE_PCT)
        reached = (spot >= target) if direction > 0 else (spot <= target)

        if reached:
            clear()
            _why = "thesis confirmed" if is_seller else "better price"
            return False, (f"ENTRY_WAIT_FILLED: spot {spot:,.0f} reached {target:,.0f} "
                           f"after {waited:.0f}m — entering ({_why})")
        if waited >= _TIMEOUT_MIN:
            clear()
            return False, (f"ENTRY_WAIT_TIMEOUT: no {abs(target - s0):.0f}pt retrace in "
                           f"{waited:.0f}m — trend is not giving one back, entering at market")
        return True, (f"ENTRY_WAIT: spot {spot:,.0f}, need {target:,.0f} "
                      f"({waited:.0f}/{_TIMEOUT_MIN:.0f}m elapsed)")
    except Exception:
        return False, "ENTRY_WAIT_ERROR_FAIL_OPEN"
