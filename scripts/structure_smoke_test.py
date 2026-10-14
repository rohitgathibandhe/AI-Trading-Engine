#!/usr/bin/env python3
"""STRUCTURE SMOKE TEST — can the agent actually BUILD every structure on today's live chain?

The sell side was dead for days (2026-10-12..14): the chooser picked a bear-call 150+ times a day and the
builder rejected every one (credit floor vs delta bands), silently — reported as "no trade". This runs at
09:25 on weekdays, builds each structure the agent can choose with the agent's own code on the live
snapshot, and Telegrams if any can't be built (with the rejection reason), so a dead path is caught on
day one instead of discovered in a retro.

  PYTHONPATH=data_engine python scripts/structure_smoke_test.py [--no-telegram]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_engine"))
sys.path.insert(0, str(ROOT / "scripts"))
STATE = ROOT / "data_engine" / "market_ai" / "state"

# (structure, metadata overrides the live chooser would set for it)
CASES = [
    ("PUT_DEBIT_SPREAD", {}),
    ("CALL_DEBIT_SPREAD", {}),
    ("BULL_PUT_CREDIT_SPREAD", {"seller_exit": True, "conf_bull_sell_closer": True}),
    ("BEAR_CALL_CREDIT_SPREAD", {"seller_exit": True}),
]


def run() -> list[dict]:
    from market_ai.intraday_defined_risk.monitor import load_object
    from market_ai.intraday_defined_risk.regime import classify_regime
    from market_ai.intraday_defined_risk import strikes as K
    from market_ai.intraday_defined_risk.data_models import AdaptiveParameters, StrategyType

    cfg = json.loads((STATE / "intraday_v83_run_live_config.json").read_text())
    snap = load_object(cfg["provider_class"])(cfg).current_snapshot()
    out = []
    for name, overrides in CASES:
        regime = classify_regime(snap)
        regime.metadata.update(overrides)
        res = K.select_structure(StrategyType[name], snap, regime, AdaptiveParameters())
        st, reasons = res[0], (res[1] if len(res) > 1 else [])
        out.append({"structure": name, "ok": st is not None,
                    "detail": (f"{[(l.action, l.strike) for l in st.legs]} credit {st.credit_points:+.1f} w{st.width_points:.0f}"
                               if st is not None else "; ".join(map(str, reasons))[:200]),
                    "spot": snap.option_chain.spot, "expiry": str(snap.option_chain.expiry)})
    return out


def main() -> int:
    try:
        res = run()
    except Exception as exc:  # noqa: BLE001
        res = [{"structure": "ALL", "ok": False, "detail": f"smoke test crashed: {type(exc).__name__}: {exc}"}]
    for r in res:
        print(f"[{'OK ' if r['ok'] else 'FAIL'}] {r['structure']:26} {r['detail']}")
    (STATE / "structure_smoke_test.json").write_text(json.dumps(res, indent=2, default=str))
    failed = [r for r in res if not r["ok"]]
    if failed and "--no-telegram" not in sys.argv:
        try:
            import telegram_alerts as ta
            ta.cmd_send("🧪 <b>Structure smoke test — paths that CANNOT be built today</b>\n" + "\n".join(
                f"• {r['structure']}: {r['detail']}" for r in failed))
        except Exception as exc:  # noqa: BLE001
            print(f"telegram failed: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
