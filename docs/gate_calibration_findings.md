# Gate Calibration Findings (2026-06-06)

Data-driven analysis of why the v83 intraday agent rarely trades, using a
counterfactual backtest over **21 sessions / 12,835 decisions** (2026-04-22 →
2026-06-05). Reconstructs the intraday spot path from each decision's metadata
and labels whether a directional credit spread placed at the OI wall would have
won or lost by session close.

Reproduce with: `python scripts/gate_calibration_backtest.py`

## Headline result

| Policy (max 1 trade/day) | Trades/day | Win rate |
|---|---|---|
| Loosened gates, both directions, 09:30+, ≥60pt wall (what the 2026-06-05 change created) | 0.95 | **30%** |
| **Bear-only, 12:00+ IST, ≥80pt wall** | 0.71 | **80%** |
| Bear-only, 11:00+ IST, ≥100pt wall | 0.71 | 60% |
| Both directions, 11:00+, ≥100pt wall | 0.76 | 56% |

## Three dominant factors (consistent across every cut)

1. **Direction asymmetry** — bearish (bear-call) spreads win 60–80%; bullish
   (bull-put) spreads win only 27–39%. The agent's edge is overwhelmingly on the
   short-call side. Raising the *bullish* trade-score threshold does **not**
   improve bullish win rate (stays ~38–44% at every threshold) — bullish score
   has almost no predictive power.

2. **Time of day** — the first two hours are noise:
   | IST hour | BEAR_CALL win | BULL_PUT win |
   |---|---|---|
   | 09:00 | 19% | 19% |
   | 10:00 | 43% | 20% |
   | 11:00 | 59% | 24% |
   | 12:00 | 74% | 37% |
   | 13:00 | 70% | 38% |

3. **Distance to short strike (wall)** is the #1 structural gate:
   | Distance | Win rate |
   |---|---|
   | 0–40 pt | 19% |
   | 40–80 pt | 33% |
   | 80–120 pt | 46% |
   | 120–200 pt | 61% |

## Changes applied

1. **Reverted** the 2026-06-05 `regime.py` bullish-gate loosening (commit
   901b204). That change lowered bullish entry/confluence/candle thresholds and
   cut the `open_space_up` floor 100→60 — both push toward the losing 30% policy.
   The pandas import fix from that commit is kept (correct, unrelated).
2. **Entry window start 09:30 → 11:00** (`intraday_v83_runtime_config.json`).
   Single biggest lever: removes the ~19–20% win-rate morning trades.

## Caveats

- "Win" is a proxy (short strike never breached + adverse < half the cushion,
  held to close), not realized P&L with fills/slippage. Directional asymmetry and
  the time/distance gradients are robust; exact percentages are not.
- 21 sessions is a small sample. Treat thresholds as a starting point and
  re-run this backtest as more paper sessions accumulate.

## Next candidate (not yet applied — needs more data)

- Bias playbook selection toward bearish Tier-A setups; bull-put spreads only
  midday + far-from-wall. Hold until a larger sample confirms the bull-side
  filter generalises.
