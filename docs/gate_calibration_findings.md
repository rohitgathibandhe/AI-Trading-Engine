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

## Real-premium validation (no proxy)

`scripts/real_premium_backtest.py` re-runs the analysis using the archived option
chain (`state/rolling_option/<date>/`, per-minute strike-level LTP + Greeks),
constructing the actual credit spread at the wall, pricing it with REAL premiums,
and simulating TP(50%)/SL(1x)/square-off exits. Overlap: 7 sessions / 12 trades
(chain archive ends 2026-05-15).

| Direction | Trades | Win % | Net P&L | Expectancy |
|---|---|---|---|---|
| BEAR_CALL | 6 | **83%** | **+₹2,243** | **+₹374/trade** |
| BULL_PUT  | 6 | **17%** | **−₹1,069** | **−₹178/trade** |

The directional edge holds with **real money**: bearish credit spreads are
profitable, bullish ones lose. This is the strongest, most actionable finding and
it agrees with the proxy.

## Caveats

- The proxy "win" (short strike never breached + adverse < half cushion) is
  directionally right but not P&L. The real-premium run above is P&L but only 12
  trades — the **direction** signal is strong; time-of-day / distance cuts need a
  bigger sample (most real-premium trades landed at 09:00 here, so that run can't
  confirm the midday edge — the larger proxy sample does).
- 21 proxy sessions / 7 real-premium sessions is small. Re-run both as forward
  paper trading (real premiums via PaperOnlyExecutor) accumulates outcomes.

## Bias detector is anti-predictive — and the fix (2026-06-08)

The agent's raw directional bias is worse than a coin flip at the intraday→close
horizon: **BULLISH bias is correct only 36%** of the time, BEARISH 44% (n=10,086).
So bull-put losses are not a strategy flaw — the agent deploys bullish into
markets that then fall. Disabling bull-put outright was rejected; instead we found
*what conditionally makes a bullish read win* via within-bucket feature
separation.

**`option_chain_pressure_state` is the decisive separator for bull-put:**

| Chain pressure state | Bull-put win | n |
|---|---|---|
| NEUTRAL | 69% | 972 |
| OVERHEAD_CALL_PRESSURE | 54% | 971 |
| DOWNSIDE_PUT_SUPPORT | 27% | 651 |
| BALANCED_WALLS | 15% | 1455 |

Gating bullish entry to NEUTRAL/OVERHEAD_CALL_PRESSURE lifts win rate **39% → 62%**
(n=1943); the excluded states win 18%.

**Applied:** `regime.py` now vetoes `bullish_entry_ready` when
`option_chain_pressure_state` is `BALANCED_WALLS` or `DOWNSIDE_PUT_SUPPORT`.
Bearish is left broad (already 80%); best when `ema_alignment != BULLISH` (82%).

## Next candidate (not yet applied — needs more data)

- Re-run this within-bucket separation as forward sessions accumulate; the
  bullish veto is the highest-impact lever so far and should be re-validated on
  real-premium outcomes once the agent has traded under it.
