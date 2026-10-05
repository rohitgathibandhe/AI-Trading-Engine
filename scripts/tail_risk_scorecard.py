#!/usr/bin/env python
"""TAIL-RISK SCORECARD — judge the short-premium book by the numbers that matter.

A short-premium book (what this agent now is) has an ASYMMETRIC payoff: many small wins, occasional
very large loss. Sharpe and win-rate are actively MISLEADING for it — they look beautiful right up
until the tail shows up, because the loss hasn't happened in the sample yet. Per the Nifty
option-selling architecture, the real metrics are: worst single day, worst week, max drawdown, and
CVaR at the tail — plus the STRESS worst-day (what a gap through a wing costs), which for a
defined-risk book is bounded and knowable even though it hasn't occurred.

Reads the actual agent (paper_trades) and every shadow structure (shadow_book), builds each one's
daily P&L series, and reports the tail metrics side by side. Run after the close:

    python scripts/tail_risk_scorecard.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _agent_daily() -> dict[str, float]:
    """Actual agent realized P&L per session (paper)."""
    daily: dict[str, float] = defaultdict(float)
    for r in _jsonl(STATE / "intraday_v83_paper_live_trades.jsonl"):
        if r.get("event") == "PAPER_EXIT" and r.get("realized_paper_pnl") is not None:
            daily[str(r.get("session_date"))] += float(r["realized_paper_pnl"])
    return dict(daily)


def _shadow_daily() -> dict[str, dict[str, float]]:
    """Per-structure daily P&L from the shadow book: {structure: {date: pnl}}."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in _jsonl(STATE / "shadow_book.jsonl"):
        d = str(r.get("date"))
        for k, v in (r.get("structures") or {}).items():
            if isinstance(v, dict) and v.get("pnl_rupees") is not None:
                out[k][d] = float(v["pnl_rupees"])
    return out


def _max_drawdown(series: list[float]) -> float:
    """Largest peak-to-trough drop of the cumulative curve (<= 0)."""
    peak = cum = 0.0
    mdd = 0.0
    for x in series:
        cum += x
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return mdd


def _worst_window(pnls: list[float], w: int) -> float:
    if len(pnls) < w:
        return sum(pnls)
    return min(sum(pnls[i:i + w]) for i in range(len(pnls) - w + 1))


def _cvar(pnls: list[float], frac: float = 0.20) -> float:
    """Mean of the worst `frac` of days (a practical CVaR proxy at small n)."""
    k = max(1, round(len(pnls) * frac))
    return sum(sorted(pnls)[:k]) / k


def _row(name: str, by_date: dict[str, float]) -> dict | None:
    if not by_date:
        return None
    pnls = [by_date[d] for d in sorted(by_date)]
    n = len(pnls)
    wins = sum(1 for x in pnls if x > 0)
    return {
        "name": name, "n": n, "total": sum(pnls), "mean": sum(pnls) / n,
        "win_rate": 100.0 * wins / n,
        "worst_day": min(pnls), "worst_week": _worst_window(pnls, 5),
        "max_dd": _max_drawdown(pnls), "cvar20": _cvar(pnls, 0.20),
    }


def main() -> int:
    rows = []
    ag = _row("AGENT (actual paper)", _agent_daily())
    if ag:
        rows.append(ag)
    for struct, by_date in _shadow_daily().items():
        r = _row(struct, by_date)
        if r:
            rows.append(r)

    if not rows:
        print("No P&L history yet.")
        return 0

    print("TAIL-RISK SCORECARD — short-premium book judged on the tail, not Sharpe/win-rate\n")
    print(f"  {'book':<22}{'days':>5}{'total':>9}{'mean/d':>8}{'win%':>6}  |{'WORST DAY':>11}{'WORST WK':>10}{'MAX DD':>9}{'CVaR20':>9}")
    print("  " + "-" * 96)
    # sort by worst_day ascending — the most tail-exposed at the bottom is what we care about
    for r in sorted(rows, key=lambda x: x["worst_day"]):
        star = "  <- agent" if r["name"].startswith("AGENT") else ""
        print(f"  {r['name']:<22}{r['n']:>5}{r['total']:>+9,.0f}{r['mean']:>+8,.0f}{r['win_rate']:>5.0f}%  |"
              f"{r['worst_day']:>+11,.0f}{r['worst_week']:>+10,.0f}{r['max_dd']:>+9,.0f}{r['cvar20']:>+9,.0f}{star}")

    print("\n  READ THE RIGHT COLUMNS: worst-day / worst-week / max-drawdown / CVaR — NOT total or win%.")
    n_max = max(r["n"] for r in rows)
    print(f"\n  HONEST CAVEAT — the sample is {n_max} days and contains NO crisis day.")
    print("  A short-premium book's real risk is a gap through a wing, which has NOT occurred here.")
    print("  So every 'worst' above is a FLOOR on the true risk, not the true risk. The number that")
    print("  IS knowable is the STRESS worst-day = the structure's defined max loss x lots on a gap")
    print("  that blows through a short strike — bounded ONLY because the book is now defined-risk.")
    print("  Judge readiness on that bounded worst case, and do not trust win% until a gap is in-sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
