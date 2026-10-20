"""LIVE-PATH REPLAY — run the REAL `run_live` loop on recorded market data.

The backtest harness (`backtest.run_backtest`) is NOT the live path: it skips evaluate_entry_gate, health,
manage_paper_position (live exits), record_paper_entry, entry-timing (off by default), and it hands the
agent the 5m bar that STARTS at the snapshot time already complete (up to 5 min of look-ahead). Its P&L
therefore overstated what the agent actually does (owner, 2026-10-14: "your replay shows profits but the
agent's actual code doesn't").

This provider plugs into `run_live` exactly like DhanLiveMarketDataProvider, but serves snapshots from a
dataset built by scripts/build_live_dataset.py (real live chains, 5-min cadence) and drives a replay clock:
- each snapshot is stamped at bucket+SNAPSHOT_OFFSET (when the chain was captured) and carries only 5m
  bars that had CLOSED by then (no look-ahead);
- `install_clock()` patches `datetime.now`/`sleep` in the modules the live loop reads them from, and each
  `sleep()` advances to the next recorded snapshot; the end of data raises ReplayFinished.
Everything else — selector, chooser, gates, entry wait, paper entry/exit, ledgers — is the production code.
Run it only from an isolated copy of the repo (scripts/live_path_replay.py does that) so it can never touch
the live state folder.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .data_models import AccountRiskLimits, AccountState, MarketSnapshot, OhlcvSeries
from .live_runtime import _vwap_from_bars

IST = timezone(timedelta(hours=5, minutes=30))
SNAPSHOT_OFFSET = timedelta(minutes=4)
STEP = timedelta(seconds=60)


class ReplayFinished(Exception):
    pass


class _Clock:
    now: datetime = datetime(2000, 1, 1, tzinfo=IST)


CLOCK = _Clock()


class _ReplayDateTime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D401
        n = CLOCK.now
        return n.astimezone(tz) if tz is not None else n.replace(tzinfo=None)

    @classmethod
    def today(cls):
        return CLOCK.now.replace(tzinfo=None)


class _ReplayDate(date):
    @classmethod
    def today(cls):
        return CLOCK.now.date()


_PROVIDER: "ReplayMarketDataProvider | None" = None


def _replay_sleep(_seconds: float = 0) -> None:
    if _PROVIDER is not None:
        _PROVIDER.advance()


def install_clock() -> None:
    """Point the live loop's clock + sleep at the replay. Call before run_live()."""
    from . import day_thesis, data_pipeline_health, monitor, ops_runtime
    for mod in (monitor, ops_runtime, data_pipeline_health, day_thesis):
        if hasattr(mod, "datetime"):
            mod.datetime = _ReplayDateTime
        if hasattr(mod, "date"):
            mod.date = _ReplayDate
    monitor.sleep = _replay_sleep


class ReplayMarketDataProvider:
    def __new__(cls, config: dict[str, Any]):
        # after a simulated crash run_live builds a NEW provider; like live data after a launchd restart, the
        # replay must continue from where it was, not rewind
        if config.get("_resume") and _PROVIDER is not None:
            return _PROVIDER
        return super().__new__(cls)

    def __init__(self, config: dict[str, Any]):
        global _PROVIDER
        if config.get("_resume") and _PROVIDER is self:
            return
        from .backtest import load_backtest_dataset, _group_bars_by_day, _session_previous_closes
        ds = load_backtest_dataset(config["replay_data_path"])
        self.config = config
        start = date.fromisoformat(str(config["replay_start"]))
        end = date.fromisoformat(str(config["replay_end"]))
        self._chains = {ts: ch for ts, ch in ds["chain_map"].items() if start <= ts.date() <= end}
        self._times = sorted(self._chains)
        self._bars5 = _group_bars_by_day(ds["bars_5m"])
        self._bars15 = _group_bars_by_day(ds["bars_15m"])
        self._prev_close = _session_previous_closes(ds["bars_5m"])
        # identical to DhanLiveMarketDataProvider.current_snapshot (same run_live config keys)
        self._risk = AccountRiskLimits(
            max_risk_rupees_per_trade=float(config.get("max_risk_rupees_per_trade") or 10_000.0),
            max_margin_rupees=float(config.get("max_margin_rupees") or 200_000.0),
            max_daily_loss_rupees=float(config.get("max_daily_loss_rupees") or 10_000.0),
            min_lots_per_trade=int(config.get("min_lots_per_trade") or 1))
        self._lot_size = int(config.get("lot_size") or 65)
        self._slippage = float(config.get("slippage_points") or 0.0)
        self._i = -1
        self._prev_chain = None
        _PROVIDER = self
        self.advance()

    def advance(self) -> None:
        """One live-loop cycle = STEP of replay time (live polls ~30s; the loop's wake-guard treats gaps
        > 180s as a sleep, so the replay must tick at live-like cadence). The snapshot served is the latest
        RECORDED one at that time; at a day's end the clock jumps to the next day's first recording."""
        if self._i < 0:
            self._i = 0
            CLOCK.now = (self._times[0] + SNAPSHOT_OFFSET).replace(tzinfo=IST)
            return
        nxt = CLOCK.now + STEP
        day = CLOCK.now.date()
        day_last = max(t for t in self._times if t.date() == day) + SNAPSHOT_OFFSET
        if nxt.replace(tzinfo=None) > day_last + STEP:
            later = [t for t in self._times if t.date() > day]
            if not later:
                raise ReplayFinished()
            nxt = (later[0] + SNAPSHOT_OFFSET).replace(tzinfo=IST)
        CLOCK.now = nxt
        now = CLOCK.now.replace(tzinfo=None)
        while self._i + 1 < len(self._times) and self._times[self._i + 1] + SNAPSHOT_OFFSET <= now:
            self._i += 1

    def current_snapshot(self) -> MarketSnapshot:
        ts = self._times[self._i]
        stamp = CLOCK.now.replace(tzinfo=None)          # candles close on the replay clock, not the bucket
        day = ts.date()
        closed5 = [b for b in self._bars5.get(day, []) if b.timestamp + timedelta(minutes=5) <= stamp]
        closed15 = [b for b in self._bars15.get(day, []) if b.timestamp + timedelta(minutes=15) <= stamp]
        if len(closed5) < 3 or not closed15:
            from .live_runtime import LiveDataReadinessError  # same failure the live provider raises
            raise LiveDataReadinessError("MIN_5M_CANDLES_NOT_READY", "replay: not enough closed candles yet",
                                         diagnostics={"replay": True})
        chain = self._chains[ts]
        chain = type(chain)(timestamp=stamp, expiry=chain.expiry, spot=chain.spot, quotes=chain.quotes,
                            margin_estimate_per_lot=chain.margin_estimate_per_lot)
        snap = MarketSnapshot(
            nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=closed5),
            nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=closed15),
            option_chain=chain, risk_limits=self._risk,
            account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0),
            live_vwap=_vwap_from_bars(closed5), lot_size=self._lot_size, slippage_points=self._slippage,
            previous_option_chain=self._prev_chain, previous_session_close=self._prev_close.get(day))
        self._prev_chain = chain
        self._write_market_context(chain, stamp)
        return snap

    def _write_market_context(self, chain, stamp) -> None:
        """Publish market_context.json each cycle like DhanLiveMarketDataProvider (VIX via the same ATM-IV
        proxy live falls back to). Lives in the replay worktree's own state folder."""
        import os
        if os.environ.get("REPLAY_HARNESS") != "1":      # never write a live-looking market_context outside the harness
            return
        try:
            import json
            from .live_runtime import STATE_ROOT
            ivs = [float(q.iv) for q in chain.quotes if q.iv and q.ltp > 0 and abs(q.strike - chain.spot) <= 200.0]
            ctx = {"nifty_spot": chain.spot, "updated_at": stamp.isoformat(),
                   "india_vix": round(sum(ivs) / len(ivs), 2) if ivs else None, "india_vix_source": "chain_iv_proxy"}
            (STATE_ROOT / "market_context.json").write_text(json.dumps(ctx))
        except Exception:
            pass
