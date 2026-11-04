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

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .data_models import AccountRiskLimits, AccountState, MarketSnapshot, OhlcvSeries
from .live_runtime import _aged_previous_chain, _vwap_from_bars

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
        # prior-session daily bars, like DhanLiveMarketDataProvider (90 calendar days before today)
        from .data_models import OhlcvBar as _Bar
        self._daily = {d: _Bar(timestamp=datetime.combine(d, datetime.min.time()), open=bs[0].open,
                               high=max(b.high for b in bs), low=min(b.low for b in bs), close=bs[-1].close,
                               volume=0.0) for d, bs in sorted(self._bars5.items()) if len(bs) >= 60}
        # recorded VIX / BankNifty 5m series (scripts/live_path_replay.py fetches them from Dhan)
        self._mseries = {}
        _mp = config.get("replay_market_series_path")
        if _mp and Path(str(_mp)).exists():
            try:
                _raw = json.loads(Path(str(_mp)).read_text())
                for _n in ("vix", "bank"):
                    self._mseries[_n] = sorted((datetime.fromisoformat(t), float(c)) for t, c in (_raw.get(_n) or []))
            except Exception:  # noqa: BLE001
                self._mseries = {}
        _dp = config.get("replay_daily_path")
        if _dp and Path(str(_dp)).exists():
            from .live_runtime import _bars_from_candles
            _real = _bars_from_candles(json.loads(Path(str(_dp)).read_text()) or [])
            if _real:          # live parity: the official Dhan daily candles, not ones rebuilt from 5m bars
                from ..utils.market_calendar import special_sessions as _ss
                self._daily = {b.timestamp.date(): b for b in _real if b.timestamp.date() not in _ss()}
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
        self._chain_hist = []
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
        eod = datetime.combine(day, datetime.min.time()).replace(hour=15, minute=16)
        if nxt.replace(tzinfo=None) > day_last + STEP and day_last < eod and CLOCK.now.replace(tzinfo=None) < eod:
            # the day's recording ended early (25 of 66 recorded days stop before 15:15): live's clock keeps running
            # and its 15:15 time exit / EOD guardian closes on the last mark. One final cycle at 15:16 on the last
            # recorded snapshot does the same — before this, open positions rode overnight ("STALE_OVERNIGHT_EXIT").
            CLOCK.now = eod.replace(tzinfo=IST)
            return
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
        prior_daily = [b for d, b in self._daily.items() if day - timedelta(days=90) <= d < day]
        chain = self._chains[ts]
        chain = type(chain)(timestamp=stamp, expiry=chain.expiry, spot=chain.spot, quotes=chain.quotes,
                            margin_estimate_per_lot=chain.margin_estimate_per_lot)
        snap = MarketSnapshot(
            nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=closed5),
            nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=closed15),
            nifty_daily=OhlcvSeries(timeframe_minutes=1440, bars=prior_daily) if prior_daily else None,
            option_chain=chain, risk_limits=self._risk,
            account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0),
            live_vwap=_vwap_from_bars(closed5), lot_size=self._lot_size, slippage_points=self._slippage,
            previous_option_chain=_aged_previous_chain(self._chain_hist, chain, self._prev_chain),
            previous_session_close=self._prev_close.get(day))
        self._prev_chain = chain
        if not self._chain_hist or self._chain_hist[-1].timestamp != chain.timestamp:
            self._chain_hist = (self._chain_hist + [chain])[-40:]
        self._write_market_context(chain, stamp)
        self._archive_oi(chain, stamp)
        return snap

    def _series_at(self, name, stamp, lag_min=0):
        """Last closed 5m close of a recorded series (VIX / BankNifty) at or before stamp - lag."""
        ser = getattr(self, "_mseries", {}).get(name) or []
        cut = stamp - timedelta(minutes=lag_min)
        val = None
        for t, c in ser:                      # sorted
            if t + timedelta(minutes=5) <= cut:
                val = c
            else:
                break
        return val

    def _archive_oi(self, chain, stamp) -> None:
        """Mirror live: oi_today.json each cycle; at a new session the previous session's last OI is archived to
        oi_snapshots.json (the watchdog does this at 15:35 live) — the replay used a frozen pre-start archive and
        never had today's OI (2026-11-03 audit)."""
        try:
            from .live_runtime import STATE_ROOT
            day = stamp.date().isoformat()
            last = getattr(self, "_oi_last", None)
            if last and last["date"] != day:
                arc_p = STATE_ROOT / "oi_snapshots.json"
                arc = json.loads(arc_p.read_text()) if arc_p.exists() else []
                arc = [a for a in arc if str(a.get("date")) < last["date"]] + [last]
                arc_p.write_text(json.dumps(arc[-10:]))
            by = {}
            for q in chain.quotes:
                if q.oi:
                    k = str(int(q.strike))
                    by.setdefault(k, {"ce_oi": 0, "pe_oi": 0})
                    by[k]["ce_oi" if str(getattr(q.option_type, "value", q.option_type)).upper().startswith("C") else "pe_oi"] += int(q.oi)
            snap = {"date": day, "spot": chain.spot, "expiry": str(chain.expiry), "strikes": by}
            (STATE_ROOT / "oi_today.json").write_text(json.dumps(snap))
            self._oi_last = snap
        except Exception:  # noqa: BLE001
            pass

    def _write_market_context(self, chain, stamp) -> None:
        """Publish market_context.json each cycle like DhanLiveMarketDataProvider (VIX via the same ATM-IV
        proxy live falls back to). Lives in the replay worktree's own state folder."""
        try:
            import json
            from .live_runtime import STATE_ROOT
            ivs = [float(q.iv) for q in chain.quotes if q.iv and q.ltp > 0 and abs(q.strike - chain.spot) <= 200.0]
            ctx = {"nifty_spot": chain.spot, "updated_at": stamp.isoformat(),
                   "india_vix": round(sum(ivs) / len(ivs), 2) if ivs else None, "india_vix_source": "chain_iv_proxy"}
            _vix = self._series_at("vix", stamp)
            if _vix and 5.0 <= _vix <= 100.0:            # live uses the real India VIX; so does the replay now
                ctx["india_vix"], ctx["india_vix_source"] = round(_vix, 2), "dhan_api"
            _bk, _bk5 = self._series_at("bank", stamp), self._series_at("bank", stamp, lag_min=5)
            if _bk:
                ctx["banknifty_spot"] = _bk
                ctx["banknifty_spot_prev"] = _bk5 or _bk
            _sp = [c for c in getattr(self, "_spot_hist", []) if c[0] <= stamp - timedelta(minutes=5)]
            if _sp:
                ctx["nifty_spot_prev"] = _sp[-1][1]
            self._spot_hist = (getattr(self, "_spot_hist", []) + [(stamp, chain.spot)])[-30:]
            (STATE_ROOT / "market_context.json").write_text(json.dumps(ctx))
        except Exception:
            pass
