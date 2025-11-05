import os
import math
import datetime
import logging
from datetime import date
from typing import Dict, Any, Optional, Tuple
import pandas as pd

# === Local utilities (same folder) ===
from .event_gate import is_blackout_now
from .iv_store import IVStore
from .market_features import summarize_timeframe, topdown_gate
from .options_features import compute_pcr, symmetry_check, pick_delta_strikes

from modules.data_fetch.option_chain_ingestor import OptionChainIngestor, ChainConfig

# === MarketAdapter (robust import) ===
MarketAdapter = None
try:
    from modules.adapters import dhan_market_adapter as _dma
    MarketAdapter = getattr(_dma, "MarketAdapter", None) or getattr(_dma, "DhanMarketAdapter", None)
except Exception:
    MarketAdapter = None
if MarketAdapter is None:
    try:
        from modules.data_fetch import dhan_market_adapter as _dma
        MarketAdapter = getattr(_dma, "MarketAdapter", None) or getattr(_dma, "DhanMarketAdapter", None)
    except Exception:
        MarketAdapter = None
if MarketAdapter is None:
    raise ImportError("Could not resolve MarketAdapter/DhanMarketAdapter in adapters or data_fetch.")

# === OptionChain (robust import) ===
OptionChainAdapter = None
try:
    from modules.data_fetch import dhan_option_chain as _doc
    for _cand in ("OptionChain", "OptionChainAdapter", "DhanOptionChainAdapter", "DhanOptionChain"):
        if hasattr(_doc, _cand):
            OptionChainAdapter = getattr(_doc, _cand)
            break
except Exception:
    OptionChainAdapter = None
# Wrapper (function-style) fallback
_dhan_wrapper = None
try:
    from modules.data_fetch import dhan_wrapper as _dw
    _dhan_wrapper = _dw
except Exception:
    _dhan_wrapper = None

# Extra fallbacks (other modules in your tree)
_live_chain_mod = None
_repo_chain_mod = None
try:
    from modules.data_fetch import live_option_chain as _lc
    _live_chain_mod = _lc
except Exception:
    pass
try:
    from modules.data_fetch import optionchain_repo as _ocr
    _repo_chain_mod = _ocr
except Exception:
    pass


def _now_ist() -> datetime.datetime:
    return datetime.datetime.now()


class StrategySelector:
    """
    Monthly Short Strangle + Weekly (Tuesday) Wings
    - Adapter-agnostic chain fetch (tries many names; wrapper; repo; live; synth fallback).
    - Entry gates: time, event, regime, PCR, IVP.
    - Strike pick: |Δ| >= min_abs_delta with target 0.18±0.03, symmetry check.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.market = MarketAdapter(config)
        self.chain = OptionChainAdapter(config) if OptionChainAdapter else None

        oc_cfg = config.get("option_chain", {})
        underlying_id = int(oc_cfg.get("underlying_id", config.get("underlying_id", 13)))
        underlying_seg = oc_cfg.get("underlying_seg", config.get("underlying_seg", "IDX_I"))
        try:
            self.chain_ingestor = OptionChainIngestor(ChainConfig(underlying_id, underlying_seg))
        except Exception as exc:
            logging.warning("OptionChainIngestor unavailable: %s", exc)
            self.chain_ingestor = None

        # Resolve IV store path (modules/config by default)
        data_cfg = self.cfg.get("data", {})
        store_path = data_cfg.get("roll_store_path", "iv_store.json")
        if not os.path.isabs(store_path):
            base_cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
            os.makedirs(base_cfg_dir, exist_ok=True)
            store_path = os.path.join(base_cfg_dir, store_path)
        self.iv_store = IVStore(store_path, max_days=int(data_cfg.get("iv_history_days", 60)))

    # ---------------------------------------------------------------------
    #                        Time & Event Gates
    # ---------------------------------------------------------------------
    def within_trading_window(self, now: datetime.datetime) -> bool:
        start = self.cfg["schedules"]["trading_window_start"]
        end   = self.cfg["schedules"]["trading_window_end"]
        t = now.strftime("%H:%M")

        if t < start or t > end:
            return False

        # Skip open (N mins from 09:15)
        open_skip_to = (
            datetime.datetime.strptime("09:15", "%H:%M")
            + datetime.timedelta(minutes=int(self.cfg["entry"]["skip_open_minutes"]))
        ).strftime("%H:%M")
        if t < open_skip_to:
            return False

        # Skip last M mins before 15:25 cutoff
        cutoff = (
            datetime.datetime.strptime("15:25", "%H:%M")
            - datetime.timedelta(minutes=int(self.cfg["entry"]["skip_last_minutes"]))
        ).strftime("%H:%M")
        if t > cutoff:
            return False

        return True

    def event_blackout(self, now: datetime.datetime) -> Optional[dict]:
        minutes_before = int(self.cfg["entry"]["event_blackout_minutes_before"])
        cal_path = self.cfg["data"]["event_calendar_path"]
        if not os.path.isabs(cal_path):
            cal_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", cal_path))
        return is_blackout_now(now, cal_path, minutes_before)

    # ---------------------------------------------------------------------
    #                      Market / Option Features
    # ---------------------------------------------------------------------
    def compute_mtf(self, symbol: str) -> Dict[str, Any]:
        # Multi-timeframe candles (comes from MarketAdapter)
        daily = self.market.get_ohlc(symbol, "1D", 80)
        h1    = self.market.get_ohlc(symbol, "1H", 120)
        m15   = self.market.get_ohlc(symbol, "15m", 240)
        m5    = self.market.get_ohlc(symbol, "5m", 300)

        mtf = {
            "D":   summarize_timeframe(daily, "1D"),
            "H1":  summarize_timeframe(h1, "1H"),
            "M15": summarize_timeframe(m15, "15m"),
            "M5":  summarize_timeframe(m5, "5m"),
        }
        mtf["gate_topdown"] = topdown_gate(mtf["D"], mtf["H1"], mtf["M15"])
        return mtf

    # ---------- Helpers: chain fetch, normalize, ATM detection, synthetic ----------
    def _try_adapter(self, expiry_kind: str, symbol: str) -> Optional[pd.DataFrame]:
        if not self.chain:
            return None
        # Preferred explicit methods
        try:
            if expiry_kind == "monthly" and hasattr(self.chain, "get_monthly_chain"):
                df = self.chain.get_monthly_chain(symbol)
                if isinstance(df, pd.DataFrame) and not df.empty: return df
        except Exception:
            pass
        try:
            if expiry_kind == "weekly" and hasattr(self.chain, "get_weekly_chain"):
                weekday = self.cfg.get("strangle_mode", {}).get("weekly_expiry_day", "Tuesday")
                try:
                    df = self.chain.get_weekly_chain(symbol, weekday=weekday)
                except TypeError:
                    df = self.chain.get_weekly_chain(symbol)
                if isinstance(df, pd.DataFrame) and not df.empty: return df
        except Exception:
            pass
        try:
            if hasattr(self.chain, "get_current_week_chain"):
                df = self.chain.get_current_week_chain(symbol)
                if isinstance(df, pd.DataFrame) and not df.empty: return df
        except Exception:
            pass
        try:
            if hasattr(self.chain, "get_chain"):
                df = self.chain.get_chain(symbol, expiry=expiry_kind)
                if isinstance(df, pd.DataFrame) and not df.empty: return df
        except Exception:
            pass
        return None

    def _try_wrapper(self, expiry_kind: str, symbol: str) -> Optional[pd.DataFrame]:
        if not _dhan_wrapper or not hasattr(_dhan_wrapper, "get_option_chain"):
            return None
        # Try a few signatures/keyword combos
        for kwargs in (
            {"symbol": symbol, "expiry": expiry_kind},
            {"symbol": symbol},
            {"symbol": symbol, "kind": expiry_kind},
        ):
            try:
                df = _dhan_wrapper.get_option_chain(**kwargs)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
                if isinstance(df, dict) and "data" in df:
                    dff = pd.DataFrame(df["data"])
                    if not dff.empty: return dff
                if isinstance(df, list):
                    dff = pd.DataFrame(df)
                    if not dff.empty: return dff
            except Exception:
                pass
        return None

    def _resolve_monthly_expiry(self) -> Optional[str]:
        if not getattr(self, "chain_ingestor", None):
            return None
        try:
            exp_series = self.chain_ingestor.list_expiries()
        except Exception as exc:
            logging.warning("list_expiries failed: %s", exc)
            return None
        future_dates = []
        today = date.today()
        for item in exp_series.dropna().tolist():
            try:
                d = datetime.datetime.strptime(str(item), "%Y-%m-%d").date()
                if d >= today:
                    future_dates.append(d)
            except Exception:
                continue
        if not future_dates:
            return None
        future_dates.sort()
        for d in future_dates:
            if d.month != today.month or d.day >= 20:
                return d.isoformat()
        return future_dates[-1].isoformat()

    def _try_ingestor(self, expiry_kind: str, symbol: str) -> Optional[pd.DataFrame]:
        _ = symbol  # symbol handled via config (underlying_id)
        if not getattr(self, "chain_ingestor", None):
            return None
        tag = expiry_kind
        if expiry_kind.lower() == "weekly":
            tag = "weekly"
        elif expiry_kind.lower() == "monthly":
            resolved = self._resolve_monthly_expiry()
            if resolved:
                tag = resolved
            else:
                return None
        try:
            df = self.chain_ingestor.fetch(tag)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df["expiry_kind"] = expiry_kind
                return df
        except Exception as exc:
            logging.warning("OptionChainIngestor fetch failed for %s: %s", tag, exc)
        return None

    def _try_other_sources(self, expiry_kind: str, symbol: str) -> Optional[pd.DataFrame]:
        # live_option_chain module
        if _live_chain_mod:
            for name in ("get_chain", "get_option_chain", "fetch_chain"):
                if hasattr(_live_chain_mod, name):
                    try:
                        fn = getattr(_live_chain_mod, name)
                        df = fn(symbol, expiry=expiry_kind)
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            return df
                    except Exception:
                        pass
        # optionchain_repo module (class-based)
        if _repo_chain_mod:
            for cls_name in ("OptionChainRepo", "OptionChainRepository", "OptionChainStore"):
                if hasattr(_repo_chain_mod, cls_name):
                    try:
                        repo = getattr(_repo_chain_mod, cls_name)()
                        for m in ("get_chain", "fetch_chain"):
                            if hasattr(repo, m):
                                df = getattr(repo, m)(symbol, expiry=expiry_kind)
                                if isinstance(df, pd.DataFrame) and not df.empty:
                                    return df
                    except Exception:
                        pass
        return None

    def _normalize_chain_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            # OI/Volume
            "OI_CE": "oi_ce", "OI_PE": "oi_pe", "VOL_CE": "vol_ce", "VOL_PE": "vol_pe",
            "oiCE": "oi_ce", "oiPE": "oi_pe", "volCE": "vol_ce", "volPE": "vol_pe",
            # Prices
            "LTP_CE": "ltp_ce", "LTP_PE": "ltp_pe", "ltpCE": "ltp_ce", "ltpPE": "ltp_pe",
            # Greeks
            "deltaCE": "delta_ce", "deltaPE": "delta_pe", "Delta_CE": "delta_ce", "Delta_PE": "delta_pe",
            # IV and strikes
            "iv": "IV", "StrikePrice": "strike", "strikePrice": "strike",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        # Ensure numeric dtypes where expected
        for col in ("strike", "ltp_ce", "ltp_pe", "oi_ce", "oi_pe", "vol_ce", "vol_pe", "IV",
                    "delta_ce", "delta_pe"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Best-effort IV if side IVs present
        if "IV" not in df.columns and {"IV_CE", "IV_PE"}.issubset(df.columns):
            df["IV"] = pd.to_numeric((df["IV_CE"] + df["IV_PE"]) / 2.0, errors="coerce")
        return df

    def _atm_row_from_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Prefer explicit underlying column
        for k in ("underlying_ltp", "underlying_value", "spot"):
            if k in df.columns and pd.notna(df[k]).any():
                spot = float(df[k].dropna().iloc[0])
                break
        else:
            spot = None
            if {"ltp_ce", "ltp_pe", "strike"}.issubset(df.columns):
                tmp = df[["strike", "ltp_ce", "ltp_pe"]].dropna()
                if not tmp.empty:
                    tmp["mis"] = (tmp["ltp_ce"] - tmp["ltp_pe"]).abs()
                    near = tmp.sort_values("mis").iloc[0]
                    spot = float(near["strike"])
            if spot is None and "strike" in df.columns and not df["strike"].isna().all():
                spot = float(df["strike"].median())
            if spot is None:
                spot = float(self.market.get_spot(self.cfg["general"]["symbol"]))
        # Pick nearest strike as ATM row
        if "strike" in df.columns and not df["strike"].isna().all():
            row = df.iloc[(df["strike"] - spot).abs().argsort().iloc[0]].to_dict()
        else:
            row = {}
        row.setdefault("underlying_ltp", spot)
        return row

    def _synthesize_chain(self, symbol: str, expiry_kind: str) -> pd.DataFrame:
        """
        Last-resort synthetic chain so the agent can run end-to-end without a live chain.
        Produces: strike, ltp_ce, ltp_pe, delta_ce, delta_pe, oi_ce, oi_pe, vol_ce, vol_pe, IV, underlying_ltp
        """
        spot = float(self.market.get_spot(symbol))
        # Create ±20 strikes around spot with 50-pt spacing (NIFTY-ish)
        step = 50.0
        n_each = 20
        k0 = round(spot / step) * step
        strikes = [k0 + i * step for i in range(-n_each, n_each + 1)]

        # Simple price/greeks heuristics
        scale = 300.0  # wider => slower decay
        rows = []
        for k in strikes:
            m = (spot - k) / scale  # moneyness scaled
            # Approximate deltas via logistic ≈ norm.cdf
            delta_ce = 1.0 / (1.0 + math.exp(-m * 4.0))
            delta_pe = delta_ce - 1.0  # put-call parity for deltas
            # Symmetric premium decay
            base = max(0.5, 12.0 * math.exp(-abs(spot - k) / scale))
            ltp_ce = base * max(0.05, delta_ce)
            ltp_pe = base * max(0.05, abs(delta_pe))
            rows.append({
                "strike": float(k),
                "ltp_ce": float(ltp_ce),
                "ltp_pe": float(ltp_pe),
                "delta_ce": float(delta_ce),
                "delta_pe": float(delta_pe),
                "oi_ce": 100000.0 * math.exp(-abs(spot - k) / (scale * 1.5)),
                "oi_pe": 100000.0 * math.exp(-abs(spot - k) / (scale * 1.5)),
                "vol_ce": 5000.0,
                "vol_pe": 5000.0,
                "IV": 0.12,
                "underlying_ltp": spot,
                "expiry_kind": expiry_kind,
            })
        df = pd.DataFrame(rows)
        return df

    def _fetch_chain(self, symbol: str, expiry_kind: str) -> pd.DataFrame:
        """
        Try adapter, wrapper, other modules, then synthesize if all fail.
        """
        for fetcher in (self._try_ingestor, self._try_adapter, self._try_wrapper, self._try_other_sources):
            df = fetcher(expiry_kind, symbol)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return self._normalize_chain_columns(df)
        # Fallback: synth chain
        return self._normalize_chain_columns(self._synthesize_chain(symbol, expiry_kind))

    # ---------------------------------------------------------------------
    #                  Chains (monthly shorts + weekly wings)
    # ---------------------------------------------------------------------
    def get_short_and_hedge_chains(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        short_df = self._fetch_chain(symbol, "monthly")
        hedge_df = self._fetch_chain(symbol, "weekly")
        return short_df, hedge_df

    def chain_features(self, symbol: str) -> Dict[str, Any]:
        short_df, hedge_df = self.get_short_and_hedge_chains(symbol)

        # PCR on monthly
        needed_cols = {"oi_ce", "oi_pe", "vol_ce", "vol_pe"}
        if needed_cols.issubset(set(short_df.columns)):
            pcr = compute_pcr(short_df[list(needed_cols)].copy())
        else:
            pcr = {"pcr_oi": 1.0, "pcr_vol": 1.0}  # neutral default if missing

        # ATM row/IV
        atm_row = self._atm_row_from_df(short_df)
        current_iv = float(atm_row.get("IV", 0.0)) if pd.notna(atm_row.get("IV", None)) else 0.0

        # Update IV store & compute percentile
        self.iv_store.add_today_iv(symbol, current_iv)
        ivp = self.iv_store.percentile(symbol, current_iv)

        return {
            "short_df": short_df,
            "hedge_df": hedge_df,
            "pcr": pcr,
            "atm_iv": current_iv,
            "iv_percentile": ivp,
            "atm": atm_row,
        }

    # ---------------------------------------------------------------------
    #                           Entry Gates
    # ---------------------------------------------------------------------
    def entry_gates(self, symbol: str, mtf: Dict[str, Any], ofx: Dict[str, Any]) -> Dict[str, Any]:
        res = {"pass": True, "reasons": []}
        cfg_e = self.cfg["entry"]
        now = _now_ist()

        # Time window
        if not self.within_trading_window(now):
            res["pass"] = False; res["reasons"].append("time_window")

        # Event blackout
        ev = self.event_blackout(now)
        if ev:
            res["pass"] = False; res["reasons"].append(f"event_blackout:{ev.get('title', 'unknown')}")

        # Regime gate (prefer range)
        if not mtf.get("gate_topdown", False):
            res["pass"] = False; res["reasons"].append("topdown_regime")

        # IV / IVP
        ivp_floor = int(cfg_e.get("iv_percentile_floor", 0))
        current_iv = float(ofx["atm_iv"])
        ivp = ofx["iv_percentile"]
        if ivp is None:
            if current_iv < float(cfg_e.get("iv_floor_raw", 0)):
                res["pass"] = False; res["reasons"].append("iv_floor_raw")
        else:
            if ivp < ivp_floor:
                res["pass"] = False; res["reasons"].append("iv_percentile_floor")

        # PCR neutrality band
        pcr_oi = float(ofx["pcr"].get("pcr_oi", 1.0))
        lo, hi = float(cfg_e["pcr_band"]["lo"]), float(cfg_e["pcr_band"]["hi"])
        if not (lo <= pcr_oi <= hi):
            res["pass"] = False; res["reasons"].append("pcr_band")

        return res

    # ---------------------------------------------------------------------
    #                       Strike Proposal (Shorts)
    # ---------------------------------------------------------------------
    def propose_strangle(self, symbol: str, ofx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Picks monthly CE/PE shorts with |Δ| >= min_abs_delta and close to target (0.18 ± 0.03).
        Enforces premium symmetry, forwards the weekly chain as hedge_df.
        """
        df = ofx["short_df"]
        target = float(self.cfg["entry"]["target_delta_abs"])
        tol    = float(self.cfg["entry"]["delta_tolerance"])
        dmin   = float(self.cfg["entry"]["min_abs_delta"])

        # Need Greek columns (can be synthetic if we used the synth-chain)
        if not {"delta_ce", "delta_pe"}.issubset(df.columns):
            return None

        picked = pick_delta_strikes(df, target_abs_delta=target, tol=tol, min_abs_delta=dmin)
        if not picked:
            return None
        ce, pe = picked

        # Safety floor
        if abs(float(ce["delta_ce"])) < dmin or abs(float(pe["delta_pe"])) < dmin:
            return None

        # Premium symmetry
        if not symmetry_check(float(ce["ltp_ce"]), float(pe["ltp_pe"]),
                              float(self.cfg["entry"]["symmetry_max_misprice_pct"])):
            return None

        return {
            "CE": ce,
            "PE": pe,
            "net_credit": float(ce["ltp_ce"]) + float(pe["ltp_pe"]),
            "hedge_df": ofx.get("hedge_df"),
        }
