# -*- coding: utf-8 -*-
"""
Algo Agent Dashboard (Streamlit) — stable, end-to-end working build.

What this file does:
- Sidebar: save/verify credentials, start/stop agent, choose paper/live, set refresh.
- Trade tab: funds cards + NIFTY spot LTP (polls in background and shows latest).
- Positions tab:
    • Open positions table (SHORT → sellQty/sellAvg, LONG → buyQty/buyAvg; LTP=costPrice; PnL=unrealizedProfit)
    • Today’s closed trades table (Qty, Buy Avg, Sell Avg, LTP=costPrice, PnL=realizedProfit) + Overall row
- Agent Logs tab: shows last ~800 log lines.
- Settings tab: read/write JSON config.
"""

from __future__ import annotations

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
from queue import Queue, Empty


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from math import erf

try:  # Streamlit 1.18+
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # fallback for older builds
    st_autorefresh = None


# =============================================================================
# Auto-refresh helper
# =============================================================================
def _auto_refresh() -> None:
    """Trigger periodic reruns so LTP and positions update without manual refresh."""
    try:
        refresh_sec = max(1, int(st.session_state.get("refresh_sec", 5)))
    except Exception:
        refresh_sec = 5

    if st_autorefresh is not None:
        st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh_tick")
    else:
        # fallback: rely on browser refresh every ~refresh_sec seconds via hidden script
        placeholder = st.empty()
        placeholder.markdown(
            f"""
            <script>
                setTimeout(function() {{ window.location.reload(); }}, {refresh_sec * 1000});
            </script>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# Paths & constants
# =============================================================================
_UI_FILE = Path(__file__).resolve()            # .../data_engine/market_ai/ui/app.py
ENGINE_DIR = _UI_FILE.parents[1]               # .../data_engine/market_ai
# Repo root = parent of the top-level "data_engine" package (works for AI-Trading-Engine and older layouts)
ROOT = (ENGINE_DIR.parents[2])
PYTHON = sys.executable

# Ensure repo root is on sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE_DIR = ENGINE_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

AGENT_LOG = STATE_DIR / "agent.log"
SETTINGS_JSON = STATE_DIR / "agent_settings.json"
PID_FILE = STATE_DIR / "agent.pid"
CREDS_FILE = STATE_DIR / "creds.json"
AGENT_ENTRY = ENGINE_DIR / "start_agent.py"

# Fallback for unusual layout (older copies)
if not AGENT_ENTRY.exists():
    _alt = ROOT / "data_engine" / "market_ai" / "start_agent.py"
    if _alt.exists():
        AGENT_ENTRY = _alt

# Thread-safe queue for LTP updates from background poller
LTP_QUEUE: Queue = Queue(maxsize=128)

DEFAULT_SETTINGS = {
    "max_legs": 1,
    "lot_size": 75,
    "leg_sl_pct": 2.5,
    "profit_pct": 2.25,
    "warn_only": False,
    "batman_enabled": True,
    "batman_delta_breach": 0.30,
    "batman_premium_hard_x": 2.0,
    "batman_roll_distance": 150.0,
    "batman_hedge_delta_max": 0.12,
    "batman_hedge_price_max": 35.0,
    "batman_salvage_wing_ltp": 5.0,
}

BLOTTER_CSV = STATE_DIR / "trade_blotter.csv"
BLOTTER_SUMMARY = STATE_DIR / "trade_blotter_summary.json"
FEATURE_LOG_CSV = STATE_DIR / "feature_history.csv"
EQUITY_LOG_CSV = STATE_DIR / "equity_history.csv"

try:
    from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import BLOTTER_FIELDS as STRAT_BLOTTER_FIELDS  # type: ignore
except Exception:
    STRAT_BLOTTER_FIELDS = [
        "timestamp",
        "trade_mode",
        "warn_only",
        "executed",
        "side",
        "order_type",
        "exchange_seg",
        "product_type",
        "security_id",
        "quantity",
        "price",
        "strike",
        "tag",
        "notes",
    ]


def _ensure_csv_exists(path: Path, headers: List[str]) -> None:
    try:
        if path.exists() and path.stat().st_size > 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write(",".join(headers) + "\n")
    except Exception:
        pass

# =============================================================================
# Dhan wrapper import
# =============================================================================
try:
    from data_engine.market_ai.dhan_wrapper import DhanWrapper  # runtime import
    _import_err = None
except Exception as e:
    # Keep type-checkers happy without breaking runtime
    DhanWrapper = Any  # type: ignore[assignment]
    _import_err = e

# =============================================================================
# Session init
# =============================================================================
def _init_state() -> None:
    ss = st.session_state
    saved_creds: Dict[str, Any] = {}
    if CREDS_FILE.exists():
        try:
            saved_creds = json.loads(CREDS_FILE.read_text())
        except Exception:
            saved_creds = {}

    ss.setdefault("client_id", saved_creds.get("client_id", os.getenv("DHAN_CLIENT_ID", "")))
    ss.setdefault("access_token", saved_creds.get("access_token", os.getenv("DHAN_ACCESS_TOKEN", "")))
    ss.setdefault("creds_verified", bool(saved_creds.get("creds_verified", False)))

    ss.setdefault("funds", {"available": None, "collateral": None, "utilized": None, "withdrawable": None})
    ss.setdefault("last_funds_at", None)

    ss.setdefault("agent_pid", None)
    ss.setdefault("agent_running", False)
    ss.setdefault("trade_mode", "live")
    ss.setdefault("refresh_sec", 5)

    ss.setdefault("ltp_value", None)
    ss.setdefault("ltp_ts", None)

    ss.setdefault("dw", None)

    if not SETTINGS_JSON.exists():
        SETTINGS_JSON.write_text(json.dumps(DEFAULT_SETTINGS, indent=2))


def _load_settings() -> Dict[str, Any]:
    try:
        data = json.loads(SETTINGS_JSON.read_text())
        if not isinstance(data, dict):
            raise TypeError("settings not a dict")
    except Exception:
        data = {}
    merged = {**DEFAULT_SETTINGS, **data}
    # ensure types
    merged["max_legs"] = int(merged.get("max_legs", DEFAULT_SETTINGS["max_legs"]))
    merged["lot_size"] = int(merged.get("lot_size", DEFAULT_SETTINGS["lot_size"]))
    merged["leg_sl_pct"] = float(merged.get("leg_sl_pct", DEFAULT_SETTINGS["leg_sl_pct"]))
    merged["profit_pct"] = float(merged.get("profit_pct", DEFAULT_SETTINGS["profit_pct"]))
    merged["warn_only"] = bool(merged.get("warn_only", DEFAULT_SETTINGS["warn_only"]))
    merged["batman_enabled"] = bool(merged.get("batman_enabled", DEFAULT_SETTINGS["batman_enabled"]))
    merged["batman_delta_breach"] = float(merged.get("batman_delta_breach", DEFAULT_SETTINGS["batman_delta_breach"]))
    merged["batman_premium_hard_x"] = float(merged.get("batman_premium_hard_x", DEFAULT_SETTINGS["batman_premium_hard_x"]))
    merged["batman_roll_distance"] = float(merged.get("batman_roll_distance", DEFAULT_SETTINGS["batman_roll_distance"]))
    merged["batman_hedge_delta_max"] = float(merged.get("batman_hedge_delta_max", DEFAULT_SETTINGS["batman_hedge_delta_max"]))
    merged["batman_hedge_price_max"] = float(merged.get("batman_hedge_price_max", DEFAULT_SETTINGS["batman_hedge_price_max"]))
    merged["batman_salvage_wing_ltp"] = float(merged.get("batman_salvage_wing_ltp", DEFAULT_SETTINGS["batman_salvage_wing_ltp"]))
    return merged


def _load_blotter() -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    df: Optional[pd.DataFrame] = None
    summary: Dict[str, Any] = {}

    _ensure_csv_exists(BLOTTER_CSV, STRAT_BLOTTER_FIELDS)
    if BLOTTER_CSV.exists():
        try:
            df = pd.read_csv(BLOTTER_CSV)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.sort_values("timestamp")
        except Exception as exc:
            st.warning(f"Failed to read paper blotter: {exc}")
            df = None

    if BLOTTER_SUMMARY.exists():
        try:
            summary = json.loads(BLOTTER_SUMMARY.read_text())
        except Exception:
            summary = {}

    return df, summary


def _load_feature_history(limit: int = 200) -> Optional[pd.DataFrame]:
    if not FEATURE_LOG_CSV.exists():
        return None
    try:
        df = pd.read_csv(FEATURE_LOG_CSV)
    except Exception:
        return None
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
    if limit and len(df.index) > limit:
        df = df.tail(limit)
    return df.reset_index(drop=True)


def _load_equity_history(limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    if not EQUITY_LOG_CSV.exists():
        return None
    try:
        df = pd.read_csv(EQUITY_LOG_CSV)
    except Exception:
        return None
    if df.empty or "timestamp" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    numeric_cols = [
        "available",
        "collateral",
        "utilized",
        "withdrawable",
        "gross_exposure",
        "net_exposure",
        "unrealized",
        "realized",
        "equity_estimate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "equity_estimate" not in df.columns:
        df["equity_estimate"] = np.nan
    zero_series = pd.Series(0.0, index=df.index)
    available = df["available"] if "available" in df.columns else zero_series
    collateral = df["collateral"] if "collateral" in df.columns else zero_series
    utilized = df["utilized"] if "utilized" in df.columns else zero_series
    fallback_equity = (available.fillna(0.0) + collateral.fillna(0.0) + utilized.fillna(0.0))
    df["equity_estimate"] = df["equity_estimate"].fillna(fallback_equity)
    df = df.sort_values("timestamp")
    if limit and len(df.index) > limit:
        df = df.tail(limit)
    return df.reset_index(drop=True)


def _parse_context_blob(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"raw": value}
    return {}


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return False


def _fmt_optional(val: Any, digits: Optional[int] = 2, placeholder: str = "—") -> str:
    if val is None:
        return placeholder
    if isinstance(val, float) and np.isnan(val):
        return placeholder
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    try:
        if isinstance(val, (int, np.integer)):
            return f"{int(val):,}"
        if digits is None:
            return str(val)
        return f"{float(val):,.{digits}f}"
    except Exception:
        return str(val)


def _fmt_rupee(val: Optional[float], digits: int = 0) -> str:
    if val is None:
        return "—"
    try:
        value = float(val)
    except Exception:
        return "—"
    if np.isnan(value):
        return "—"
    return f"₹ {value:,.{digits}f}"


def _format_context_summary(ctx: Dict[str, Any], limit: int = 6) -> str:
    if not ctx:
        return ""
    items = []
    for key, value in ctx.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, bool):
            items.append(f"{key}={'yes' if value else 'no'}")
        elif isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
            if isinstance(value, float) and abs(value) < 0.01:
                items.append(f"{key}={value:.3f}")
            else:
                items.append(f"{key}={value}")
        else:
            items.append(f"{key}={value}")
        if len(items) >= limit:
            break
    return ", ".join(items)


def _compute_equity_change(df: pd.DataFrame, days: float) -> Optional[Tuple[float, Optional[float]]]:
    if df is None or df.empty:
        return None
    if "timestamp" not in df.columns or "equity_estimate" not in df.columns:
        return None
    if df["timestamp"].empty:
        return None
    window_start = df["timestamp"].iloc[-1] - pd.Timedelta(days=days)
    window = df.loc[df["timestamp"] >= window_start]
    if window.empty:
        return None
    start_val = float(window["equity_estimate"].iloc[0])
    end_val = float(window["equity_estimate"].iloc[-1])
    if np.isnan(start_val) or np.isnan(end_val) or start_val == 0:
        return None
    delta = end_val - start_val
    pct = (delta / start_val) * 100 if start_val else None
    return delta, pct


def _recent_risk_events(feature_df: Optional[pd.DataFrame], limit: int = 5) -> List[Dict[str, Any]]:
    if feature_df is None or feature_df.empty or "strategy" not in feature_df.columns:
        return []
    risk_rows = feature_df.loc[feature_df["strategy"] == "risk_event"].copy()
    if risk_rows.empty:
        return []
    if "timestamp" in risk_rows.columns:
        risk_rows = risk_rows.dropna(subset=["timestamp"]).sort_values("timestamp")
    risk_rows = risk_rows.tail(limit)
    events: List[Dict[str, Any]] = []
    for _, row in risk_rows.iterrows():
        ctx = _parse_context_blob(row.get("context"))
        events.append(
            {
                "timestamp": row.get("timestamp"),
                "label": ctx.get("label", "Risk Event"),
                "severity": ctx.get("severity", "info"),
                "message": ctx.get("message", ""),
                "details": {
                    k: v for k, v in ctx.items() if k not in {"label", "message", "severity"} and v not in (None, "")
                },
            }
        )
    return list(reversed(events))


def _render_risk_alerts(feature_df: Optional[pd.DataFrame]) -> None:
    events = _recent_risk_events(feature_df, limit=5)
    st.caption("Recent Risk Alerts")
    if not events:
        st.write("None")
        return
    severity_colors = {
        "INFO": "#2563eb",
        "WARNING": "#b45309",
        "ERROR": "#b91c1c",
    }
    for ev in events:
        severity = str(ev["severity"]).upper()
        color = severity_colors.get(severity, "#374151")
        ts = ev.get("timestamp")
        header = f"{_fmt_optional(ts)} · {ev.get('label', 'Risk Event')} ({severity})"
        lines = [ev.get("message") or "—"]
        details = ev.get("details") or {}
        if details:
            detail_text = ", ".join(f"{k}={v}" for k, v in details.items())
            lines.append(detail_text)
        st.markdown(
            f"<div style='border-left:4px solid {color}; padding-left:8px; margin-bottom:6px;'>"
            f"<strong>{header}</strong><br>{'<br>'.join(lines)}</div>",
            unsafe_allow_html=True,
        )
def _fmt_size(num: Optional[int]) -> str:
    if num is None:
        return "—"
    step = 1024.0
    value = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < step:
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= step
    return f"{value:,.1f} PB"


def _collect_file_info(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": path,
        "exists": path.exists(),
        "size": None,
        "updated": None,
    }
    try:
        if path.exists():
            stat = path.stat()
            info["size"] = stat.st_size
            info["updated"] = datetime.fromtimestamp(stat.st_mtime)
    except Exception:
        pass
    return info


def _age_string(ts: Optional[datetime]) -> str:
    if ts is None:
        return "—"
    delta = datetime.now() - ts
    if delta.total_seconds() < 60:
        return f"{int(delta.total_seconds())}s ago"
    if delta.total_seconds() < 3600:
        return f"{int(delta.total_seconds()//60)}m ago"
    if delta.total_seconds() < 86400:
        return f"{int(delta.total_seconds()//3600)}h ago"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _read_tail(path: Path, lines: int = 50) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            data = fh.readlines()
        return data[-lines:]
    except Exception:
        return []


def _clear_stale_pid_file() -> None:
    """Remove PID file and reset session flags."""
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass
    ss = st.session_state
    ss["agent_pid"] = None
    ss["agent_running"] = False
    ss.pop("agent_started_mode", None)

def _mark_creds_dirty() -> None:
    """Flag credentials as needing verification when the user edits them."""
    ss = st.session_state
    ss["creds_verified"] = False
    ss["dw"] = None
    ss["funds"] = {"available": None, "collateral": None, "utilized": None, "withdrawable": None}
    ss["last_funds_at"] = None
    try:
        if CREDS_FILE.exists():
            CREDS_FILE.unlink()
    except Exception:
        pass

def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True

def _sync_agent_state_from_pidfile() -> None:
    """Reload agent status from pid file when the page/session refreshes."""
    ss = st.session_state
    if ss.get("agent_running") and ss.get("agent_pid"):
        return
    if not PID_FILE.exists():
        return
    try:
        data = json.loads(PID_FILE.read_text())
        pid = int(data.get("pid"))
    except Exception:
        return
    if not _pid_alive(pid):
        try:
            if PID_FILE.exists():
                PID_FILE.unlink()
        except Exception:
            pass
        return
    ss["agent_pid"] = pid
    ss["agent_running"] = True
    if data.get("trade_mode"):
        ss["trade_mode"] = data["trade_mode"]
        ss["agent_started_mode"] = data["trade_mode"]

# =============================================================================
# Wrapper helpers
# =============================================================================
def _new_dw() -> Optional[Any]:
    """Create a DhanWrapper instance from sidebar creds."""
    if _import_err:
        st.error(f"Failed to import DhanWrapper: {_import_err}")
        return None
    cid = (st.session_state.get("client_id") or "").strip()
    tok = (st.session_state.get("access_token") or "").strip()
    if not cid or not tok:
        st.error("Enter Client ID and Access Token.")
        return None
    try:
        return DhanWrapper(dhan_client_id=cid, access_token=tok)
    except Exception as e:
        st.error(f"Failed to create Dhan client: {e}")
        return None

def _drain_ltp_queue_into_state() -> None:
    """Transfer any pending LTP updates from background thread to session_state."""
    ss = st.session_state
    while True:
        try:
            ltp, ts = LTP_QUEUE.get_nowait()
        except Empty:
            break
        ss["ltp_value"] = ltp
        ss["ltp_ts"] = ts

def _post_dw_setup(dw: Any) -> None:
    """Fetch funds, remember timestamp, and ensure NIFTY poller runs."""
    ss = st.session_state

    try:
        ss["funds"] = dw.get_funds()
        ss["last_funds_at"] = datetime.now()
    except Exception as e:
        st.warning(f"Funds fetch failed: {e}")

    try:
        seg, secid = "IDX_I", 13

        def _on_ltp_threadsafe(ltp: float, ts: datetime) -> None:
            try:
                LTP_QUEUE.put_nowait((ltp, ts))
            except Exception:
                pass

        dw.start_ltp_poller(
            seg,
            secid,
            max(1, int(ss.get("refresh_sec", 5))),
            _on_ltp_threadsafe,
            poll_when_closed=True,
        )
        # Prime once so the tile shows data immediately after verification
        try:
            ltp_once = dw.get_ltp_once(seg, secid)
            if ltp_once is not None:
                ss["ltp_value"] = float(ltp_once)
                ss["ltp_ts"] = datetime.now()
        except Exception:
            pass
    except Exception as e:
        st.info(f"No LTP from /marketfeed/ltp. Market may be closed or permission missing. ({e})")


def _verify_and_remember_creds() -> bool:
    """Verify creds, fetch funds once, and start/refresh NIFTY LTP poller."""
    ss = st.session_state
    os.environ["DHAN_CLIENT_ID"] = (ss.get("client_id") or "").strip()
    os.environ["DHAN_ACCESS_TOKEN"] = (ss.get("access_token") or "").strip()

    dw = _new_dw()
    if not dw:
        ss["creds_verified"] = False
        ss["dw"] = None
        return False

    ss["dw"] = dw
    ss["creds_verified"] = True
    try:
        CREDS_FILE.write_text(json.dumps({
            "client_id": ss.get("client_id", ""),
            "access_token": ss.get("access_token", ""),
            "creds_verified": True,
            "verified_at": datetime.now().isoformat(),
        }, indent=2))
    except Exception:
        pass
    _post_dw_setup(dw)

    return True

# =============================================================================
# Agent process control
# =============================================================================
def _agent_env() -> dict:
    base = os.environ.copy()
    base["DHAN_CLIENT_ID"] = st.session_state.get("client_id", "")
    base["DHAN_ACCESS_TOKEN"] = st.session_state.get("access_token", "")
    base["TRADE_MODE"] = st.session_state.get("trade_mode", "live")
    base["AGENT_SETTINGS_JSON"] = str(SETTINGS_JSON)
    base["PYTHONUNBUFFERED"] = "1"
    base["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([base["PYTHONPATH"]] if base.get("PYTHONPATH") else []))
    settings = _load_settings()
    base["MAX_STRANGLES"] = str(settings.get("max_legs", 1))
    base["NIFTY_LOT"] = str(settings.get("lot_size", 75))
    # convert % -> decimal
    base["SL_PCT"] = str(float(settings.get("leg_sl_pct", 2.5)) / 100.0)
    base["TP_PCT"] = str(float(settings.get("profit_pct", 2.25)) / 100.0)
    warn_only = bool(settings.get("warn_only")) or base["TRADE_MODE"].lower() == "paper"
    base["WARN_ONLY"] = "1" if warn_only else "0"
    base["BLOTTER_PATH"] = str(STATE_DIR / "trade_blotter.csv")
    base["BLOTTER_SUMMARY_PATH"] = str(STATE_DIR / "trade_blotter_summary.json")
    base["FEATURE_LOG_PATH"] = str(STATE_DIR / "feature_history.csv")
    base["BATMAN_ENABLED"] = "1" if settings.get("batman_enabled", True) else "0"
    base["BATMAN_DELTA_BREACH"] = str(settings.get("batman_delta_breach", 0.30))
    base["BATMAN_PREMIUM_HARD_X"] = str(settings.get("batman_premium_hard_x", 2.0))
    base["BATMAN_ROLL_DISTANCE"] = str(settings.get("batman_roll_distance", 150.0))
    base["BATMAN_HEDGE_DELTA_MAX"] = str(settings.get("batman_hedge_delta_max", 0.12))
    base["BATMAN_HEDGE_PRICE_MAX"] = str(settings.get("batman_hedge_price_max", 35.0))
    base["BATMAN_SALVAGE_WING_LTP"] = str(settings.get("batman_salvage_wing_ltp", 5.0))
    base["STRATEGY_MODEL_PATH"] = str(STATE_DIR / "strategy_selector_model.json")
    return base

def start_agent() -> None:
    if st.session_state.get("agent_running"):
        return
    if not AGENT_ENTRY.exists():
        st.error(f"start_agent.py not found at: {AGENT_ENTRY}")
        return
    try:
        AGENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        AGENT_LOG.touch(exist_ok=True)
        env = _agent_env()
        proc = subprocess.Popen(
            [PYTHON, str(AGENT_ENTRY)],
            cwd=str(AGENT_ENTRY.parent),
            env=env,
            stdout=open(AGENT_LOG, "a"),
            stderr=open(AGENT_LOG, "a"),
            start_new_session=True,
        )
        st.session_state["agent_pid"] = proc.pid
        st.session_state["agent_running"] = True
        st.session_state["agent_started_mode"] = st.session_state.get("trade_mode", "live")
        try:
            PID_FILE.write_text(json.dumps({
                "pid": proc.pid,
                "trade_mode": st.session_state.get("trade_mode", "live"),
                "started_at": datetime.now().isoformat(),
            }, indent=2))
        except Exception:
            pass
    except Exception as e:
        st.error(f"Failed to start agent: {e}")

def stop_agent() -> None:
    pid = st.session_state.get("agent_pid")
    if not pid:
        st.session_state["agent_running"] = False
        return
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
        else:
            os.kill(pid, 15)  # SIGTERM
    except Exception:
        pass
    finally:
        st.session_state["agent_pid"] = None
        st.session_state["agent_running"] = False
        st.session_state.pop("agent_started_mode", None)
        try:
            if PID_FILE.exists():
                PID_FILE.unlink()
        except Exception:
            pass

# =============================================================================
# Trade tab widgets
# =============================================================================
def _funds_cards(funds: Dict[str, Any]) -> None:
    cols = st.columns(4)
    labels = ["Available", "Collateral", "Utilized", "Withdrawable"]
    keys   = ["available", "collateral", "utilized", "withdrawable"]
    for col, label, key in zip(cols, labels, keys):
        with col:
            st.caption(label)
            val = funds.get(key)
            st.write("—" if val is None else f"₹ {val:,.0f}")

def _nifty_tile() -> None:
    _auto_refresh()
    _drain_ltp_queue_into_state()
    st.markdown("### NIFTY Spot Price")
    dw = st.session_state.get("dw")
    if not st.session_state.get("creds_verified") or not dw:
        st.info("Enter your credentials and click **Save & Verify** to start the LTP feed.")
        return
    v = st.session_state.get("ltp_value")
    ts = st.session_state.get("ltp_ts")

    if dw and (v is None or (ts and (datetime.now() - ts) > timedelta(seconds=max(5, int(st.session_state.get("refresh_sec", 5)) * 2)))):
        try:
            ltp = dw.get_ltp_once("IDX_I", 13)
            if ltp is not None:
                st.session_state["ltp_value"] = ltp
                st.session_state["ltp_ts"] = datetime.now()
                v, ts = ltp, st.session_state["ltp_ts"]
        except Exception:
            pass

    if v is None:
        st.info("No LTP from /marketfeed/ltp.")
    else:
        st.markdown(f"**{v:,.2f}**")
        if ts:
            st.caption(f"Last update: {ts:%H:%M:%S}")

# =============================================================================
# Positions tab (open + today's closed)
# =============================================================================
def _positions_tab(dw) -> None:

    st.markdown("### Positions")

    if not st.session_state.get("creds_verified") or not dw:
        st.info("Save & verify credentials to view live positions.")
        return

    try:
        rows = dw.get_positions_live_with_ltp()  # type: ignore[attr-defined]
    except Exception as e:
        st.error(f"Failed to fetch positions: {e}")
        return

    def _raw(r: Dict[str, Any]) -> Dict[str, Any]:
        val = r.get("_raw")
        return val if isinstance(val, dict) else {}

    # ---- helpers over normalized rows (with raw fallback) ----
    def S(r, k, default=""):
        v = r.get(k)
        if v is None:
            v = _raw(r).get(k)  # fallback
        return v if v is not None else default

    def F(r, k, default=0.0):
        try:
            v = r.get(k)
            if v is None:
                v = _raw(r).get(k)
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    def I(r, k, default=0):
        try:
            v = r.get(k)
            if v is None:
                v = _raw(r).get(k)
            return int(float(v)) if v is not None else int(default)
        except Exception:
            return int(default)

    def _pt(r: Dict[str, Any]) -> str:
        return str(r.get("position_type") or r.get("side") or S(r, "positionType")).upper()

    def _net_qty(r: Dict[str, Any]) -> int:
        if r.get("qty") is not None:
            return int(r.get("qty"))
        return I(r, "netQty", 0)

    def _security_id(r: Dict[str, Any]) -> Optional[int]:
        raw = r.get("security_id") or r.get("securityId") or r.get("id")
        if raw is None:
            raw = _raw(r).get("securityId")
        try:
            return int(str(raw)) if raw is not None else None
        except Exception:
            return None

    # ================= OPEN POSITIONS (netQty != 0) =================
    open_src = [r for r in rows if _net_qty(r) != 0 and _pt(r) in ("LONG", "SHORT")]
    ltp_map: Dict[tuple[str, int], Optional[float]] = {}
    ltp_pairs: List[tuple[str, int]] = []
    seen_pairs: set[tuple[str, int]] = set()
    for r in open_src:
        seg = r.get("exchange_seg") or S(r, "exchangeSegment")
        sid = _security_id(r)
        if isinstance(seg, str) and sid is not None:
            key = (seg, sid)
            if key not in seen_pairs:
                seen_pairs.add(key)
                ltp_pairs.append(key)
    if ltp_pairs:
        try:
            ltp_map = dw.get_ltp_bulk(ltp_pairs)  # type: ignore[attr-defined]
        except Exception as exc:
            st.warning(f"Could not fetch live LTPs: {exc}")

    # ---- Payoff Analyzer Button ----
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button("🔍 Analyze Payoff", key="analyze_payoff", help="Open payoff chart based on open positions"):
                st.session_state["payoff_open"] = True

    payoff_open = bool(st.session_state.get("payoff_open", False))
    if payoff_open:
        with st.expander("Payoff Chart (Live Positions)", expanded=True):
            c_close, _ = st.columns([1, 5])
            if c_close.button("Close analyzer", key="close_payoff"):
                st.session_state["payoff_open"] = False
            else:
                _render_payoff_analyzer(open_src, dw)

    if open_src:
        mapped_open = []
        total_unreal = 0.0
        for r in open_src:
            typ = _pt(r)
            sym = r.get("symbol") or S(r, "tradingSymbol")
            prod = r.get("product") or S(r, "productType")
            cost_price = F(r, "cost_price", F(r, "avg_price", 0.0))
            buy_avg = r.get("buy_avg")
            if buy_avg is None:
                buy_avg = F(r, "buyAvg", None)
            sell_avg = r.get("sell_avg")
            if sell_avg is None:
                sell_avg = F(r, "sellAvg", None)
            unreal = r.get("unrealized_profit")
            if unreal is None:
                unreal = F(r, "unrealizedProfit", 0.0)
            seg = r.get("exchange_seg") or S(r, "exchangeSegment")
            sid = _security_id(r)
            live_ltp = ltp_map.get((seg, sid)) if seg and sid is not None else None
            if live_ltp is None and seg and sid is not None:
                try:
                    live_single = dw.get_ltp_once(seg, sid)  # type: ignore[attr-defined]
                    if live_single is not None:
                        live_ltp = live_single
                except Exception:
                    pass

            avg_disp = sell_avg if typ == "SHORT" else buy_avg
            if avg_disp is None:
                avg_disp = cost_price
            qty_val = abs(_net_qty(r))
            if typ == "SHORT":
                qty_val = I(r, "sellQty", qty_val)
            else:
                qty_val = I(r, "buyQty", qty_val)
            qty_val = abs(int(qty_val or 0))

            ltp_val = live_ltp if live_ltp is not None else r.get("ltp")
            if ltp_val is None:
                ltp_val = cost_price

            mapped_open.append({
                "B/S": "S" if typ == "SHORT" else "B",
                "Name": sym,
                "Product": prod,
                "Qty": qty_val,
                "Avg Price": avg_disp,
                "LTP": ltp_val,
                "P&L": unreal,
            })
            total_unreal += unreal

        import pandas as pd
        df_open = pd.DataFrame(mapped_open)
        # Qty as nullable Int64 so Streamlit/Arrow is happy
        try:
            df_open["Qty"] = df_open["Qty"].astype("Int64")
        except Exception:
            pass

        def _pnl_style_col(col):
            return ["color: #d32f2f" if v < 0 else ("color: #2e7d32" if v > 0 else "") for v in col]
        st.markdown("#### Open Positions")
        styler_open = (
            df_open.style
            .format({"Avg Price": "₹ {:.2f}", "LTP": "₹ {:.2f}", "P&L": "₹ {:.2f}"})
            .apply(_pnl_style_col, subset=["P&L"], axis=0)
        )
        st.dataframe(styler_open, hide_index=True, width="stretch")
        total_color = "#2e7d32" if total_unreal > 0 else ("#d32f2f" if total_unreal < 0 else "inherit")
        st.markdown(f"<span style='color:{total_color};'>Total Unrealized P&L: ₹ {total_unreal:,.2f}</span>", unsafe_allow_html=True)
    else:
        st.caption("No open positions.")

    # ================= CLOSED (today) =================
    # Treat as closed if positionType == CLOSED or netQty == 0.
    # Show if there was intraday activity or realized PnL (like Dhan UI).
    closed_candidates = [r for r in rows if _pt(r) == "CLOSED" or _net_qty(r) == 0]
    closed_today = [
        r for r in closed_candidates
        if I(r, "dayBuyQty", 0) > 0 or I(r, "daySellQty", 0) > 0 or F(r, "realizedProfit", 0.0) != 0.0
    ]

    if closed_today:
        # de-dupe by tradingSymbol (keep the last occurrence)
        seen = {}
        for r in closed_today:
            key = S(r, "tradingSymbol").strip()
            if key:
                seen[key] = r
        uniq = list(seen.values()) if seen else closed_today

        mapped_closed = []
        total_realized = 0.0
        for r in uniq:
            realized = F(r, "realizedProfit", 0.0)
            total_realized += realized
            mapped_closed.append({
                "Symbol":   S(r, "tradingSymbol"),
                "Product":  S(r, "productType"),
                "Qty":      max(I(r, "buyQty", 0), I(r, "sellQty", 0)),  # show traded size
                "Buy Avg":  F(r, "buyAvg", 0.0),
                "Sell Avg": F(r, "sellAvg", 0.0),
                "LTP":      F(r, "costPrice", 0.0),
                "P&L":      realized,
            })

        import pandas as pd
        df_closed = pd.DataFrame(mapped_closed)
        try:
            df_closed["Qty"] = df_closed["Qty"].astype("Int64")
        except Exception:
            pass

        def _pnl_style_col_closed(col):
            return ["color: #d32f2f" if v < 0 else ("color: #2e7d32" if v > 0 else "") for v in col]
        st.markdown("#### Today’s Closed Trades")
        styler_closed = (
            df_closed.style
            .format({"Buy Avg": "₹ {:.2f}", "Sell Avg": "₹ {:.2f}", "LTP": "₹ {:.2f}", "P&L": "₹ {:.2f}"})
            .apply(_pnl_style_col_closed, subset=["P&L"], axis=0)
        )
        st.dataframe(styler_closed, hide_index=True, width="stretch")
        closed_color = "#2e7d32" if total_realized > 0 else ("#d32f2f" if total_realized < 0 else "inherit")
        st.markdown(f"<span style='color:{closed_color};'>Realized P&L (today): ₹ {total_realized:,.2f}</span>", unsafe_allow_html=True)
    else:
        st.caption("No closed trades today.")

# ---------- Payoff & Greeks Analyzer ----------
def _collect_open_option_legs(rows):
    legs = []
    for r in rows or []:
        try:
            raw = r.get("_raw") or {}
            pt = str(r.get("positionType") or raw.get("positionType") or "").upper()
            if pt not in ("LONG", "SHORT"):
                continue
            nqty = r.get("qty")
            if nqty is None:
                nqty = r.get("net_qty") or r.get("netQty") or raw.get("netQty")
            nqty = int(float(nqty or 0))
            if nqty == 0:
                continue
            side = pt
            qty = abs(nqty)
            strike = r.get("strike") or r.get("drvStrikePrice") or raw.get("drvStrikePrice")
            strike = float(strike or 0)
            otype = str(r.get("type") or r.get("drvOptionType") or raw.get("drvOptionType") or "").upper()
            if otype not in ("CALL", "PUT") or strike == 0:
                continue
            premium = r.get("sell_avg") if side == "SHORT" else r.get("buy_avg")
            if premium is None:
                premium = r.get("sellAvg") if side == "SHORT" else r.get("buyAvg")
            if premium is None:
                premium = raw.get("sellAvg") if side == "SHORT" else raw.get("buyAvg")
            premium = float(premium or 0.0)
            ltp = r.get("ltp") or r.get("cost_price") or r.get("costPrice") or raw.get("costPrice") or 0.0
            ltp = float(ltp)
            expiry = str(r.get("expiry") or r.get("drvExpiryDate") or raw.get("drvExpiryDate") or "")
            legs.append({
                "symbol": r.get("symbol") or raw.get("tradingSymbol"),
                "type": otype,          # CALL / PUT
                "side": side,           # LONG / SHORT
                "qty": qty,
                "strike": strike,
                "premium": premium,
                "ltp": ltp,
                "expiry": expiry,
            })
        except Exception:
            continue
    return legs


def _fallback_plan_from_positions() -> Optional[pd.DataFrame]:
    dw = st.session_state.get("dw")
    if not dw or not st.session_state.get("creds_verified"):
        return None
    try:
        rows = dw.get_positions_live_with_ltp()
    except Exception:
        return None
    legs = _collect_open_option_legs(rows)
    if not legs:
        return None
    combos: List[Dict[str, Any]] = []
    short_calls = [leg for leg in legs if leg["side"] == "SHORT" and leg["type"] == "CALL"]
    short_puts = [leg for leg in legs if leg["side"] == "SHORT" and leg["type"] == "PUT"]
    expiries = sorted({leg["expiry"] for leg in short_calls + short_puts if leg.get("expiry")})
    for expiry in expiries:
        ce_candidates = [leg for leg in short_calls if leg["expiry"] == expiry]
        pe_candidates = [leg for leg in short_puts if leg["expiry"] == expiry]
        if not ce_candidates or not pe_candidates:
            continue
        ce_leg = max(ce_candidates, key=lambda l: l["strike"])
        pe_leg = min(pe_candidates, key=lambda l: l["strike"])
        qty = min(ce_leg["qty"], pe_leg["qty"])
        net_credit = qty * ((ce_leg.get("premium") or 0.0) + (pe_leg.get("premium") or 0.0))
        combos.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "strategy": "live_positions",
            "expiry": expiry,
            "ce_strike": ce_leg["strike"],
            "pe_strike": pe_leg["strike"],
            "ce_ltp": ce_leg.get("ltp"),
            "pe_ltp": pe_leg.get("ltp"),
            "ce_delta": None,
            "pe_delta": None,
            "net_credit": net_credit,
            "context": json.dumps({"source": "open_positions", "qty": qty}, default=str),
        })
    if not combos:
        return None
    return pd.DataFrame(combos)


def _fallback_blotter_from_positions() -> Optional[pd.DataFrame]:
    dw = st.session_state.get("dw")
    if not dw or not st.session_state.get("creds_verified"):
        return None
    try:
        rows = dw.get_positions_live_with_ltp()
    except Exception:
        return None
    events: List[Dict[str, Any]] = []
    for row in rows:
        qty = row.get("qty")
        if not qty:
            continue
        side = "BUY" if qty > 0 else "SELL"
        strike = (row.get("_raw") or {}).get("drvStrikePrice")
        events.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trade_mode": st.session_state.get("trade_mode", "live"),
            "side": side,
            "quantity": abs(int(qty)),
            "price": row.get("avg_price"),
            "strike": strike or "",
            "notes": row.get("symbol") or "",
        })
    if not events:
        return None
    return pd.DataFrame(events)

# --- Colours roughly matching Dhan ---
_DHAN_COLORS = {
    "today": "#2e7d32",   # green
    "target": "#1976d2",  # blue
    "expiry": "#d32f2f",  # red
    "breakeven": "#9e9e9e"
}

def _to_years(date_str: str) -> float:
    """Time to expiry in years from ISO date, fallback ~20 trading days."""
    try:
        dt = datetime.fromisoformat(date_str)
    except Exception:
        return 20/252.0
    days = max(0.0, (dt - datetime.now()).days)
    return days/252.0

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """
    Standard normal CDF for array-like x.

    Uses math.erf (vectorized) to avoid relying on np.erf (which may not exist in some NumPy builds).
    """
    x = np.asarray(x, dtype=float)
    v_erf = np.vectorize(erf)
    return 0.5 * (1.0 + v_erf(x / np.sqrt(2.0)))

def _bs_price_greeks(S, K, r, q, sigma, T, is_call):
    """
    Black–Scholes price + greeks (Delta, Gamma, Theta/day, Vega per 1%).
    """
    if sigma <= 0 or T <= 0:
        # expiry-like fallback
        if is_call:
            price = max(0.0, S - K)
            delta = 1.0 if S > K else 0.0
        else:
            price = max(0.0, K - S)
            delta = -1.0 if S < K else 0.0
        return {"price": price, "delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    n_pdf = (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*d1**2)

    if is_call:
        price = S*np.exp(-q*T)*Nd1 - K*np.exp(-r*T)*Nd2
        delta = np.exp(-q*T)*Nd1
    else:
        price = K*np.exp(-r*T)*(1-Nd2) - S*np.exp(-q*T)*(1-Nd1)
        delta = -np.exp(-q*T)*(1-Nd1)

    gamma = (np.exp(-q*T) * n_pdf) / (S * sigma * np.sqrt(T))
    theta = (- (S*np.exp(-q*T)*n_pdf*sigma) / (2*np.sqrt(T))
             - (r*K*np.exp(-r*T)*Nd2 if is_call else r*K*np.exp(-r*T)*(1-Nd2))
             + (q*S*np.exp(-q*T)*Nd1 if is_call else q*S*np.exp(-q*T)*(1-Nd1)))
    theta = theta / 365.0
    vega = S*np.exp(-q*T)*n_pdf*np.sqrt(T) / 100.0
    return {"price": float(price), "delta": float(delta), "gamma": float(gamma),
            "theta": float(theta), "vega": float(vega)}


def _render_payoff_analyzer(open_rows, dw):
    legs = _collect_open_option_legs(open_rows)
    if not legs:
        st.warning("No option legs available for payoff analysis.")
        return

    # --- left: selectable legs & settings ---
    st.markdown("&nbsp;", unsafe_allow_html=True)
    cols = st.columns([0.35, 0.65])

    with cols[0]:
        st.caption("Open Positions:")
        enabled = {}
        for i, leg in enumerate(legs):
            tag = "S" if leg["side"] == "SHORT" else "B"
            label = f"{tag}  {leg['symbol'] or (leg['type'][0] + ' ' + str(int(leg['strike'])))}"
            enabled[i] = st.checkbox(label, value=True, key=f"leg_{i}")

        st.divider()
        st.caption("Settings")
        spot_guess = None
        try:
            spot_guess = float(dw.get_ltp_once("IDX_I", 13))
        except Exception:
            pass
        spot = st.number_input("Spot (NIFTY)", min_value=0.0,
                               value=float(spot_guess or np.mean([l['strike'] for l in legs])),
                               step=1.0)
        iv = st.slider("Assumed IV (%)", 5.0, 60.0, 18.0, 0.5)
        r = st.slider("Risk-free (%)", 0.0, 10.0, 6.5, 0.1)
        q = st.slider("Div. yield (%)", 0.0, 3.0, 0.0, 0.1)

    active_legs = [leg for i, leg in enumerate(legs) if enabled.get(i)]
    if not active_legs:
        with cols[1]:
            st.info("Select at least one leg.")
        return

    with cols[1]:
        tabs = st.tabs(["Pay-Off", "Greeks"])

        # ===== Pay-Off (today vs expiry) =====
        with tabs[0]:
            xs = np.linspace(spot * 0.9, spot * 1.1, 250)
            total_today = np.zeros_like(xs)
            total_exp = np.zeros_like(xs)

            for leg in active_legs:
                K = float(leg["strike"])
                sigma = float(iv)/100.0
                T = _to_years(leg.get("expiry") or "")
                qty = int(leg["qty"])
                side_mult = -1 if leg["side"] == "SHORT" else 1
                is_call = (leg["type"] == "CALL")

                # Today: Black–Scholes mark-to-model vs entry premium
                today_vals = []
                for S in xs:
                    g = _bs_price_greeks(S, K, r/100.0, q/100.0, sigma, max(T, 1/252.0), is_call)
                    price = g["price"]
                    pnl = (price - leg["premium"]) if side_mult == 1 else (leg["premium"] - price)
                    today_vals.append(pnl * qty)
                total_today += np.array(today_vals)

                # Expiry: intrinsic vs premium
                intrinsic = np.maximum(0.0, xs - K) if is_call else np.maximum(0.0, K - xs)
                exp_pnl = (intrinsic - leg["premium"]) if side_mult == 1 else (leg["premium"] - intrinsic)
                total_exp += exp_pnl * qty

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=total_today, mode="lines", name="Today",
                                     line=dict(color=_DHAN_COLORS["today"]),
                                     hovertemplate="Spot ₹%{x:.2f}<br>P&L ₹%{y:,.0f}"))
            fig.add_trace(go.Scatter(x=xs, y=total_exp, mode="lines", name="Expiry",
                                     line=dict(color=_DHAN_COLORS["expiry"]),
                                     hovertemplate="Spot ₹%{x:.2f}<br>P&L ₹%{y:,.0f}"))
            fig.add_vline(x=float(spot), line_dash="dash", line_color="#455a64",
                          annotation_text=f"Spot ₹{spot:,.0f}")

            # Breakevens from expiry curve
            try:
                idx = np.where(np.diff(np.sign(total_exp)))[0]
                for i in idx:
                    x0, x1, y0, y1 = xs[i], xs[i+1], total_exp[i], total_exp[i+1]
                    be = x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0)
                    fig.add_vline(x=float(be), line_dash="dot",
                                  line_color=_DHAN_COLORS["breakeven"],
                                  annotation_text=f"BE ₹{be:,.0f}")
            except Exception:
                pass

            fig.update_layout(title="Pay-Off Simulation",
                              xaxis_title="Spot Price", yaxis_title="Net P&L (₹)",
                              template="plotly_white", height=520,
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")

            # Quick stats
            mx = np.nanmax(total_exp)
            mn = np.nanmin(total_exp)
            st.markdown(
                f"**Max Profit**: ₹ {mx:,.0f} &nbsp;&nbsp; "
                f"**Max Loss**: {'Unlimited' if mn < -1e6 else '₹ ' + format(mn, ',.0f')}"
            )

        # ===== Greeks (at current spot; signed by side/qty) =====
        with tabs[1]:
            rows = []
            for leg in active_legs:
                K = float(leg["strike"])
                T = _to_years(leg.get("expiry") or "")
                g = _bs_price_greeks(spot, K, r/100.0, q/100.0, float(iv)/100.0,
                                     max(T, 1/252.0), leg["type"] == "CALL")
                mult = (-1 if leg["side"] == "SHORT" else 1) * int(leg["qty"])
                rows.append({
                    "Leg": (leg["symbol"] or f"{leg['type'][0]} {int(K)}"),
                    "Side": "S" if leg["side"] == "SHORT" else "B",
                    "Qty": leg["qty"],
                    "Δ": round(mult * g["delta"], 4),
                    "Γ": round(mult * g["gamma"], 6),
                    "Θ/day": round(mult * g["theta"], 2),
                    "Vega": round(mult * g["vega"], 2),
                    "Model Px": round(g["price"], 2),
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# =============================================================================
# Agent Logs tab
# =============================================================================
def _agent_logs_tab() -> None:
    st.markdown("### Agent Logs")
    try:
        if not AGENT_LOG.exists():
            st.info("No agent log found yet.")
            return
        with open(AGENT_LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if not lines:
            st.info("Log file is empty.")
            return
        tail = lines[-800:]
        st.code("".join(tail), language="log")
    except Exception as e:
        st.error(f"Failed to read agent logs: {e}")


def _render_blotter_panel(blotter_df: Optional[pd.DataFrame], summary: Dict[str, Any]) -> None:
    st.markdown("#### Trade Blotter")
    if summary:
        metrics = [
            ("Total", summary.get("total_orders")),
            ("Executed", summary.get("executed_orders")),
            ("Warn-only", summary.get("warn_only_orders")),
        ]
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.metric(label, _fmt_optional(value, digits=None))
    if blotter_df is None:
        st.info("Blotter file not available yet. Start the agent to capture trades.")
        return
    if blotter_df.empty:
        fallback = _fallback_blotter_from_positions()
        if fallback is not None:
            st.info("No agent trades recorded yet. Showing live positions (read-only).")
            st.dataframe(fallback, hide_index=True, width="stretch")
        else:
            st.info("No trades recorded yet.")
        return
    table = blotter_df.copy()
    if "timestamp" in table.columns:
        table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    latest = table.tail(100)
    columns = [c for c in ["timestamp", "trade_mode", "side", "order_type", "product_type", "quantity", "price", "strike", "notes"] if c in latest.columns]
    st.dataframe(latest[columns], hide_index=True, width="stretch")
    st.caption("Showing the latest 100 blotter entries.")


def _render_agent_activity(feature_df: Optional[pd.DataFrame]) -> None:
    st.markdown("#### Agent Activity Feed")
    if feature_df is None:
        st.info("Decision feed not available yet. Ensure FEATURE_LOG_PATH is configured for the agent.")
        return
    if feature_df.empty:
        st.info("Waiting for the first strategy telemetry event.")
        return
    if "timestamp" not in feature_df.columns:
        st.info("Feature log missing timestamp column.")
        return
    recent = feature_df.dropna(subset=["timestamp"]).tail(25)
    recent = recent.sort_values("timestamp", ascending=False)
    for _, row in recent.iterrows():
        ts = row.get("timestamp")
        ts_text = _fmt_optional(ts)
        raw_strategy = str(row.get("strategy") or "unknown")
        strategy_name = raw_strategy.title()

        if raw_strategy == "environment":
            ctx = _parse_context_blob(row.get("context"))
            state = ctx.get("market_state", {})
            candidates = ctx.get("strategy_candidates", [])
            trend = str(state.get("trend", "—")).replace("_", " ").title()
            vol = str(state.get("volatility", "—")).replace("_", " ").title()
            ivr = state.get("iv_rank")
            bullet = [f"Trend {trend}", f"Vol {vol}"]
            if ivr is not None:
                bullet.append(f"IV rank {_fmt_optional(ivr)}")
            if candidates:
                top = candidates[0]
                bullet.append(f"Top: {top.get('name', '—')} (score {top.get('score')})")
            st.markdown(f"**{ts_text}** · Market Environment\n\n- " + " | ".join(bullet))
            continue

        if raw_strategy == "risk_event":
            ctx = _parse_context_blob(row.get("context"))
            label = str(ctx.get("label", "Risk Event")).title()
            severity = str(ctx.get("severity", "info")).upper()
            message = ctx.get("message") or ""
            extras = [
                f"{k}={v}"
                for k, v in ctx.items()
                if k not in {"label", "message", "severity"} and v not in (None, "", [], {})
            ]
            extra_text = (" | ".join(extras)) if extras else ""
            lines = [message or "—"]
            if extra_text:
                lines.append(extra_text)
            st.markdown(f"**{ts_text}** · {label} ({severity})\n\n" + "\n".join(f"- {line}" for line in lines))
            continue

        details = []
        spot_val = row.get("spot")
        if not _is_missing(spot_val):
            details.append(f"Spot ₹ {_fmt_optional(spot_val)}")
        ce_strike = row.get("ce_strike")
        pe_strike = row.get("pe_strike")
        ce_delta = row.get("ce_delta")
        pe_delta = row.get("pe_delta")
        ce_ltp = row.get("ce_ltp")
        pe_ltp = row.get("pe_ltp")
        if not _is_missing(ce_strike):
            details.append(f"CE { _fmt_optional(ce_strike, digits=0) } Δ {_fmt_optional(ce_delta, digits=3)} @ {_fmt_optional(ce_ltp)}")
        if not _is_missing(pe_strike):
            details.append(f"PE { _fmt_optional(pe_strike, digits=0) } Δ {_fmt_optional(pe_delta, digits=3)} @ {_fmt_optional(pe_ltp)}")
        net_credit = row.get("net_credit")
        if not _is_missing(net_credit):
            details.append(f"Net credit ₹ {_fmt_optional(net_credit)}")
        ctx_summary = _format_context_summary(_parse_context_blob(row.get("context")))
        bullet_lines = []
        if details:
            bullet_lines.append(" | ".join(details))
        if ctx_summary:
            bullet_lines.append(f"Context: {ctx_summary}")
        trade_mode = row.get("trade_mode")
        if trade_mode:
            bullet_lines.append(f"Mode: {trade_mode}")
        note = "\n".join(f"- {line}" for line in bullet_lines) if bullet_lines else "- —"
        st.markdown(f"**{ts_text}** · {strategy_name}\n\n{note}")


def _render_plan_snapshot(feature_df: Optional[pd.DataFrame]) -> None:
    st.markdown("#### Active Plan Snapshot")
    fallback_df: Optional[pd.DataFrame] = None
    if feature_df is None or "strategy" not in feature_df.columns:
        fallback_df = _fallback_plan_from_positions()
        feature_df = fallback_df
    if feature_df is None:
        st.info("Decision feed not available yet.")
        return
    feature_df = feature_df[feature_df["strategy"] != "environment"] if not feature_df.empty else feature_df
    if feature_df is None or feature_df.empty:
        if fallback_df is None:
            fallback_df = _fallback_plan_from_positions()
        if fallback_df is not None and not fallback_df.empty:
            feature_df = fallback_df
        else:
            st.info("No active plan information recorded yet.")
            return
    latest = feature_df.sort_values("timestamp").groupby("strategy", as_index=False).last()
    for _, row in latest.iterrows():
        strategy_name = str(row.get("strategy") or "unknown").title()
        updated_at = _fmt_optional(row.get("timestamp"))
        ctx = _parse_context_blob(row.get("context"))

        lines = []
        lines.append(f"Updated {updated_at}")
        lines.append(
            f"CE { _fmt_optional(row.get('ce_strike'), digits=0) } Δ {_fmt_optional(row.get('ce_delta'), digits=3)} @ {_fmt_optional(row.get('ce_ltp'))}"
        )
        lines.append(
            f"PE { _fmt_optional(row.get('pe_strike'), digits=0) } Δ {_fmt_optional(row.get('pe_delta'), digits=3)} @ {_fmt_optional(row.get('pe_ltp'))}"
        )
        lines.append(f"Net credit ₹ {_fmt_optional(row.get('net_credit'))}")
        context_text = _format_context_summary(ctx)
        if context_text:
            lines.append(context_text)

        st.markdown(f"**{strategy_name}**\n\n" + "\n".join(f"- {line}" for line in lines))


def _render_environment_summary(feature_df: Optional[pd.DataFrame]) -> None:
    st.markdown("#### Market Regime")
    if feature_df is None or feature_df.empty:
        st.info("No regime data yet.")
        return
    if "strategy" not in feature_df.columns:
        st.info("No regime data yet.")
        return
    env_rows = feature_df.loc[feature_df["strategy"] == "environment"]
    if env_rows.empty:
        st.info("No regime data yet.")
        return
    latest = env_rows.sort_values("timestamp").iloc[-1]
    ctx = _parse_context_blob(latest.get("context"))
    state = ctx.get("market_state", {})
    candidates = ctx.get("strategy_candidates", [])

    cols = st.columns(3)
    cols[0].metric("Trend", str(state.get("trend", "—")).replace("_", " ").title())
    cols[1].metric("Volatility", str(state.get("volatility", "—")).replace("_", " ").title())
    cols[2].metric("IV Rank", _fmt_optional(state.get("iv_rank")))

    cols2 = st.columns(3)
    cols2[0].metric("Realized Vol", _fmt_optional(state.get("realized_vol")))
    cols2[1].metric("Drift", _fmt_optional(state.get("drift_pct")))
    cols2[2].metric("ATR %", _fmt_optional(state.get("atr_pct")))

    if candidates:
        df = pd.DataFrame(candidates)
        st.dataframe(df[[c for c in ["name", "score", "confidence", "sizing_hint", "rationale"] if c in df.columns]], width="stretch")
    else:
        st.caption("Strategy recommender idle – waiting for sufficient data.")


def _strategy_monitor_tab() -> None:
    st.markdown("### Strategy Monitor")
    blotter_df, summary = _load_blotter()
    feature_df = _load_feature_history(limit=250)

    left, right = st.columns([0.55, 0.45])
    with left:
        _render_blotter_panel(blotter_df, summary)
    with right:
        _render_environment_summary(feature_df)
        st.divider()
        _render_agent_activity(feature_df)
        st.divider()
        _render_plan_snapshot(feature_df)


def _render_capital_telemetry() -> None:
    st.markdown("#### Capital Telemetry")
    df = _load_equity_history(limit=1500)
    if df is None or df.empty or "equity_estimate" not in df.columns:
        st.info("No equity telemetry yet. Keep the agent running to build history.")
        return
    df = df.dropna(subset=["equity_estimate"])
    if df.empty:
        st.info("No equity telemetry yet. Keep the agent running to build history.")
        return
    equity_series = df["equity_estimate"].astype(float)
    current_equity = float(equity_series.iloc[-1])
    previous = float(equity_series.iloc[-2]) if len(equity_series) > 1 else None
    delta_label = None
    if previous is not None and previous != 0 and not np.isnan(previous):
        delta_val = current_equity - previous
        pct = (delta_val / previous) * 100 if previous else None
        delta_label = f"{delta_val:+,.0f}"
        if pct is not None and not np.isnan(pct):
            delta_label = f"{delta_val:+,.0f} ({pct:+.2f}%)"

    rolling_max = equity_series.cummax().replace(0, np.nan)
    drawdown_series = (equity_series / rolling_max) - 1.0
    max_drawdown_pct = float(drawdown_series.min()) * 100 if not drawdown_series.empty else None
    current_drawdown_pct = float(drawdown_series.iloc[-1]) * 100 if not drawdown_series.empty else None

    header_cols = st.columns(3)
    header_cols[0].metric("Current Equity", _fmt_rupee(current_equity, digits=0), delta=delta_label)
    header_cols[1].metric(
        "Max Drawdown",
        f"{max_drawdown_pct:.2f}%" if max_drawdown_pct is not None and not np.isnan(max_drawdown_pct) else "—",
    )
    header_cols[2].metric(
        "Current Drawdown",
        f"{current_drawdown_pct:.2f}%" if current_drawdown_pct is not None and not np.isnan(current_drawdown_pct) else "—",
    )

    timeframe_cols = st.columns(4)
    frames = [("Day Δ", 1), ("Week Δ", 7), ("Month Δ", 30), ("Year Δ", 365)]
    for col, (label, days) in zip(timeframe_cols, frames):
        change = _compute_equity_change(df, days)
        if not change:
            col.metric(label, "—", delta=None)
            continue
        delta_val, delta_pct = change
        col.metric(
            label,
            _fmt_rupee(delta_val, digits=0),
            delta=f"{delta_pct:+.2f}%" if delta_pct is not None else None,
        )

    chart_df = df.tail(720).copy()
    rolling_max_chart = chart_df["equity_estimate"].cummax().replace(0, np.nan)
    chart_df["drawdown_pct"] = ((chart_df["equity_estimate"] / rolling_max_chart) - 1.0) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["equity_estimate"],
            name="Equity",
            mode="lines",
            line=dict(width=2, color="#1f77b4"),
            hovertemplate="Time %{x|%Y-%m-%d %H:%M}<br>Equity ₹%{y:,.0f}<extra></extra>",
        )
    )
    if chart_df["drawdown_pct"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=chart_df["timestamp"],
                y=chart_df["drawdown_pct"],
                name="Drawdown %",
                mode="lines",
                line=dict(width=1.5, color="#d62728", dash="dot"),
                fill="tozeroy",
                opacity=0.35,
                yaxis="y2",
                hovertemplate="Time %{x|%Y-%m-%d %H:%M}<br>Drawdown %{y:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            yaxis=dict(title="Equity (₹)", rangemode="tozero"),
            yaxis2=dict(
                title="Drawdown %",
                overlaying="y",
                side="right",
                showgrid=False,
                rangemode="tozero",
            ),
        )
    else:
        fig.update_layout(yaxis=dict(title="Equity (₹)", rangemode="tozero"))

    fig.update_layout(
        xaxis=dict(title="Timestamp"),
        margin=dict(l=10, r=10, t=35, b=35),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")


def _observability_tab() -> None:
    st.markdown("### Observability")
    ss = st.session_state
    feature_df = _load_feature_history(limit=400)

    pid = ss.get("agent_pid")
    running_flag = bool(ss.get("agent_running"))
    alive_flag = _pid_alive(pid)
    mode = ss.get("agent_started_mode") or ss.get("trade_mode", "live")

    status_icon = "🟢" if running_flag and alive_flag else ("🟡" if pid and not alive_flag else "🔴")
    status_text = "Running" if running_flag and alive_flag else ("Stale PID" if pid and not alive_flag else "Stopped")
    st.markdown(f"**Agent Status:** {status_icon} {status_text}")
    st.caption(f"Mode: {mode} | PID: {pid or '—'}")

    cols = st.columns(3)
    with cols[0]:
        st.metric("PID alive", "Yes" if alive_flag else "No")
    with cols[1]:
        uptime_delta: Optional[timedelta] = None
        try:
            if running_flag and PID_FILE.exists():
                data = json.loads(PID_FILE.read_text())
                started_at = data.get("started_at")
                if started_at:
                    started_dt = datetime.fromisoformat(started_at)
                    uptime_delta = datetime.now() - started_dt
        except Exception:
            uptime_delta = None
        if uptime_delta:
            total_seconds = int(uptime_delta.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            uptime_str = f"{hours}h {minutes}m"
        else:
            uptime_str = "—"
        st.metric("Uptime", uptime_str)
    with cols[2]:
        st.metric("Credentials", "Verified" if ss.get("creds_verified") else "Missing")

    st.divider()
    _render_risk_alerts(feature_df)
    st.divider()
    action_cols = st.columns([1, 1, 1])
    with action_cols[0]:
        restart_disabled = not ss.get("creds_verified")
        if st.button("Restart Agent", key="obs_restart", disabled=restart_disabled, help="Stop then start the agent process."):
            stop_agent()
            time.sleep(0.5)
            ok = True
            if not ss.get("creds_verified"):
                ok = _verify_and_remember_creds()
            if ok:
                start_agent()
            st.rerun()
    with action_cols[1]:
        stop_disabled = not running_flag
        if st.button("Stop Agent", key="obs_stop", disabled=stop_disabled):
            stop_agent()
            st.rerun()
    with action_cols[2]:
        stale = bool(pid and not alive_flag)
        clear_disabled = not (PID_FILE.exists() and (stale or not running_flag))
        if st.button("Clear Stale PID", key="obs_clear_pid", disabled=clear_disabled, help="Remove PID file when the process is already gone."):
            _clear_stale_pid_file()
            st.rerun()

    st.divider()
    file_infos = [
        ("Agent Log", _collect_file_info(AGENT_LOG)),
        ("Trade Blotter", _collect_file_info(BLOTTER_CSV)),
        ("Feature Feed", _collect_file_info(FEATURE_LOG_CSV)),
        ("Equity Telemetry", _collect_file_info(EQUITY_LOG_CSV)),
        ("PID File", _collect_file_info(PID_FILE)),
    ]
    info_cols = st.columns(len(file_infos))
    for col, (label, info) in zip(info_cols, file_infos):
        with col:
            st.caption(label)
            st.write(f"Exists: {'Yes' if info['exists'] else 'No'}")
            st.write(f"Size: {_fmt_size(info.get('size'))}")
            st.write(f"Updated: {_age_string(info.get('updated'))}")

    st.divider()
    _render_capital_telemetry()
    st.divider()
    tail_lines = _read_tail(AGENT_LOG, 30)
    if tail_lines:
        st.caption("Recent Agent Log")
        st.code("".join(tail_lines), language="log")
    else:
        st.info("Agent log is empty or unavailable.")

# =============================================================================
# Settings tab
# =============================================================================
def _settings_tab() -> None:
    st.markdown("### Settings")
    current = _load_settings()

    c1, c2 = st.columns(2)
    with c1:
        max_legs = st.number_input("Max active strangles", 1, 10, int(current.get("max_legs", 1)))
        lot_size = st.number_input("Lot size (NIFTY)", 50, 1000, int(current.get("lot_size", 75)))
    with c2:
        leg_sl_pct = st.number_input("Stop-loss per leg (%)", 0.5, 10.0, float(current.get("leg_sl_pct", 2.5)))
        profit_pct = st.number_input("Profit booking (%)", 0.5, 10.0, float(current.get("profit_pct", 2.25)))
    warn_only = st.checkbox("Warn-only mode (log orders, no execution)", value=bool(current.get("warn_only", False)))

    with st.expander("Batman (short body) adjustments"):
        batman_enabled = st.checkbox(
            "Enable Batman monitoring",
            value=bool(current.get("batman_enabled", True)),
            help="Detects existing Batman structures, rolls stressed legs, and adds hedges automatically.",
        )
        bc1, bc2 = st.columns(2)
        with bc1:
            batman_delta_breach = st.number_input(
                "Delta breach threshold",
                0.05,
                0.9,
                float(current.get("batman_delta_breach", 0.30)),
                help="Roll a side when |delta| exceeds this value.",
            )
            batman_roll_distance = st.number_input(
                "Roll distance (pts)",
                50.0,
                500.0,
                float(current.get("batman_roll_distance", 150.0)),
                help="How far to move the body when rolling.",
            )
            batman_salvage = st.number_input(
                "Wing salvage LTP",
                0.0,
                20.0,
                float(current.get("batman_salvage_wing_ltp", 5.0)),
                help="Wings below this value can be considered for salvage/refresh (future use).",
            )
        with bc2:
            batman_premium_hard_x = st.number_input(
                "Premium multiple trigger",
                1.0,
                4.0,
                float(current.get("batman_premium_hard_x", 2.0)),
                help="Roll when current premium >= multiple × entry premium.",
            )
            batman_hedge_delta_max = st.number_input(
                "Hedge target delta",
                0.01,
                0.5,
                float(current.get("batman_hedge_delta_max", 0.12)),
                help="Select hedges with |delta| below this value.",
            )
            batman_hedge_price_max = st.number_input(
                "Max hedge cost",
                1.0,
                200.0,
                float(current.get("batman_hedge_price_max", 35.0)),
                help="Do not auto-buy hedges above this LTP.",
            )

    if st.button("💾 Save Settings"):
        SETTINGS_JSON.write_text(json.dumps({
            "max_legs": int(max_legs),
            "lot_size": int(lot_size),
            "leg_sl_pct": float(leg_sl_pct),
            "profit_pct": float(profit_pct),
            "warn_only": bool(warn_only),
            "batman_enabled": bool(batman_enabled),
            "batman_delta_breach": float(batman_delta_breach),
            "batman_premium_hard_x": float(batman_premium_hard_x),
            "batman_roll_distance": float(batman_roll_distance),
            "batman_hedge_delta_max": float(batman_hedge_delta_max),
            "batman_hedge_price_max": float(batman_hedge_price_max),
            "batman_salvage_wing_ltp": float(batman_salvage),
        }, indent=2))
        st.success("Settings saved.")


def _paper_pnl_tab() -> None:
    st.markdown("### Paper Ledger")
    df, summary = _load_blotter()

    if not summary and (df is None or df.empty):
        st.info("No paper trades recorded yet. Start the agent in paper mode to populate the blotter.")
        return

    total_orders = int(summary.get("total_orders", 0))
    executed_orders = int(summary.get("executed_orders", 0))
    warn_orders = int(summary.get("warn_only_orders", 0))
    credit = float(summary.get("credit_value", 0.0))
    debit = float(summary.get("debit_value", 0.0))
    net = float(summary.get("net_value", 0.0))

    c1, c2, c3 = st.columns(3)
    c1.metric("Orders Logged", total_orders, warn_orders, help="Total blotter entries (delta shows warn-only count)")
    c2.metric("Executed Orders", executed_orders)
    net_delta = credit if net >= 0 else -debit
    c3.metric("Net Credit", f"₹{net:,.2f}", f"₹{net_delta:,.2f}")

    if df is not None and not df.empty:
        st.markdown("#### Order Blotter")
        show_df = df.copy()
        if "timestamp" in show_df.columns:
            show_df["timestamp"] = pd.to_datetime(show_df["timestamp"], errors="coerce")
        for col in ("warn_only", "executed"):
            if col in show_df.columns:
                show_df[col] = pd.to_numeric(show_df[col], errors="coerce").astype('Int64')
        st.dataframe(show_df.fillna(""), width="stretch")

        executed_df = show_df[show_df.get("executed", 0) == 1].copy()
        if not executed_df.empty and "timestamp" in executed_df.columns:
            executed_df.sort_values("timestamp", inplace=True)
            executed_df["signed_notional"] = pd.to_numeric(executed_df.get("price"), errors="coerce").fillna(0.0) \
                * pd.to_numeric(executed_df.get("quantity"), errors="coerce").fillna(0.0) \
                * executed_df["side"].map({"SELL": 1.0, "BUY": -1.0}).fillna(0.0)
            executed_df["cumulative"] = executed_df["signed_notional"].cumsum()

            st.markdown("#### Cumulative Executed Credit")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=executed_df["timestamp"],
                y=executed_df["cumulative"],
                mode="lines+markers",
                name="Net Credit",
            ))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=40))
            st.plotly_chart(fig, width="stretch")

    if summary:
        st.markdown("#### Latest Summary")
        st.json(summary)

# =============================================================================
# Sidebar
# =============================================================================
def _sidebar() -> None:
    st.sidebar.markdown("### Broker Credentials")
    st.sidebar.text_input("Dhan Client ID", key="client_id", on_change=_mark_creds_dirty)
    st.sidebar.text_input("Dhan Access Token", key="access_token", type="password", on_change=_mark_creds_dirty)

    cred_status = "✅ Verified" if st.session_state.get("creds_verified") else "⚪️ Not verified"
    st.sidebar.caption(f"Credentials: {cred_status}")

    st.sidebar.markdown("### Agent Control")
    agent_running = st.session_state.get("agent_running", False)
    st.sidebar.radio(
        "Trade Mode",
        options=["paper", "live"],
        key="trade_mode",
        disabled=agent_running,
        help="Stop the agent before switching modes." if agent_running else None,
    )

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶️ Start Agent"):
        if _verify_and_remember_creds():
            start_agent()
            st.rerun()
    if c2.button("⏹ Stop Agent"):
        stop_agent()
        st.rerun()

    agent_mode = st.session_state.get("agent_started_mode")
    if agent_running:
        st.sidebar.caption(f"Agent: 🟢 Running ({agent_mode or st.session_state.get('trade_mode','live')})")
    else:
        st.sidebar.caption("Agent: 🔴 Stopped")

    st.sidebar.markdown("### Refresh")
    st.sidebar.slider("Refresh interval (sec)", 2, 30, key="refresh_sec", value=5)

# =============================================================================
# Main
# =============================================================================
def _rehydrate_dw_if_needed() -> None:
    """
    Make sure st.session_state['dw'] and ['creds_verified'] are live.
    This is resilient to cache clears / reruns.
    """
    ss = st.session_state
    if not ss.get("creds_verified"):
        return
    # Already valid?
    if ss.get("dw") is not None and ss.get("creds_verified") is True:
        return

    cid = (ss.get("client_id") or "").strip()
    tok = (ss.get("access_token") or "").strip()
    if not cid or not tok:
        return  # nothing to build from

    # Try constructing a fresh DhanWrapper
    dw = _new_dw()
    if not dw:
        ss["creds_verified"] = False
        ss["dw"] = None
        return

    ss["dw"] = dw
    ss["creds_verified"] = True
    _post_dw_setup(dw)

def main() -> None:
    st.set_page_config(page_title="Algo Agent Dashboard", layout="wide")
    _init_state()
    _sync_agent_state_from_pidfile()
    _sidebar()
    ss = st.session_state

    if ss.get("agent_running") and not ss.get("creds_verified"):
        _verify_and_remember_creds()

    _rehydrate_dw_if_needed()

    tab_labels = ["Trade", "Positions", "Strategy Monitor", "Observability"]
    show_paper = ss.get("trade_mode") == "paper"
    if show_paper:
        tab_labels.append("Paper P&L")
    tab_labels.extend(["Agent Logs", "Settings"])

    tabs = st.tabs(tab_labels)

    idx = 0
    with tabs[idx]:
        st.markdown("## Funds Overview")
        _funds_cards(st.session_state.get("funds", {}))
        st.divider()
        _nifty_tile()
    idx += 1

    with tabs[idx]:
        _positions_tab(st.session_state.get("dw"))
    idx += 1

    with tabs[idx]:
        _strategy_monitor_tab()
    idx += 1

    with tabs[idx]:
        _observability_tab()
    idx += 1

    if show_paper:
        with tabs[idx]:
            _paper_pnl_tab()
        idx += 1

    with tabs[idx]:
        _agent_logs_tab()
    idx += 1

    with tabs[idx]:
        _settings_tab()

    # gentle pause so the UI cadence feels steady with refresh slider
    time.sleep(max(0.1, float(st.session_state.get("refresh_sec", 5))))

if __name__ == "__main__":
    main()
